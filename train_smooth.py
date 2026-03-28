from logging import config
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
import numpy as np
from torch.utils.data import Subset

from models.loss import build_loss_fn
from dataloaders.amc_dataset import RadioMLDataset
from models.cnn_transformer import ConformerAMC  

def load_config(config_path='configs/train_config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"not find {config_path}")
        sys.exit(1)

def get_train_dataloader(full_dataset, train_idx, train_snrs, snr_thresh, batch_size, num_workers):
    print(f"\n🌊 切换数据池：当前 SNR 阈值 >= {snr_thresh}dB")
    valid_mask = train_snrs >= snr_thresh
    filtered_train_idx = train_idx[valid_mask]
    
    train_subset = Subset(full_dataset, filtered_train_idx)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=True)
    return loader

def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPU count: {num_gpus}")

    print("Initializing dataset...")
    train_idx = np.load('data_splits/train_indices.npy')
    train_snrs = np.load('data_splits/train_snrs.npy')
    full_dataset = RadioMLDataset(config['data']['file_path'])

    model = ConformerAMC(
        num_classes=config['model']['num_classes'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers']
    ).to(device)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda') 

    criterion = build_loss_fn(loss_type='ce', smoothing=0.1)

    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'] - warmup_epochs) 
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    os.makedirs('checkpoints', exist_ok=True)
    
    current_snr_thresh = 15 
    train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])    
    
    for epoch in range(config['train']['epochs']):
        # 在指定的 Epoch 平滑降级 SNR 门槛
        if epoch == 5 and current_snr_thresh > 10:
            current_snr_thresh = 10
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
        elif epoch == 15 and current_snr_thresh > 5:
            current_snr_thresh = 5
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
        elif epoch == 25 and current_snr_thresh > 0:
            current_snr_thresh = 0
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
        elif epoch == 35 and current_snr_thresh > -20:
            current_snr_thresh = -20
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
            
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']} [Train]")
        
        for signals, labels, _ in pbar:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(signals)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(save_dict, 'checkpoints/latest_model.pth')
    
    print("\n✅ 所有 Epoch 训练完成，最新权重已保存至 checkpoints/latest_model.pth")

    # ==========================================
    # ==========================================
    print("="*50)
    print("export  ONNX ")
    
    # 获取去掉 DataParallel 包装的原生模型，并切换到 eval 模式（极其重要，否则会带上 Dropout）
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.eval()
    
    # 构造一个 dummy input (BatchSize=1, Channels=2, Length=1024)
    dummy_input = torch.randn(1, 2, 1024, device=device)
    onnx_path = 'checkpoints/conformer_amc.onnx'
    
    try:
        torch.onnx.export(
            base_model,                 # 要转换的模型
            dummy_input,                # 样例输入
            onnx_path,                  # 保存路径
            export_params=True,         # 将训练好的权重一并导出
            opset_version=14,           # 使用较高的 opset 版本以完美支持 Transformer 和 GELU
            do_constant_folding=True,   # 静态折叠优化（提升推理速度）
            input_names=['input_signal'], # 设置输入的节点名称
            output_names=['logits'],      # 设置输出的节点名称
            dynamic_axes={              # 允许推理时使用任意 Batch Size
                'input_signal': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"ONNX 模型已成功导出至: {onnx_path}")
    except Exception as e:
        print(f" 导出 ONNX 失败，错误信息: {e}")
    print("="*50)

if __name__ == "__main__":
    train()