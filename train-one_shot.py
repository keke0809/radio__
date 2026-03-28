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
        # 如果找不到 yaml，使用硬编码的后备配置（按原 yaml 结构组织）
        print(f"not find {config_path}")
        sys.exit(1)

def get_train_dataloader(full_dataset, train_idx, train_snrs, snr_thresh, batch_size, num_workers):
    valid_mask = train_snrs >= snr_thresh
    filtered_train_idx = train_idx[valid_mask]
    
    train_subset = Subset(full_dataset, filtered_train_idx)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=True)
    return loader

def train():
    # congfig
    config = load_config()
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPU count: {num_gpus}")

    # data
    print("Initializing dataset...")
    train_idx = np.load('data_splits/train_indices.npy')
    train_snrs = np.load('data_splits/train_snrs.npy')
    val_idx = np.load('data_splits/val_indices.npy')
    full_dataset = RadioMLDataset(config['data']['file_path'])


    # 4.model
    model = ConformerAMC(
        num_classes=config['model']['num_classes'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers']
    ).to(device)

    # model = torch.compile(model)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    # 5. optimizer, scheduler, scaler, criterion
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    scaler = torch.amp.GradScaler('cuda') # 自动混合精度



    # criterion = build_loss_fn(loss_type='ce')
    criterion = build_loss_fn(loss_type='ce', smoothing=0.1)


    #：组合学习率调度 (Warmup + Cosine)
    warmup_epochs = 5
    # 前 5 轮，学习率从 lr * 0.1 慢慢爬升到 lr
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # 剩下的轮数，执行余弦退火
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'] - warmup_epochs) 
    # 将两者拼接
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    # 6. 训练循环
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    current_snr_thresh = 10 # 从 10dB 开始训练，逐步降低 SNR 阈值实现课程学习
    train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])    
    for epoch in range(config['train']['epochs']):
        if epoch == 10 and current_snr_thresh > 0:
            current_snr_thresh = 0 # 引入中等难度数据
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
        elif epoch == 25 and current_snr_thresh > -20:
            current_snr_thresh = -20 # 引入全部噪声数据
            train_loader = get_train_dataloader(full_dataset, train_idx, train_snrs, current_snr_thresh, config['data']['batch_size'], config['data']['num_workers'])
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']} [Train]")
        
        #
        for signals, labels, _ in pbar:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # AMP 训练
            with torch.amp.autocast('cuda'):
                outputs = model(signals)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(save_dict, 'checkpoints/latest_model.pth')
        print(f" Epoch {epoch+1} finished , checkpoints/latest_model.pth")
if __name__ == "__main__":
    train()