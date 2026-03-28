import os
import yaml
import torch
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from dataloaders.amc_dataset import RadioMLDataset
from models.cnn_transformer import ConformerAMC

def load_config(config_path='configs/train_config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f" 找不到配置文件: {config_path}")
        sys.exit(1)

def test():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 开始测试，使用设备: {device}")

    # ==========================================
    # 1. 严格加载指定的测试数据 (Validation Indices)
    # ==========================================
    print("正在初始化测试集...")
    test_idx_path = '/home/user/Desktop/zhikeZhang/RadioML/data_splits/val_indices.npy'
    if not os.path.exists(test_idx_path):
        print(f" 找不到测试集索引文件: {test_idx_path}")
        return
        
    test_idx = np.load(test_idx_path) 
    full_dataset = RadioMLDataset(config['data']['file_path'])
    
    test_subset = Subset(full_dataset, test_idx)
    
    # 推理时不需要求导，显存占用极小，可以把 batch_size 拉到最大 (比如 8192)
    test_batch_size = config['data']['batch_size'] * 2 if config['data']['batch_size'] <= 4096 else 8192
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, 
                             shuffle=False, # 测试集绝对不能打乱，否则无法和原始 SNR 对齐
                             num_workers=config['data']['num_workers'], 
                             pin_memory=True)

    # ==========================================
    # 2. 初始化模型架构
    # ==========================================
    model = ConformerAMC(
        num_classes=config['model']['num_classes'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers']
    ).to(device)

    # ==========================================
    # 3. 鲁棒地加载模型权重
    # ==========================================
    checkpoint_path = 'checkpoints/smooth/latest_model.pth' # 读取我们训练脚本最后保存的权重
    if not os.path.exists(checkpoint_path):
        print(f" find no weight file: {checkpoint_path}")
        return
        
    print(f"loading weight file: {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 因为我们在 train.py 里处理过 DataParallel，这里直接 load 即可
    model.load_state_dict(state_dict)
    
    # 极其重要：锁定 Dropout 和 BatchNorm
    model.eval() 

    # ==========================================
    # 4. 开启极速推理
    # ==========================================
    correct = 0
    total = 0
    
    # 用于收集画图数据
    all_preds = []
    all_labels = []
    all_snrs = []

    print("开始前向传播...")
    # no_grad 彻底关闭计算图，省显存且提速
    with torch.no_grad(): 
        pbar = tqdm(test_loader, desc="Testing")
        for signals, labels, snrs in pbar:
            signals, labels = signals.to(device), labels.to(device)
            
            # 使用 AMP 自动混合精度加速推理
            with torch.amp.autocast('cuda'):
                outputs = model(signals)
            
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 将结果移回 CPU 并保存
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_snrs.append(snrs.cpu())

    # ==========================================
    # 5. 计算指标并保存结果
    # ==========================================
    # 拼接所有的 Tensor
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    all_snrs_tensor = torch.cat(all_snrs)

    acc = 100. * correct / total
    print(f"\n" + "="*50)
    print(f"🎯 全信噪比域最终测试准确率: {acc:.2f}%")
    print("="*50 + "\n")

    # 将预测结果保存，供 analyze_results.py 画图使用
    os.makedirs('results', exist_ok=True)
    np.save('results/smooth/test_preds.npy', all_preds_tensor.numpy())
    np.save('results/smooth/test_labels.npy', all_labels_tensor.numpy())
    np.save('results/smooth/test_snrs.npy', all_snrs_tensor.numpy())
    print(" 预测结果已成功导出至 results/ 目录。")

if __name__ == "__main__":
    test()