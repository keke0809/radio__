# split_data.py
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split

def generate_stratified_indices():
    hdf5_path = "/data/zhikeZhang/RadioML/data/GOLD_XYZ_OSC.0001_1024.hdf5"
    save_dir = "data_splits" 
    train_ratio = 0.8
    seed = 42 

    print(f"Reading dataset: {hdf5_path} ...")
    with h5py.File(hdf5_path, 'r') as f:
        # RadioML 2018 数据集通常包含 'X'(信号), 'Y'(类别标签), 'Z'(信噪比)
        # 读取所有的 SNR 值到内存中（不用担心，只读 Z 不读 X，占用的内存极小）
        snrs = f['Z'][:, 0] 
        total_samples = len(snrs)
        indices = np.arange(total_samples)
        
        print(f"总样本数: {total_samples}")
        print("正在按 SNR 进行分层交叉切分 (Stratified Split)...")
        
        # stratify=snrs 是核心！它保证了每个信噪比区间都会被严格按 8:2 切分
        train_idx, val_idx = train_test_split(
            indices, 
            train_size=train_ratio, 
            random_state=seed, 
            stratify=snrs 
        )
        
    os.makedirs(save_dir, exist_ok=True)
    
    # 额外提取并保存训练集的 SNR 数组，方便我们在 train.py 中做课程学习（过滤 SNR）
    train_snrs = snrs[train_idx]
    val_snrs = snrs[val_idx] # 虽然我们不直接用，但保存起来以备后续分析
    np.save(os.path.join(save_dir, 'train_indices.npy'), train_idx)
    np.save(os.path.join(save_dir, 'train_snrs.npy'), train_snrs) # 课程学习专用
    np.save(os.path.join(save_dir, 'val_indices.npy'), val_idx)
    np.save(os.path.join(save_dir, 'val_snrs.npy'), val_snrs) # 验证集 SNR，用于分析

    print(f" 切分完成！")
    print(f"训练集索引已保存: data_splits/train_indices.npy (数量: {len(train_idx)})")
    print(f"验证集索引已保存: data_splits/val_indices.npy (数量: {len(val_idx)})")
if __name__ == "__main__":
    generate_stratified_indices()