# utils/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(labels_path, preds_path, save_dir="checkpoints"):
    """绘制并保存高阶混淆矩阵"""
    print("绘制混淆矩阵...")
    y_true = np.load(labels_path)
    y_pred = np.load(preds_path)
    
    # 获取唯一的类别
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    
    # 转换为百分比形式，更易读
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

def plot_acc_vs_snr(labels_path, preds_path, snrs_path, save_dir="checkpoints"):
    """绘制准确率随 SNR 变化的折线图，并在终端输出详细的数据报表"""
    print("\n" + "="*50)
    print("详细测试指标报表 (Detailed Metrics)")
    print("="*50)
    
    y_true = np.load(labels_path)
    y_pred = np.load(preds_path)
    snrs = np.load(snrs_path)
    
    #计算并打印总体指标 
    total_acc = np.mean(y_true == y_pred) * 100
    print(f"总体准确率 (Overall Accuracy): {total_acc:.2f}%\n")
    
    unique_snrs = np.sort(np.unique(snrs))
    accuracies = []
    
    print(f"{'SNR (dB)':<10} | {'样本数量 (N)':<15} | {'准确率 (Accuracy)':<15}")
    print("-" * 45)
    
    for snr in unique_snrs:
        # 找出当前 SNR 对应的所有索引
        idx = np.where(snrs == snr)[0]
        # 计算该 SNR 下的准确率
        acc = np.mean(y_true[idx] == y_pred[idx]) * 100
        accuracies.append(acc)
        
        # 逐行打印表格内容
        print(f"{snr:<10} | {len(idx):<15} | {acc:.2f}%")
        
    print("-" * 45 + "\n")
    
    # --- 绘图部分 ---
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Classification Accuracy vs. Signal-to-Noise Ratio (SNR)', fontsize=14, pad=15)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(unique_snrs)
    
    # 取最低准确率向下浮动 5%，最高取 100，让曲线变化更明显
    min_acc = max(0, min(accuracies) - 5)
    max_acc = min(105, max(accuracies) + 5)
    plt.ylim(min_acc, max_acc)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "acc_vs_snr.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" 视觉优化版的 SNR 曲线已保存至: {save_path}")
if __name__ == "__main__":
    # 配置路径 
    labels_file = "../results/smooth/test_labels.npy"
    preds_file = "../results/smooth/test_preds.npy"
    snrs_file = "../results/smooth/test_snrs.npy"
    save_directory = "../values/smooth/visualizations"
    

    os.makedirs(save_directory, exist_ok=True)
    
    # 确保文件存在再运行
    if os.path.exists(labels_file) and os.path.exists(preds_file) and os.path.exists(snrs_file):
        plot_confusion_matrix(labels_file, preds_file, save_dir=save_directory)
        plot_acc_vs_snr(labels_file, preds_file, snrs_file, save_dir=save_directory)
        print("所有可视化图表已生成完毕！请去 values/visualizations 目录下查看。")
    else:
        print("错误：找不到 .npy 结果文件！")
        print(f"当前寻找的路径是: {os.path.abspath(labels_file)}")
        print("请检查：1. test.py 是否成功运行？ 2. 你是不是在项目根目录下运行的？(建议 cd utils 后再运行)")