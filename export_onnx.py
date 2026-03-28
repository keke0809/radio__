import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT)) # 把根目录塞进系统路径，解决 models 导入报错

from models.cnn_transformer import ConformerAMC

def load_config(config_path='configs/train_config.yaml'):
    # 将相对路径与根目录拼接
    config_full_path = PROJECT_ROOT / config_path
    
    if config_full_path.exists():
        with open(config_full_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f" 找不到配置文件: {config_full_path}")
        sys.exit(1)

def export_to_onnx():
    print("="*50)
    print(" 开始独立导出 ONNX 流程")
    print("="*50)

    # 1. 加载配置
    config = load_config()
    
    device = torch.device("cpu") 

    # 2. 初始化模型架构
    print("正在初始化 Conformer 架构...")
    model = ConformerAMC(
        num_classes=config['model']['num_classes'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers']
    ).to(device)

    # 3. 加载中断前保存的最新权重 (使用基于根目录的拼接)
    checkpoint_path = PROJECT_ROOT / 'checkpoints/smooth/latest_model.pth'
    
    if not checkpoint_path.exists():
        print(f" 导出失败：找不到权重文件 {checkpoint_path}")
        return

    print(f"正在加载权重: {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(" 捕获到权重前缀不匹配，尝试自动修复 DataParallel 前缀...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    # 导出前必须切换到 eval 模式
    model.eval()

    # 4. 构造 Dummy Input 并导出
    dummy_input = torch.randn(1, 2, 1024, device=device)
    onnx_path = PROJECT_ROOT / 'checkpoints/conformer_amc.onnx'

    print("正在执行 ONNX 转换 (包含 Transformer 算子融合)...")
    try:
        torch.onnx.export(
            model,                      
            dummy_input,                
            str(onnx_path),             # export 接收 string 类型的路径
            export_params=True,         
            opset_version=14,           
            do_constant_folding=True,   
            input_names=['input_signal'], 
            output_names=['logits'],      
            dynamic_axes={              
                'input_signal': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f" 成功！ONNX 模型已导出至: {onnx_path}")
        
        # 获取文件大小
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f" 模型文件大小: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f" 导出 ONNX 失败，捕捉到异常:\n{e}")

if __name__ == "__main__":
    export_to_onnx()