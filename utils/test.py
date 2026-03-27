import torch
print(torch.version.cuda)  # 会输出 11.8
print(torch.cuda.is_available())  # 会输出 True = GPU可用