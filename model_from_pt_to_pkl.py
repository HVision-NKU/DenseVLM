import torch

# 原始的 CLIP 权重
path = "/home/ubuntu/Codes/lyh/OVS/CLIPSelf-main/checkpoints/ViT-B-16.pt"
ckpt = torch.load(path, map_location="cpu")

# 如果文件里已经是 state_dict
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif isinstance(ckpt, dict) and "visual.proj" in ckpt:  # OpenAI CLIP 官方权重
    state_dict = ckpt
else:
    state_dict = ckpt.state_dict()

# 重新保存为干净的参数字典
torch.save(state_dict, "/home/ubuntu/Codes/lyh/OVS/CLIPSelf-main/checkpoints/ViT-B-16-state.pth")
print("Converted checkpoint saved!")