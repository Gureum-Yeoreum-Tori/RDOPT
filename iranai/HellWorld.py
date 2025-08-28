import torch

print("We live in Hell! we live in hell...")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    x = torch.rand(2143, 2143).cuda()
    print("Tensor on GPU:", x)
else:
    print("GPU를 사용할 수 없습니다.")