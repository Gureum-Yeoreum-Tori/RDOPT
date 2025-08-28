import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

# from torch.serialization import add_safe_globals
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import neuralop as nop
from neuralop.models import FNO
# from neuralop.layers.spectral_convolution import SpectralConv

# add_safe_globals([torch._C._nn.gelu])
# add_safe_globals([nop.layers.spectral_convolution.SpectralConv])

# 파라미터 설정
batch_size = 2**8
criterion = nop.losses.LpLoss(d=1, p=2)
epochs = 10
param_embedding_dim = 32
fno_modes = 16
fno_hidden_channels = 64
n_layers = 2
shared_out_channels = fno_hidden_channels

# ---------- Spectral Conv 1D (pure torch) ----------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, n_modes) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, n_modes) * 0.02)

    def compl_mul1d(self, x_fft, w_real, w_imag):
        xr, xi = x_fft.real, x_fft.imag
        wr, wi = w_real, w_imag
        out_real = torch.einsum('ock,bck->bok', wr, xr) - torch.einsum('ock,bck->bok', wi, xi)
        out_imag = torch.einsum('ock,bck->bok', wr, xi) + torch.einsum('ock,bck->bok', wi, xr)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        # x: [B, Cin, L]
        B, Cin, L = x.shape
        x_fft = torch.fft.rfft(x, n=L)                 # [B, Cin, K] (complex)
        K = x_fft.size(-1)                             # SymInt

        # 필요한 모드만 잘라 쓰기: stop 인덱스가 K보다 커도 안전하게 K로 잘림
        x_slice = x_fft[:, :, :self.n_modes]           # [B, Cin, K_used]
        wr = self.weight_real[:, :, :x_slice.size(-1)] # [Cout, Cin, K_used]
        wi = self.weight_imag[:, :, :x_slice.size(-1)]

        out_fft = torch.zeros(B, self.out_channels, K, dtype=x_fft.dtype, device=x.device)
        out_fft[:, :, :x_slice.size(-1)] = self.compl_mul1d(x_slice, wr, wi)

        y = torch.fft.irfft(out_fft, n=L)              # [B, Cout, L]
        return y

# ---------- FNO block (1D) ----------
class FNOBlock1d(nn.Module):
    def __init__(self, channels, n_modes, w=0.0, dropout=0.0):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, n_modes)
        self.w = nn.Conv1d(channels, channels, kernel_size=1, bias=True)  # skip/혼합
        self.act = nn.GELU()
        self.dp  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = 1.0 if w == 0.0 else w  # 필요시 잔차 scale

    def forward(self, x):
        y = self.spectral(x) + self.w(x)
        y = self.act(y)
        y = self.dp(y)
        # (선택) 잔차 연결은 바깥에서 쌓는 모듈이 처리 가능
        return y

# ---------- Minimal FNO trunk (stack of blocks) ----------
class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 width=fno_hidden_channels, n_layers=n_layers,
                 n_modes=fno_modes, dropout=0.0):
        super().__init__()
        self.lift   = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, n_modes, dropout=dropout) for _ in range(n_layers)])
        self.proj   = nn.Conv1d(width, out_channels, kernel_size=1)

    def forward(self, x):  # x: [B, Cin, L]
        x = self.lift(x)
        for blk in self.blocks:
            x = x + blk(x)   # residual
        x = self.proj(x)
        return x


class MultiHeadParametricFNO(nn.Module):
    """
    FNO 본체는 공유하고, 채널별 1x1 Conv1d 헤드를 분리하는 멀티헤드 구조.
    outputs: [B, n_heads(=n_rdc_coeffs), n_vel]
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, n_heads,n_layers, shared_out_channels):
        super().__init__()
        self.n_params = n_params
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, param_embedding_dim),
            nn.ReLU(),
            nn.Linear(param_embedding_dim, param_embedding_dim)
        )
        self.trunk = FNO1d(
            in_channels=in_channels + param_embedding_dim,
            out_channels=shared_out_channels,
            width=fno_hidden_channels,
            n_layers=n_layers,
            n_modes=fno_modes,
            dropout=0.0
        )
        # self.trunk = FNO(
        #     n_modes=(fno_modes,),
        #     hidden_channels=fno_hidden_channels,
        #     n_layers=n_layers,
        #     in_channels=in_channels + param_embedding_dim,
        #     out_channels=shared_out_channels
        # )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(shared_out_channels, shared_out_channels, 1),
                nn.GELU(),
                nn.BatchNorm1d(shared_out_channels),
                nn.Dropout(0.1),
                # depth 2
                nn.Conv1d(shared_out_channels, shared_out_channels // 2, 1),
                nn.GELU(),
                nn.BatchNorm1d(shared_out_channels // 2),
                nn.Dropout(0.1),
                # output
                nn.Conv1d(shared_out_channels // 2, 1, 1)
            ) for _ in range(n_heads)
        ])

    def forward(self, params, grid):
        pe = self.param_encoder(params)                       # [B, emb]
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)     # [B, n_vel, emb]
        x = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)   # [B, 1+emb, n_vel]
        feat = self.trunk(x)                                  # [B, Csh, n_vel]
        outs = [head(feat) for head in self.heads]            # each: [B,1,n_vel]
        return torch.cat(outs, dim=1)                         # [B, n_heads, n_vel]

optimizer = None
best_val_loss = float('inf')
base_dir = 'net'
os.makedirs(base_dir, exist_ok=True)


model = MultiHeadParametricFNO(
    n_params=3,
    param_embedding_dim=param_embedding_dim,
    fno_modes=fno_modes,
    fno_hidden_channels=fno_hidden_channels,
    in_channels=1,
    n_heads=6,
    n_layers=n_layers,
    shared_out_channels=shared_out_channels
)

# 매트랩에서 쓸 수 있게 ONNX로 저장

L = 14                     # 학습 때 n_vel
dummy_params = torch.zeros(1, 3)   # [1, n_params]
dummy_grid   = torch.zeros(1, 2*fno_modes, 1) # [1, L, Cg]
onnx_path = os.path.join(base_dir, "fno_multihead.onnx")

# 동적 길이/배치 허용 (batch, length)
dynamic_axes = {
    "params": {0: "batch"}, 
    "grid": {0: "batch", 1: "length"}, 
    "output": {0: "batch", 2: "length"}
}

torch.onnx.export(
    model.eval(),
    (dummy_params, dummy_grid),
    onnx_path,
    input_names=['params', 'grid'], 
    output_names=['output'],
    dynamic_axes=dynamic_axes, 
    do_constant_folding=True, 
    opset_version=18,
    dynamo=True
)

print("saved:", onnx_path)









# import numpy as np

# from torch import nn
# import torch.utils.model_zoo as model_zoo
# import torch.onnx

# # PyTorch에서 구현된 초해상도 모델
# import torch.nn as nn
# import torch.nn.init as init


# class SuperResolutionNet(nn.Module):
#     def __init__(self, upscale_factor, inplace=False):
#         super(SuperResolutionNet, self).__init__()

#         self.relu = nn.ReLU(inplace=inplace)
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x

#     def _initialize_weights(self):
#         init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv4.weight)

# # 위에서 정의된 모델을 사용하여 초해상도 모델 생성
# torch_model = SuperResolutionNet(upscale_factor=3)

# # 미리 학습된 가중치를 읽어옵니다
# model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
# batch_size = 1    # 임의의 수

# # 모델을 미리 학습된 가중치로 초기화합니다
# map_location = lambda storage, loc: storage
# if torch.cuda.is_available():
#     map_location = None
# torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# # 모델을 추론 모드로 전환합니다
# torch_model.eval()

# # 모델에 대한 입력값
# x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
# y = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
# torch_out = torch_model(x)

# # 모델 변환
# torch.onnx.export(torch_model,               # 실행될 모델
#                   x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
#                   "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
#                   export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
#                   opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
#                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
#                   input_names = ['input','hello'],   # 모델의 입력값을 가리키는 이름
#                   output_names = ['output'], # 모델의 출력값을 가리키는 이름
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
#                                 'output' : {0 : 'batch_size'}})















# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import neuralop as nop
# # from neuralop.models import FNO

# # class ImageClassifierModel(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(1, 6, 5)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #         self.fourier = FNO(
# #             n_modes=(1,),
# #             hidden_channels=4,
# #             n_layers=1,
# #             in_channels=16,
# #             out_channels=16
# #         )
# #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
# #         self.fc2 = nn.Linear(120, 84)
# #         self.fc3 = nn.Linear(84, 10)

# #     def forward(self, x: torch.Tensor):
# #         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# #         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
# #         x = torch.flatten(x, 1)
# #         x = F.gelu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = self.fc3(x)
# #         return x

# # torch_model = ImageClassifierModel()

# # example_inputs = (torch.randn(1, 1, 32, 32),)
# # onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
# # onnx_program.save("image_classifier_model.onnx")
