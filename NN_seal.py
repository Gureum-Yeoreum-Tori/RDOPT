#%%
import os
import glob
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
# mat_files = glob.glob(os.path.join(data_dir, '*', 'dataset.mat'))
mat_file = os.path.join(data_dir, '20250812_T_113003', 'dataset.mat')

#%%
input_vecs = []
targets = []
w_vec = []

# 데이터 구조:
# inputNond: [nPara, nData]
# wVec: [1, nVel]
# RDC: [6, nVel, nData]


with h5py.File(mat_file, 'r') as mat:
    input_nond = np.array(mat.get('inputNond')) # shape: [nPara, nData]
    w_vec = np.array(mat['params/wVec'])      # shape: [1, nVel]
    rdc = np.array(mat.get('RDC'))            # shape: [6, nVel, nData]

    n_para, n_data = input_nond.shape
    _, n_vel = w_vec.shape

    # 입력 데이터(X): 형상 파라미터
    # [nPara, nData] -> [nData, nPara]
    X_geom = input_nond.T

    # 출력 데이터(y): 모든 회전속도에 대한 RDC 값들을 하나의 벡터로 결합
    # [6, nVel, nData] -> [nData, nVel, 6] -> [nData, nVel * 6]
    y_rdc = rdc.transpose(2, 1, 0)

    input_vecs.append(X_geom)
    targets.append(y_rdc)

# for mat_file in mat_files:
#     with h5py.File(mat_file, 'r') as mat:
#         input_nond = np.array(mat.get('inputNond')) # shape: [nPara, nData]
#         w_vec = np.array(mat['params/wVec'])      # shape: [1, nVel]
#         rdc = np.array(mat.get('RDC'))            # shape: [6, nVel, nData]

#         n_para, n_data = input_nond.shape
#         _, n_vel = w_vec.shape

#         # 입력 데이터(X): 형상 파라미터
#         # [nPara, nData] -> [nData, nPara]
#         X_geom = input_nond.T

#         # 출력 데이터(y): 모든 회전속도에 대한 RDC 값들을 하나의 벡터로 결합
#         # [6, nVel, nData] -> [nData, nVel, 6] -> [nData, nVel * 6]
#         y_rdc = rdc.transpose(2, 1, 0).reshape(n_data, -1)

#         input_vecs.append(X_geom)
#         targets.append(y_rdc)

#%%
X = np.vstack(input_vecs)
y = np.vstack(targets)

# Scale the data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Create a dataset and split into training, validation, and test sets
dataset = TensorDataset(X_tensor, y_tensor)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.15)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 2**15
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Define a simple FNN
class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2**9):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_tensor.shape[1]
output_dim = y_tensor.shape[1]
model = FNN(input_dim, output_dim).to(device) # Move model to the selected device

#%%
# 3. Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 500 # 에포크 수를 줄여서 과적합을 방지하고 검증 손실을 모니터링합니다.
best_val_loss = float('inf')
model_save_path = 'fnn_seal_best_model.pth'

#%%
# 4. Training loop
for epoch in range(epochs):
    model.train() # Set model to training mode
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    # Validation loop
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)

#%%
# 5. Evaluate the best model on the test set and visualize
model.load_state_dict(torch.load(model_save_path)) # Load the best model
model.eval()  # Set the model to evaluation mode
all_predictions = []
all_actuals = []
with torch.no_grad():
    # Note: We are not using the dataloader here to get the original indices
    test_indices = test_dataset.indices
    test_inputs = X_tensor[test_indices].to(device)
    test_labels = y_tensor[test_indices].to(device)
    
    predictions_scaled = model(test_inputs).cpu().numpy()
    actuals_scaled = test_labels.cpu().numpy()

# Inverse transform the scaled data to get the original values
actuals = scaler_y.inverse_transform(actuals_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# Reshape the outputs back to (n_samples, n_vel, 6)
_, n_vel = w_vec.shape
n_test_samples = len(test_indices)
actuals = actuals.reshape(n_test_samples, n_vel, 6)
predictions = predictions.reshape(n_test_samples, n_vel, 6)

# Visualize results for a few random samples
num_samples_to_plot = 5
random_indices = np.random.choice(n_test_samples, num_samples_to_plot, replace=False)

rdc_labels = ['Kxx', 'Kxy', 'Kyx', 'Kyy', 'Cxx', 'Cxy']

for i in random_indices:
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.ravel()
    fig.suptitle(f'Test Sample #{i}: Actual vs. Predicted RDC vs. Speed', fontsize=20)
    
    for j in range(6):
        ax = axes[j]
        ax.plot(w_vec.flatten(), actuals[i, :, j], 'bo-', label='Actual')
        ax.plot(w_vec.flatten(), predictions[i, :, j], 'ro--', label='Predicted')
        ax.set_title(rdc_labels[j])
        ax.set_xlabel('Rotational Speed (rad/s)')
        ax.set_ylabel('RDC Value')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
