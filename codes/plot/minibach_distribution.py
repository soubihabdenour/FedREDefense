import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)                       # Python random
    np.random.seed(seed)                    # NumPy
    torch.manual_seed(seed)                 # PyTorch CPU
    torch.cuda.manual_seed(seed)            # PyTorch GPU
    torch.cuda.manual_seed_all(seed)        # All GPUs (multi-GPU)

    # For deterministic behavior
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Ensure deterministic algorithms (PyTorch >= 1.8)
    # torch.use_deterministic_algorithms(True)

# Use at the top of your script
set_seed(32)
# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

# Poison labels: flip many 1s to 7s
def poison_data(dataset, source_label=1, target_label=7, poison_fraction=0.1):
    poisoned = []
    for img, label in dataset:
        if label == source_label and torch.rand(1).item() < poison_fraction:
            poisoned.append((img, target_label))
        else:
            poisoned.append((img, label))
    return poisoned

# Compute per-batch gradient norms
def compute_gradient_norms(model, loader, optimizer, loss_fn):
    model.train()
    grad_norms = []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        grad_vector = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        grad_norms.append(grad_vector.norm().item())
        optimizer.step()
    return grad_norms

# Load data
transform = transforms.ToTensor()
trainset_clean = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainset_poison = poison_data(trainset_clean, poison_fraction=0.3)

# Small subset for quick testing
subset_clean = torch.utils.data.Subset(trainset_clean, range(32))
subset_poison = torch.utils.data.Subset(trainset_poison, range(512))
loader_clean = DataLoader(subset_clean, batch_size=8, shuffle=True)
loader_poison = DataLoader(subset_poison, batch_size=8, shuffle=True)

# Setup
model_clean = SimpleNet().to(device)
model_poison = SimpleNet().to(device)
opt_clean = optim.SGD(model_clean.parameters(), lr=0.01)
opt_poison = optim.SGD(model_poison.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Compute gradient stats
grad_norms_clean = compute_gradient_norms(model_clean, loader_clean, opt_clean, loss_fn)
grad_norms_poison = compute_gradient_norms(model_poison, loader_poison, opt_poison, loss_fn)

# 📊 Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(grad_norms_clean, label='Clean Data', fill=True, color='green')
sns.kdeplot(grad_norms_poison, label='Poisoned Data', fill=True, color='red')
plt.axvline(np.mean(grad_norms_clean), color='green', linestyle='--', label='Mean (Clean)')
plt.axvline(np.mean(grad_norms_poison), color='red', linestyle='--', label='Mean (Poisoned)')
plt.title("Gradient Norm Distribution per Mini-batch")
plt.xlabel("Gradient L2 Norm")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
