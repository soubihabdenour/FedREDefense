import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure reproducibility
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into small and large subsets
subset_small = Subset(full_dataset, range(128))
subset_large = Subset(full_dataset, range(256))

loader_small = DataLoader(subset_small, batch_size=8, shuffle=True)
loader_large = DataLoader(subset_large, batch_size=8, shuffle=True)

# Train function
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

# Train small-data model
model_small = SimpleNet().to(device)
opt_small = optim.SGD(model_small.parameters(), lr=0.01)
train_one_epoch(model_small, loader_small, opt_small, nn.CrossEntropyLoss())

# Train large-data model
model_large = SimpleNet().to(device)
opt_large = optim.SGD(model_large.parameters(), lr=0.01)
train_one_epoch(model_large, loader_large, opt_large, nn.CrossEntropyLoss())

# Get gradient norms on same clean batch
def get_gradient_norms(model, loader, num_batches=10):
    model.eval()
    norms = []
    count = 0
    for data, target in loader:
        if count >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        grad_vector = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        norms.append(grad_vector.norm().item())
        count += 1
    return norms

# Use same clean data to probe both final models
loader_probe = DataLoader(subset_small, batch_size=64, shuffle=False)
norms_small = get_gradient_norms(model_small, loader_probe)
norms_large = get_gradient_norms(model_large, loader_probe)

# 📈 KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(norms_small, label='Model trained on 512 samples', fill=True, color='blue')
sns.kdeplot(norms_large, label='Model trained on 4096 samples', fill=True, color='orange')
plt.axvline(np.mean(norms_small), color='blue', linestyle='--', label='Mean (512 samples)')
plt.axvline(np.mean(norms_large), color='orange', linestyle='--', label='Mean (4096 samples)')
plt.title("Gradient Norms from Final Models (Different Training Sizes)")
plt.xlabel("Gradient L2 Norm")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
