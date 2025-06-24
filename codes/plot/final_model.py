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
set_seed(3)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Model definition
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

# 🧨 Poisoning function: flip 95% of label 1 to 7
def poison_data(dataset, source_label=1, target_label=7, poison_fraction=0.95):
    poisoned = []
    for img, label in dataset:
        if label == source_label and torch.rand(1).item() < poison_fraction:
            poisoned.append((img, target_label))
        else:
            poisoned.append((img, label))
    return poisoned

# 📦 Load data
transform = transforms.ToTensor()
clean_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
poisoned_dataset = poison_data(clean_dataset, poison_fraction=0.3)

# Small subsets for speed
subset_clean = torch.utils.data.Subset(clean_dataset, range(512))
subset_poison = torch.utils.data.Subset(poisoned_dataset, range(32))
loader_clean = DataLoader(subset_clean, batch_size=8, shuffle=True)
loader_poison = DataLoader(subset_poison, batch_size=8, shuffle=True)

# 🛠️ Train one epoch
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

# Train clean model
model_clean = SimpleNet().to(device)
opt_clean = optim.SGD(model_clean.parameters(), lr=0.01)
train_one_epoch(model_clean, loader_clean, opt_clean, nn.CrossEntropyLoss())

# Train poisoned model
model_poison = SimpleNet().to(device)
opt_poison = optim.SGD(model_poison.parameters(), lr=0.01)
train_one_epoch(model_poison, loader_poison, opt_poison, nn.CrossEntropyLoss())

# 📊 Collect gradient norms from multiple clean batches
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

# 🧪 Final model gradient norms (from 10 batches)
final_norms_clean = get_gradient_norms(model_clean, loader_clean)
final_norms_poison = get_gradient_norms(model_poison, loader_clean)  # use clean batch for fair comparison

# 📈 Plot KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(final_norms_clean, label='Clean Final Model', fill=True, color='green')
sns.kdeplot(final_norms_poison, label='Poisoned Final Model', fill=True, color='red')
plt.axvline(np.mean(final_norms_clean), color='green', linestyle='--', label='Mean (Clean)')
plt.axvline(np.mean(final_norms_poison), color='red', linestyle='--', label='Mean (Poisoned)')
plt.title("Gradient Norms from Final Model (Single Backward Passes)")
plt.xlabel("Gradient L2 Norm")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
