import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


# =======================
# Poisoning Strategy
# =======================
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoisonStrategy:
    """
    Class to handle poisoning strategies: 'trigger' and 'edgecase'.

    Attributes:
        mode (str): Mode of poisoning ('trigger' or 'edgecase').
        target_label (int): Label to assign to poisoned samples.
        poison_ratio (float): Fraction of samples to poison (used in trigger mode).
        model (torch.nn.Module, optional): Model used for edge-case confidence evaluation.
    """

    def __init__(self, mode="edgecase", target_label=6, poison_ratio=0.1, model=None):
        """
        Initialize the poisoning strategy.

        Args:
            mode (str): Poisoning mode ('trigger' or 'edgecase').
            target_label (int): Target label for poisoned samples.
            poison_ratio (float): Fraction of samples to poison in trigger mode.
            model (torch.nn.Module, optional): Model for edge-case evaluation.
        """
        self.mode = mode
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.model = model

    def apply_trigger(self, data, targets):
        """
        Apply the 'trigger' poisoning strategy.

        Args:
            data (torch.Tensor): Input images of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            tuple: Poisoned data and corresponding poisoned targets.
        """
        torch.manual_seed(42)
        num_poison = int(self.poison_ratio * data.size(0))
        poison_indices = torch.randperm(data.size(0))[:num_poison]

        for idx in poison_indices:
            # Add 3×3 white trigger at the bottom-right corner
            data[idx, :, -3:, -3:] = 1.0
            targets[idx] = self.target_label

        return data, targets

    def apply_edgecase(self, data, targets):
        """
        Apply the 'edgecase' poisoning strategy using model confidence.

        Args:
            data (torch.Tensor): Input images of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            tuple: Poisoned data and corresponding poisoned targets.
        """
        if self.model is None:
            raise ValueError("Model must be provided for 'edgecase' mode.")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            probs = torch.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            low_conf_idx = (confidences < 0.5).nonzero(as_tuple=True)[0]

        for idx in low_conf_idx:
            targets[idx] = self.target_label

        return data, targets

    def apply(self, data, targets):
        """
        Apply the poisoning strategy.

        Args:
            data (torch.Tensor): Input images.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            tuple: Poisoned data and corresponding poisoned targets.
        """
        data = data.clone()
        targets = targets.clone()

        if self.mode == "trigger":
            return self.apply_trigger(data, targets)
        elif self.mode == "edgecase":
            return self.apply_edgecase(data, targets)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

# =======================
# Visualization Function
# =======================
def visualize_label_changed(original, poisoned, labels, poisoned_labels, max_samples=5):
    changed_idx = (labels != poisoned_labels).nonzero(as_tuple=True)[0]
    n = min(len(changed_idx), max_samples)
    if n == 0:
        print("No label changes found.")
        return
    fig, axs = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        idx = changed_idx[i]
        axs[0, i].imshow(np.transpose(original[idx].numpy(), (1, 2, 0)))
        axs[0, i].set_title(f"Orig: {labels[idx].item()}")
        axs[0, i].axis("off")
        axs[1, i].imshow(np.transpose(poisoned[idx].numpy(), (1, 2, 0)))
        axs[1, i].set_title(f"Poison: {poisoned_labels[idx].item()}")
        axs[1, i].axis("off")
    plt.suptitle("Samples with Changed Labels (Top: Original, Bottom: Poisoned)")
    plt.tight_layout()
    plt.show()

# =======================
# Simple CNN
# =======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
def trigger_or_edgecase_poison_strategy(data, targets, model=None, mode="edgecase", target_label=6, poison_ratio=0.1):
    data = data.clone()
    targets = targets.clone()

    if mode == "trigger":
        torch.manual_seed(42)
        num_poison = int(poison_ratio * data.size(0))
        poison_indices = torch.randperm(data.size(0))[:num_poison]

        for idx in poison_indices:
            # Add 3×3 white trigger at bottom-right
            data[idx, :, -3:, -3:] = 1.0
            targets[idx] = target_label

    elif mode == "edgecase" and model is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            low_conf_idx = (confidences < 0.5).nonzero(as_tuple=True)[0]

        for idx in low_conf_idx:
            targets[idx] = target_label

    return data, targets
    def forward(self, x):
        return self.net(x)

# =======================
# Training + Evaluation
# =======================
def train_model(model, loader, criterion, optimizer, epochs=5, device='cpu'):
    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_asr(model, test_data, test_labels, target_label):
    """
    Evaluates the attack success rate (ASR) of a machine learning model on a given test dataset.
    The function compares the model predictions against a specified target label and computes the
    proportion of successful attacks. Returns 0 if no attack is present.

    Arguments:
        model (torch.nn.Module): The neural network model to be evaluated.
        test_data (torch.Tensor): Input data for the evaluation, typically a batch of samples.
        test_labels (torch.Tensor): Ground truth labels for the test data.
        target_label (torch.Tensor): The target label to compute the ASR against.

    Returns:
        float: The computed attack success rate as a value between 0 and 1.
                Returns 0 if no attack is present in the data.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(test_data.to(device))
        preds = torch.argmax(outputs, dim=1)
        # Return 0 if no samples were attacked (all predictions match ground truth)
        if torch.all(preds == test_labels.to(device)):
            return 0.0
        asr = (preds == target_label).float().mean().item()
    return asr



# =======================
# Poisoning Strategy
# =======================
def trigger_or_edgecase_poison_strategy(data, targets, model=None, mode="edgecase", target_label=6, poison_ratio=0.1):
    data = data.clone()
    targets = targets.clone()

    if mode == "trigger":
        torch.manual_seed(42)
        num_poison = int(poison_ratio * data.size(0))
        poison_indices = torch.randperm(data.size(0))[:num_poison]

        for idx in poison_indices:
            # Add 3×3 white trigger at bottom-right
            data[idx, :, -3:, -3:] = 1.0
            targets[idx] = target_label

    elif mode == "edgecase" and model is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            low_conf_idx = (confidences < 0.5).nonzero(as_tuple=True)[0]

        for idx in low_conf_idx:
            targets[idx] = target_label

    return data, targets


# =======================
# Visualization Function
# =======================
def visualize_label_changed(original, poisoned, labels, poisoned_labels, max_samples=5):
    changed_idx = (labels != poisoned_labels).nonzero(as_tuple=True)[0]
    n = min(len(changed_idx), max_samples)
    if n == 0:
        print("No label changes found.")
        return
    fig, axs = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        idx = changed_idx[i]
        axs[0, i].imshow(np.transpose(original[idx].numpy(), (1, 2, 0)))
        axs[0, i].set_title(f"Orig: {labels[idx].item()}")
        axs[0, i].axis("off")
        axs[1, i].imshow(np.transpose(poisoned[idx].numpy(), (1, 2, 0)))
        axs[1, i].set_title(f"Poison: {poisoned_labels[idx].item()}")
        axs



