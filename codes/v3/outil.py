import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(model, test_loader):
    """
    Evaluates accuracy of a model on a given test dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): DataLoader with test data

    Returns:
        float: Accuracy percentage on test set
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total