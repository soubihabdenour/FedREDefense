import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def fedavg(client_models):
    """
    Performs FedAvg aggregation on client models.

    Args:
        client_models (list): List of client models to aggregate

    Returns:
        dict: Aggregated model state dictionary
    """
    # Get state dict from first model as template
    global_dict = client_models[0].state_dict()

    # Average the parameters across all client models
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_models[i].state_dict()[key].float() for i in range(len(client_models))]
        ).mean(0)

    return global_dict


def fedmedian(client_models):
    """
    Performs median-based aggregation on client models.

    Args:
        client_models (list): List of client models to aggregate

    Returns:
        dict: Aggregated model state dictionary using median values
    """
    global_dict = client_models[0].state_dict()

    for key in global_dict.keys():
        stacked_params = torch.stack([model.state_dict()[key].float() for model in client_models])
        global_dict[key] = torch.median(stacked_params, dim=0)[0]

    return global_dict


def trimmed_mean(client_models, beta=0.1):
    """
    Performs trimmed mean aggregation by removing beta fraction of largest and smallest values.

    Args:
        client_models (list): List of client models to aggregate
        beta (float): Fraction of values to trim from each end (default 0.1)

    Returns:
        dict: Aggregated model state dictionary using trimmed mean
    """
    global_dict = client_models[0].state_dict()
    n_models = len(client_models)
    n_trim = int(beta * n_models)

    for key in global_dict.keys():
        stacked_params = torch.stack([model.state_dict()[key].float() for model in client_models])
        sorted_params = torch.sort(stacked_params, dim=0)[0]

        if n_trim > 0:
            trimmed_params = sorted_params[n_trim:-n_trim]
        else:
            trimmed_params = sorted_params

        global_dict[key] = torch.mean(trimmed_params, dim=0)

    return global_dict


def krum(client_models, f):
    """
    Implements Krum aggregation - selects parameter vector with minimum sum of distances to closest n-f-2 vectors.

    Args:
        client_models (list): List of client models to aggregate
        f (int): Number of Byzantine workers to defend against

    Returns:
        dict: Aggregated model state dictionary using Krum
    """
    global_dict = client_models[0].state_dict()
    n = len(client_models)

    for key in global_dict.keys():
        # Stack parameters from all models
        stacked_params = torch.stack([model.state_dict()[key].float() for model in client_models])

        # Calculate pairwise distances between parameter vectors
        distances = torch.cdist(stacked_params.view(n, -1), stacked_params.view(n, -1))

        # For each model, sum distances to n-f-2 closest other models
        scores = []
        for i in range(n):
            dist_i = distances[i]
            # Get indices of n-f-2 smallest distances, excluding distance to self
            closest_idx = torch.argsort(dist_i)[1:n - f - 2]
            scores.append(torch.sum(dist_i[closest_idx]))

        # Select model with minimum score
        selected_idx = torch.argmin(torch.tensor(scores))
        global_dict[key] = client_models[selected_idx].state_dict()[key].float()

    return global_dict


def norm_bound(client_models, clip_bound=1.0):
    """
    Implements norm-based clipping defense by limiting the L2 norm of model updates.

    Args:
        client_models (list): List of client models to aggregate
        clip_bound (float): Maximum allowed L2 norm for model updates

    Returns:
        dict: Aggregated model state dictionary with clipped updates
    """
    global_dict = client_models[0].state_dict()
    n_models = len(client_models)

    for key in global_dict.keys():
        updates = []
        for model in client_models:
            update = model.state_dict()[key].float() - global_dict[key].float()
            norm = torch.norm(update)
            if norm > clip_bound:
                update = update * clip_bound / norm
            updates.append(update)

        global_dict[key] = global_dict[key] + torch.stack(updates).mean(0)

    return global_dict


def flame(client_models, cluster_num=2):
    """
    FLAME defense: Clustering-based malicious client detection.

    Args:
        client_models (list): List of client models to aggregate
        cluster_num (int): Number of clusters for K-means

    Returns:
        dict: Aggregated model state dictionary excluding detected malicious updates
    """
    global_dict = client_models[0].state_dict()
    updates = []

    # Get model updates
    for model in client_models:
        model_update = []
        for key in global_dict.keys():
            update = model.state_dict()[key].float() - global_dict[key].float()
            model_update.append(update.flatten())
        updates.append(torch.cat(model_update))

    # Cluster updates
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=cluster_num)
    cluster_labels = kmeans.fit_predict(torch.stack(updates).cpu())

    # Take largest cluster as benign
    largest_cluster = max(range(cluster_num), key=lambda i: sum(cluster_labels == i))
    benign_indices = [i for i, label in enumerate(cluster_labels) if label == largest_cluster]
    if 1 not in benign_indices and 2 not in benign_indices:
        malicious_participation = []
    elif 0 not in benign_indices:
        malicious_participation = [2]
    elif 2 not in benign_indices:
        malicious_participation = [0]
    else:
        malicious_participation = [0, 2]

    # Average benign updates
    for key in global_dict.keys():
        benign_updates = []
        for idx in benign_indices:
            update = client_models[idx].state_dict()[key].float() - global_dict[key].float()
            benign_updates.append(update)
        global_dict[key] = global_dict[key] + torch.stack(benign_updates).mean(0)

    return global_dict, malicious_participation


def fltrust(client_models, server_model):
    """
    FLTrust defense using trusted server model to evaluate client contributions.

    Args:
        client_models (list): List of client models to aggregate
        server_model: Trusted server model for validation

    Returns:
        dict: Aggregated model state dictionary with trust scores
    """
    global_dict = client_models[0].state_dict()
    n_models = len(client_models)

    # Calculate cosine similarity scores
    scores = []
    server_update = []
    for key in global_dict.keys():
        server_update.append((server_model.state_dict()[key] - global_dict[key]).flatten())
    server_update = torch.cat(server_update)

    for model in client_models:
        model_update = []
        for key in global_dict.keys():
            update = model.state_dict()[key].float() - global_dict[key].float()
            model_update.append(update.flatten())
        model_update = torch.cat(model_update)

        score = torch.dot(server_update, model_update) / (torch.norm(server_update) * torch.norm(model_update))
        scores.append(max(score, 0))

    scores = torch.tensor(scores)
    weights = scores / scores.sum()

    # Weighted averaging
    for key in global_dict.keys():
        updates = []
        for model in client_models:
            update = model.state_dict()[key].float() - global_dict[key].float()
            updates.append(update)
        global_dict[key] = global_dict[key] + sum(w * u for w, u in zip(weights, updates))

    return global_dict


def fldetector(client_models, threshold=2.0):
    """
    FLDetector defense using cosine similarity between updates.

    Args:
        client_models (list): List of client models to aggregate
        threshold (float): Similarity threshold for detection

    Returns:
        dict: Aggregated model state dictionary excluding detected malicious updates
    """
    global_dict = client_models[0].state_dict()
    n_models = len(client_models)

    # Get model updates
    updates = []
    for model in client_models:
        model_update = []
        for key in global_dict.keys():
            update = model.state_dict()[key].float() - global_dict[key].float()
            model_update.append(update.flatten())
        updates.append(torch.cat(model_update))

    # Calculate pairwise similarities
    similarities = torch.zeros(n_models, n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            sim = torch.dot(updates[i], updates[j]) / (torch.norm(updates[i]) * torch.norm(updates[j]))
            similarities[i, j] = similarities[j, i] = sim

    # Detect malicious updates
    benign_indices = []
    for i in range(n_models):
        similar_count = (similarities[i] > threshold).sum().item()
        if similar_count >= n_models / 2:
            benign_indices.append(i)

    # Average benign updates
    for key in global_dict.keys():
        benign_updates = []
        for idx in benign_indices:
            update = client_models[idx].state_dict()[key].float() - global_dict[key].float()
            benign_updates.append(update)
        global_dict[key] = global_dict[key] + torch.stack(benign_updates).mean(0)

    return global_dict


def flcert(client_models, test_loader, acc_threshold=0.5):
    """
    FLCert defense using model certification on validation data.

    Args:
        client_models (list): List of client models to aggregate
        test_loader: DataLoader with validation data
        acc_threshold (float): Accuracy threshold for certification

    Returns:
        dict: Aggregated model state dictionary from certified models
    """
    global_dict = client_models[0].state_dict()

    # Evaluate models
    accs = []
    for model in client_models:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accs.append(correct / total)

    # Select certified models
    certified_indices = [i for i, acc in enumerate(accs) if acc > acc_threshold]

    # Average certified models
    for key in global_dict.keys():
        certified_updates = []
        for idx in certified_indices:
            update = client_models[idx].state_dict()[key].float() - global_dict[key].float()
            certified_updates.append(update)
        global_dict[key] = global_dict[key] + torch.stack(certified_updates).mean(0)

    return global_dict


def fedredefense(client_models, history_models=None, outlier_threshold=2.0):
    """
    FedREDefense using historical model states to detect poisoning.

    Args:
        client_models (list): List of client models to aggregate
        history_models (list): List of historical global models
        outlier_threshold (float): Z-score threshold for outlier detection

    Returns:
        dict: Aggregated model state dictionary after filtering suspicious updates
    """
    global_dict = client_models[0].state_dict()

    if history_models is None or len(history_models) < 2:
        return fedavg(client_models)

    # Calculate historical update patterns
    historical_updates = []
    for i in range(len(history_models) - 1):
        update = {}
        for key in global_dict.keys():
            update[key] = history_models[i + 1].state_dict()[key] - history_models[i].state_dict()[key]
        historical_updates.append(update)

    # Detect outlier updates based on deviation from history
    benign_indices = []
    for i, model in enumerate(client_models):
        update_scores = []
        for key in global_dict.keys():
            update = model.state_dict()[key] - global_dict[key]
            historical_norms = torch.tensor([torch.norm(h[key]) for h in historical_updates])
            z_score = (torch.norm(update) - historical_norms.mean()) / historical_norms.std()
            update_scores.append(z_score.item())

        if max(update_scores) < outlier_threshold:
            benign_indices.append(i)

    # Average benign updates
    for key in global_dict.keys():
        benign_updates = []
        for idx in benign_indices:
            update = client_models[idx].state_dict()[key].float() - global_dict[key].float()
            benign_updates.append(update)
        global_dict[key] = global_dict[key] + torch.stack(benign_updates).mean(0)

    return global_dict
