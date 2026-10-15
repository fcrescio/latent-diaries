"""Training-only separability diagnostics; never used to condition speech."""
import torch
from torch.nn import functional as F


def ridge_scores(train, labels, query, classes, alpha=1.0):
    if alpha <= 0 or train.ndim != 2 or query.shape[1] != train.shape[1]:
        raise ValueError('Positive ridge regularization and aligned features required')
    x, z = train.double(), query.double()
    y = F.one_hot(labels, classes).double()
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered = x - mean_x
    dual = torch.linalg.solve(centered @ centered.T + alpha * torch.eye(len(x)), y - mean_y)
    return (z - mean_x) @ centered.T @ dual + mean_y


def classification_metrics(scores, labels):
    predicted = scores.argmax(-1)
    recalls = [float((predicted[labels == k] == k).float().mean()) for k in labels.unique()]
    return {'accuracy': float((predicted == labels).float().mean()),
            'macro_recall': sum(recalls) / len(recalls), 'count': len(labels),
            'represented_classes': len(recalls)}
