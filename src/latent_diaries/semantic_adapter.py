"""Continuous modality calibration with train-group regularization selection."""
import random

import torch
from torch.nn import functional as F


def semantic_condition(image, coefficient=None, *, control='image', training_mean=None):
    """Keep missing image and missing semantics distinct, including affine bias."""
    if image.ndim != 2 or image.shape[1] != 768 or not torch.isfinite(image).all():
        raise ValueError('Finite [batch,768] image features required')
    if control == 'zero_semantic_after_adapter':
        return torch.zeros_like(image)
    if control == 'mean_semantic_baseline':
        if training_mean is None or training_mean.shape != (768,) or not torch.isfinite(training_mean).all():
            raise ValueError('Verified training semantic mean required')
        return F.normalize(training_mean[None].expand_as(image), dim=-1)
    if control not in {'image', 'zero_image_through_adapter'}:
        raise ValueError('Unknown semantic control')
    value = torch.zeros_like(image) if control == 'zero_image_through_adapter' else image
    if coefficient is None:
        return value
    if coefficient.shape != (769, 768) or not torch.isfinite(coefficient).all():
        raise ValueError('Finite [769,768] affine coefficient required')
    return F.normalize(torch.cat([value.new_ones(len(value), 1), value], -1) @ coefficient, dim=-1)


def scale_semantic_deviation(coefficient, target_mean, gain):
    if gain <= 0 or target_mean.shape != coefficient.shape[1:]:
        raise ValueError('Positive semantic gain and aligned training mean required')
    result = coefficient * gain
    result[0] += target_mean * (1 - gain)
    return result


def fit_coefficient(x, y, alpha):
    if alpha <= 0 or x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError('Positive regularization and aligned training matrices required')
    x, y = x.double(), y.double()
    center_x, center_y = x.mean(0), y.mean(0)
    centered = x - center_x
    weights = centered.T @ torch.linalg.solve(
        centered @ centered.T + alpha * torch.eye(len(x), device=x.device), y - center_y)
    return torch.cat([(center_y - center_x @ weights)[None], weights], 0).float()


def select_alpha(x, y, groups, alphas=(0.001, 0.01, 0.1, 1.0), folds=5, seed=42):
    unique = sorted(set(groups))
    if len(groups) != len(x) or folds < 2 or len(unique) < folds:
        raise ValueError('Enough aligned independent training groups required')
    random.Random(seed).shuffle(unique)
    assignment = {g: i % folds for i, g in enumerate(unique)}
    scores = {}
    for alpha in alphas:
        total, count = 0.0, 0
        for fold in range(folds):
            held = torch.tensor([assignment[g] == fold for g in groups], device=x.device)
            coefficient = fit_coefficient(x[~held], y[~held], alpha).to(x.device)
            query = torch.cat([x[held].new_ones(int(held.sum()), 1), x[held]], -1)
            prediction = F.normalize(query @ coefficient, dim=-1)
            total += float((prediction - y[held]).square().sum())
            count += int(held.sum())
        scores[str(alpha)] = total / count
    selected = min(alphas, key=lambda a: (scores[str(a)], a))
    return selected, scores
