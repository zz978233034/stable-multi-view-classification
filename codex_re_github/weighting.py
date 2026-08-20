"""Adaptive sample and dimension weighting with one fixed configuration."""

import copy
import math

import torch
import torch.nn.functional as F


# Method settings are fixed here instead of exposed as dozens of CLI arguments.
RFF_FEATURES = 10
DIMENSION_PAIRS = 256
DEPENDENCY_WEIGHT = 10.0
REGULARIZATION = 1.0
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
UPDATE_STEPS = 20
TEMPERATURE = 10.0
TARGET_GRADIENT_RMS = 0.1
MIN_WEIGHT = 0.2
MAX_WEIGHT = 5.0
BACKTRACK_STEPS = 8


def make_rff_parameters(device, dtype, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    frequencies = torch.randn(
        RFF_FEATURES, device=device, dtype=dtype, generator=generator
    )
    phases = 2 * math.pi * torch.rand(
        RFF_FEATURES, device=device, dtype=dtype, generator=generator
    )
    return frequencies, phases


def optimize_weights(
    features,
    sample_weights,
    feature_weights,
    rff_parameters,
    seed,
    round_index,
):
    """Update sample and representation-dimension weights."""
    features = features.detach()
    sample_parameter = torch.nn.Parameter(weights_to_logits(sample_weights))
    feature_parameter = torch.nn.Parameter(weights_to_logits(feature_weights))
    sample_optimizer = torch.optim.SGD(
        [sample_parameter], lr=LEARNING_RATE, momentum=MOMENTUM
    )
    feature_optimizer = torch.optim.SGD(
        [feature_parameter], lr=LEARNING_RATE, momentum=MOMENTUM
    )
    pairs = sample_dimension_pairs(
        features.shape[1], DIMENSION_PAIRS, features.device, seed, round_index
    )
    rejected = 0

    def current_weights():
        return (
            bounded_weights(sample_parameter, len(sample_parameter)),
            bounded_weights(feature_parameter, len(feature_parameter)),
        )

    def objective():
        current_samples, current_features = current_weights()
        dependency = dependency_loss(
            features,
            current_samples,
            current_features,
            rff_parameters,
            pairs,
        )
        total = DEPENDENCY_WEIGHT * dependency + REGULARIZATION * (
            current_samples.square().sum() + current_features.square().sum()
        )
        return dependency, total

    def guarded_step(optimizer, parameter, base_loss):
        nonlocal rejected
        parameter_state = parameter.detach().clone()
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        original_lr = optimizer.param_groups[0]["lr"]
        for attempt in range(BACKTRACK_STEPS + 1):
            parameter.data.copy_(parameter_state)
            optimizer.load_state_dict(optimizer_state)
            optimizer.param_groups[0]["lr"] = original_lr * (0.5**attempt)
            optimizer.step()
            parameter.data.sub_(parameter.data.mean())
            with torch.no_grad():
                _, candidate = objective()
            if candidate.item() <= base_loss + 1e-10:
                optimizer.param_groups[0]["lr"] = original_lr
                return True
        parameter.data.copy_(parameter_state)
        optimizer.load_state_dict(optimizer_state)
        optimizer.param_groups[0]["lr"] = original_lr
        rejected += 1
        return False

    with torch.no_grad():
        dependency_before, _ = objective()
    sample_updates = 0
    feature_updates = 0

    for _ in range(UPDATE_STEPS):
        sample_optimizer.zero_grad()
        feature_optimizer.zero_grad()
        _, loss = objective()
        loss.backward()
        sample_score = sample_parameter.grad.norm() / sample_parameter.numel()
        feature_score = feature_parameter.grad.norm() / feature_parameter.numel()

        if sample_score >= feature_score:
            feature_parameter.grad = None
            normalize_gradient(sample_parameter)
            sample_updates += int(
                guarded_step(sample_optimizer, sample_parameter, loss.item())
            )
        else:
            sample_parameter.grad = None
            normalize_gradient(feature_parameter)
            feature_updates += int(
                guarded_step(feature_optimizer, feature_parameter, loss.item())
            )

    sample_weights, feature_weights = current_weights()
    with torch.no_grad():
        dependency_after, _ = objective()
    stats = {
        "sample_updates": sample_updates,
        "feature_updates": feature_updates,
        "sample_std": sample_weights.std().item(),
        "feature_std": feature_weights.std().item(),
        "dependency_before": dependency_before.item(),
        "dependency_after": dependency_after.item(),
        "rejected_updates": rejected,
    }
    return sample_weights.detach(), feature_weights.detach(), stats


def weights_to_logits(weights):
    logits = TEMPERATURE * torch.log(weights.detach().clamp_min(1e-8))
    return logits - logits.mean()


def bounded_weights(logits, total):
    weights = total * F.softmax(logits / TEMPERATURE, dim=0)
    for _ in range(3):
        weights = weights.clamp(MIN_WEIGHT, MAX_WEIGHT)
        weights = total * weights / weights.sum().clamp_min(1e-12)
    return weights


def normalize_gradient(parameter):
    rms = parameter.grad.square().mean().sqrt()
    parameter.grad.mul_(TARGET_GRADIENT_RMS / rms.clamp_min(1e-12))


def sample_dimension_pairs(dimensions, count, device, seed, round_index):
    pairs = torch.triu_indices(dimensions, dimensions, offset=1, device=device)
    if count <= 0 or count >= pairs.shape[1]:
        return pairs
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    permutation = torch.randperm(pairs.shape[1], generator=generator, device=device)
    start = (round_index * count) % pairs.shape[1]
    selected = torch.cat((permutation, permutation))[
        start : start + count
    ]
    return pairs[:, selected]


def dependency_loss(features, sample_weights, feature_weights, rff, pairs):
    frequencies, phases = rff
    mapped = math.sqrt(2.0) * torch.cos(
        features.unsqueeze(-1) * frequencies.view(1, 1, -1)
        + phases.view(1, 1, -1)
    )
    weighted = (
        mapped
        * sample_weights.reshape(-1, 1, 1)
        * feature_weights.reshape(1, -1, 1)
    )
    centered = weighted - weighted.mean(dim=0, keepdim=True)
    left = centered[:, pairs[0], :]
    right = centered[:, pairs[1], :]
    covariance = torch.einsum("npb,npc->pbc", left, right) / max(len(features) - 1, 1)
    sampled_loss = covariance.square().sum()
    total_pairs = features.shape[1] * (features.shape[1] - 1) // 2
    return sampled_loss * (total_pairs / max(pairs.shape[1], 1))

