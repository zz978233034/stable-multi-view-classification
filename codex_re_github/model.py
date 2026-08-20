import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 100
LABEL_SMOOTHING = 0.05


def weighted_cross_entropy(logits, targets, sample_weights=None, class_weights=None):
    losses = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=LABEL_SMOOTHING,
    )
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.reshape(-1).to(losses)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class MultiViewClassifier(nn.Module):
    def __init__(self, classes, input_dims):
        super().__init__()
        self.views = len(input_dims)
        self.encoders = nn.ModuleList(ViewEncoder(dim) for dim in input_dims)
        self.view_heads = nn.ModuleList(
            nn.Linear(EMBED_DIM, classes) for _ in input_dims
        )
        self.fusion_head = nn.Linear(EMBED_DIM, classes)
        self.view_weights = nn.Parameter(torch.ones(self.views) / self.views)

    def forward(self, inputs, feature_weights=None):
        embeddings = [
            encoder(inputs[view]) for view, encoder in enumerate(self.encoders)
        ]
        view_logits = [
            head(embedding) for head, embedding in zip(self.view_heads, embeddings)
        ]

        weights = F.softmax(self.view_weights, dim=0)
        if self.training:
            keep = torch.rand_like(weights) >= 0.1
            if not keep.any():
                keep[weights.argmax()] = True
            weights = weights * keep
            weights = weights / weights.sum()

        fused = sum(weight * embedding for weight, embedding in zip(weights, embeddings))
        classified = fused if feature_weights is None else fused * feature_weights
        return view_logits, self.fusion_head(classified), fused


class ViewEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, EMBED_DIM)
        self.dropout = nn.Dropout(0.1)
        self.style_mix = FeatureStatisticMix(0.5)

    def forward(self, inputs):
        hidden = self.input_norm(inputs)
        hidden = self.dropout(F.relu(self.fc1(hidden)))
        hidden = self.style_mix(hidden)
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        return self.fc3(hidden)


class FeatureStatisticMix(nn.Module):
    """Source-only feature-statistic mixing for style augmentation."""

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, inputs):
        if not self.training or torch.rand((), device=inputs.device) >= self.probability:
            return inputs
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True).clamp_min(1e-6)
        normalized = (inputs - mean) / std
        permutation = torch.randperm(len(inputs), device=inputs.device)
        ratio = torch.rand(len(inputs), 1, device=inputs.device, dtype=inputs.dtype)
        mixed_mean = ratio * mean + (1.0 - ratio) * mean[permutation]
        mixed_std = ratio * std + (1.0 - ratio) * std[permutation]
        return normalized * mixed_std + mixed_mean

