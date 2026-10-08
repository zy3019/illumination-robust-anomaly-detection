"""PatchCore-style baseline: local features, projected greedy coreset, exact 1-NN.

This simplified model does not implement the paper's image-score reweighting.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import resnet18


class PatchFeatures(torch.nn.Module):
    def __init__(self, weights_path, device="cpu"):
        super().__init__()
        network = resnet18(weights=None)
        network.load_state_dict(torch.load(Path(weights_path), map_location="cpu", weights_only=True))
        self.stem = torch.nn.Sequential(network.conv1, network.bn1, network.relu,
                                        network.maxpool, network.layer1)
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()
        self.to(device)

    @torch.inference_mode()
    def forward(self, images):
        self.eval()
        x = (images - self.mean) / self.std
        layer2 = self.layer2(self.stem(x))
        layer3 = self.layer3(layer2)
        f2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2, f3], dim=1)
        b, c, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)


@torch.inference_mode()
def greedy_coreset(candidates, budget, seed, device="cpu", projection_dim=64):
    if candidates.ndim != 2 or not 1 <= budget <= candidates.shape[0]:
        raise ValueError("Budget must be positive and no larger than candidate count")
    if not torch.isfinite(candidates).all():
        raise ValueError("Non-finite candidate features")
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(candidates.shape[1], projection_dim, generator=generator) / math.sqrt(projection_dim)
    # Keep full-dimensional candidates in CPU memory; selection uses a small projection.
    projected = (candidates.cpu() @ projection).to(device)
    selected = torch.empty(budget, dtype=torch.long, device=device)
    current = torch.randint(len(projected), (), generator=generator).to(device)
    distances = torch.full((len(projected),), float("inf"), device=device)
    used = torch.zeros(len(projected), dtype=torch.bool, device=device)
    for i in range(budget):
        selected[i] = current
        used[current] = True
        delta = (projected - projected[current]).square().sum(dim=1)
        distances = torch.minimum(distances, delta)
        distances.masked_fill_(used, -1)  # Never pad by selecting the same index twice.
        current = distances.argmax()
    indices = selected.cpu()
    return candidates.cpu()[indices].contiguous(), indices


@torch.inference_mode()
def nearest_distances(query, bank, query_chunk=256, bank_chunk=1024):
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1] or not len(bank):
        raise ValueError("Invalid query or bank shape")
    if query_chunk < 1 or bank_chunk < 1:
        raise ValueError("Chunk sizes must be positive")
    results = []
    for part in query.split(query_chunk):
        best = torch.full((len(part),), float("inf"), device=query.device)
        for reference in bank.split(bank_chunk):
            distances = torch.cdist(part, reference.to(query.device), p=2)
            best = torch.minimum(best, distances.min(dim=1).values)
        results.append(best)
    return torch.cat(results)


@torch.inference_mode()
def predict(extractor, image, bank):
    features, (h, w) = extractor(image.unsqueeze(0))
    distances = nearest_distances(features[0], bank)
    anomaly_map = F.interpolate(distances.reshape(1, 1, h, w), size=image.shape[-2:],
                                mode="bilinear", align_corners=False)[0, 0]
    return distances.max().item(), anomaly_map.cpu().numpy()
