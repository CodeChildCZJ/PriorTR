"""Token merge via density peak clustering.

Replicates SparseVLMs' cluster_and_merge algorithm:
1. Compute pairwise L2 distances (normalized)
2. Local density via K-nearest neighbors
3. Density peak center selection
4. Assign tokens to nearest center
5. Weighted average per cluster

Reference: SparseVLMs/llava/model/language_model/utils.py
"""

import torch


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Sample features following index. [B, N, C] + [B, S] -> [B, S, C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def cluster_and_merge(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """Cluster tokens via density peaks and merge by weighted averaging.

    Args:
        x: Token features [B, N, C].
        cluster_num: Number of clusters to produce.

    Returns:
        Merged tokens [B, cluster_num, C].
    """
    B, N, C = x.shape

    if cluster_num >= N:
        return x.clone()

    # 1. Pairwise L2 distances (normalized)
    x1 = x.unsqueeze(2)  # [B, N, 1, C]
    x2 = x.unsqueeze(1)  # [B, 1, N, C]
    dist_matrix = (x1 - x2).norm(dim=-1, p=2) / (C ** 0.5)  # [B, N, N]

    # 2. Local density via K-nearest neighbors
    dist_nearest, _ = torch.topk(
        dist_matrix, k=min(cluster_num, N), dim=-1, largest=False
    )
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # [B, N]

    # 3. Add noise for uniqueness
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype
    ) * 1e-6

    # 4. Distance to nearest higher-density neighbor
    mask = density[:, None, :] > density[:, :, None]  # [B, N, N]
    mask = mask.to(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    # 5. Select cluster centers: score = dist * density
    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)  # [B, cluster_num]

    # 6. Assign tokens to nearest center
    dist_to_centers = _index_points(dist_matrix, index_down)  # [B, cluster_num, N]
    idx_cluster = dist_to_centers.argmin(dim=1)  # [B, N]

    # 7. Ensure centers map to themselves
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # 8. Weighted average merge
    idx = idx_cluster + torch.arange(B, device=x.device)[:, None] * cluster_num  # [B, N]

    token_weight = x.new_ones(B, N, 1)
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(
        dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1)
    )
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx.reshape(B * N)].reshape(B, N, 1)

    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(
        dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C)
    )
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged
