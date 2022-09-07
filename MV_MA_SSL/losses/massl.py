
import torch
import torch.nn.functional as F


def byol_loss_multi_views_func(p: torch.Tensor, z: torch.Tensor,p1: torch.Tensor, z1: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p, p1 (torch.Tensor): NxD Tensor containing predicted features from view 1
        z, z1 (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: BYOL's loss.
    """

    if simplified:
       
        loss = F.cosine_similarity(p, z.detach(), dim=-1).mean() + F.cosine_similarity(p1, z1.detach(), dim=-1).mean() 
        return 2 - 2 * loss

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    p1 = F.normalize(p1, dim=-1)
    z1 = F.normalize(z1, dim=-1)

    return 2 - 2 * ((p * z.detach()).sum(dim=1).mean() +(p1 * z1.detach()).sum(dim=1).mean())