import torch


def entropy(p_dist: torch.Tensor) -> float:
    """"
    Calculates Shannon entropy for a probability distribution

    Args:
        p_dist: probability distribution (torch.Tensor)

    Returns:
        entropy (float)
    """
    # add epsilon because log(0) = nan
    p_dist = p_dist.view(-1) + 1e-12
    return - torch.mul(
        p_dist,
        p_dist.log()
    ).sum(0).item()
