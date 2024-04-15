import torch

def small_world_connectivity(dist, c, l):
    """
    Calculates a small-world network connectivity matrix based on the given distance matrix.

    Args:
        dist (torch.Tensor): The distance matrix representing pairwise distances between nodes.
        c (float): Maximum connection probability
        l (float): Small world connection radius

    Returns:
        torch.Tensor: Connectivity matrix.

    """

    # Normalize the distance matrix
    dist_norm = (dist - torch.min(dist, dim=1).values[:, None]) / (torch.max(dist, dim=1).values[:, None] - torch.min(dist, dim=1).values[:, None])

    # Calculate the connection probability matrix
    conn_prob = c * torch.exp(-(dist_norm / l) ** 2)

    # Create the input connectivity matrix by selecting connections based on probability
    input_conn = torch.where(torch.rand_like(conn_prob) < conn_prob, conn_prob, torch.zeros_like(conn_prob)).fill_diagonal_(0)

    return input_conn