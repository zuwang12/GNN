import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.beamsearch import *
from utils.graph_utils import *


def loss_nodes(y_pred_nodes, y_nodes, node_cw):
    """
    Loss function for node predictions.

    Args:
        y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
        y_nodes: Targets for nodes (batch_size, num_nodes)
        node_cw: Class weights for nodes loss

    Returns:
        loss_nodes: Value of loss function
    
    """
    # Node loss
    y = F.log_softmax(y_pred_nodes, dim=2)  # B x V x voc_nodes_out
    y = y.permute(0, 2, 1)  # B x voc_nodes x V
    loss_nodes = nn.NLLLoss(node_cw)(y, y_nodes)
    return loss_nodes


def loss_edges(y_pred_edges, y_edges, edge_cw):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function
    
    """
    
    if edge_cw is not None and edge_cw.device != y_pred_edges.device:
        edge_cw = edge_cw.to(y_pred_edges.device)

    # Apply log softmax to the predictions
    y = F.log_softmax(y_pred_edges, dim=3)  # Apply softmax along the last dimension (voc_edges_out)
    y = y.permute(0, 3, 1, 2).contiguous()  # Change dimension order and make contiguous

    # Compute the loss using NLLLoss with class weights
    loss_func = nn.NLLLoss(weight=edge_cw)  # Create the loss function with the class weights
    loss = loss_func(y, y_edges)  # Compute the loss between the log softmax output and the targets
    
    return loss


def beamsearch_tour_nodes(y_pred_edges, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='raw', random_start=False, constraint_type = 'basic', constraint = None):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration

    y_pred_edges = y_pred_edges.to(device).type(dtypeFloat)  # Move y_pred_edges to the correct device and data type

    if probs_type == 'raw':
        y = F.softmax(y_pred_edges, dim=3)  # Apply softmax over the last dimension
        y = y[:, :, :, 1]  # Consider the second dimension only
    elif probs_type == 'logits':
        y = F.log_softmax(y_pred_edges, dim=3)  # Apply log softmax over the last dimension
        y = y[:, :, :, 1]  # Consider the second dimension only
        y[y == 0] = -1e-20  # Set log(1) to very small negative number to avoid -inf

    # Instantiate the Beamsearch class with the device parameter
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start, device, constraint_type = constraint_type, constraint = constraint)
    
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())

    ends = torch.zeros(batch_size, 1, dtype=dtypeLong, device=device)  # Initialize ends on the correct device
    return beamsearch.get_hypothesis(ends)


def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False, constraint_type='basic', constraint = None):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Final predicted tour is the one with the shortest tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        shortest_tours: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred_edges = y_pred_edges.to(device).type(dtypeFloat)
    x_edges_values = x_edges_values.to(device).type(dtypeFloat)

    if probs_type == 'raw':
        y = F.softmax(y_pred_edges, dim=3)  # Apply softmax over the last dimension
        y = y[:, :, :, 1]  # Consider the second dimension only
    elif probs_type == 'logits':
        y = F.log_softmax(y_pred_edges, dim=3)  # Apply log softmax over the last dimension
        y = y[:, :, :, 1]  # Consider the second dimension only
        y[y == 0] = -1e-20  # Set 0s to very small negative number

    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start, device, constraint_type=constraint_type, constraint = constraint)

    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())

    ends = torch.zeros(batch_size, 1, dtype=dtypeLong, device=device)
    shortest_tours = beamsearch.get_hypothesis(ends)
    shortest_lens = [1e6] * len(shortest_tours)

    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())

    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1, dtype=dtypeLong, device=device)
        hyp_tours = beamsearch.get_hypothesis(ends)

        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx].cpu().numpy())
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len

    return shortest_tours


def update_learning_rate(optimizer, lr):
    """
    Updates learning rate for given optimizer.

    Args:
        optimizer: Optimizer object
        lr: New learning rate

    Returns:
        optimizer: Updated optimizer object
        s
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def edge_error(y_pred, y_target, x_edges):
    """
    Computes edge error metrics for given batch prediction and targets.

    Args:
        y_pred: Edge predictions (batch_size, num_nodes, num_nodes, voc_edges)
        y_target: Edge targets (batch_size, num_nodes, num_nodes)
        x_edges: Adjacency matrix (batch_size, num_nodes, num_nodes)

    Returns:
        err_edges, err_tour, err_tsp, edge_err_idx, err_idx_tour, err_idx_tsp
    
    """
    y = F.softmax(y_pred, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V

    # Edge error: Mask out edges which are not connected
    mask_no_edges = x_edges.long()
    err_edges, _ = _edge_error(y, y_target, mask_no_edges)

    # TSP tour edges error: Mask out edges which are not on true TSP tours
    mask_no_tour = y_target
    err_tour, err_idx_tour = _edge_error(y, y_target, mask_no_tour)

    # TSP tour edges + positively predicted edges error:
    # Mask out edges which are not on true TSP tours or are not predicted positively by model
    mask_no_tsp = ((y_target + y) > 0).long()
    err_tsp, err_idx_tsp = _edge_error(y, y_target, mask_no_tsp)

    return 100 * err_edges, 100 * err_tour, 100 * err_tsp, err_idx_tour, err_idx_tsp


def _edge_error(y, y_target, mask):
    """
    Helper method to compute edge errors.

    Args:
        y: Edge predictions (batch_size, num_nodes, num_nodes)
        y_target: Edge targets (batch_size, num_nodes, num_nodes)
        mask: Edges which are not counted in error computation (batch_size, num_nodes, num_nodes)

    Returns:
        err: Mean error over batch
        err_idx: One-hot array of shape (batch_size)- 1s correspond to indices which are not perfectly predicted
    
    """
    # Compute equalities between pred and target
    acc = (y == y_target).long()
    # Multipy by mask => set equality to 0 on disconnected edges
    acc = (acc * mask)
    #  Get accuracy of each y in the batch (sum of 1s in acc_edges divided by sum of 1s in edges mask)
    acc = acc.sum(dim=1).sum(dim=1).to(dtype=torch.float) / mask.sum(dim=1).sum(dim=1).to(dtype=torch.float)
    # Compute indices which are not perfect
    err_idx = (acc < 1.0)
    # Take mean over batch
    acc = acc.sum().to(dtype=torch.float).item() / acc.numel()
    # Compute error
    err = 1.0 - acc
    return err, err_idx
