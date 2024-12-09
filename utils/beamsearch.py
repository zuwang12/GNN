import numpy as np
import torch


class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.

    References:
        General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        For TSP: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
    """

    def __init__(self, beam_size, batch_size, num_nodes,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, 
                 probs_type='raw', random_start=False, device='cpu', constraint_type='basic', constraint=None):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
            random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
            constraint_type: Type of constraint applied (e.g., 'basic', 'box')
            constraint: Constraint matrix, indicating restrictions on node transitions
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        self.device = device
        self.constraint_type = constraint_type
        self.constraint = constraint
        
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong)
        if random_start == True:
            # Random starting nodes
            # self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong)
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size), device=device).type(self.dtypeLong)
        # Mask for constructing valid hypothesis
        # self.mask = torch.ones(batch_size, beam_size, num_nodes).type(self.dtypeFloat)
        self.mask = torch.ones(batch_size, beam_size, num_nodes, dtype=dtypeFloat, device=device)
        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search
        # Cluster interconnection tracking (new)
        self.cluster_interconnection_count = torch.zeros(batch_size, beam_size, num_nodes, dtype=dtypeLong, device=device)
        
        # Score for each translation on the beam
        # self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat)
        self.scores = torch.zeros(batch_size, beam_size, dtype=dtypeFloat, device=device)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes.to(device)]

    def get_current_state(self):
        """Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state.type(torch.int64).to(self.device)

    def get_current_origin(self):
        """Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        """Advances the beam based on transition probabilities.

        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """
        # Compound the previous scores (summing logits == multiplying probabilities)
        if len(self.prev_Ks) > 0:
            if self.probs_type == 'raw':
                beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
            elif self.probs_type == 'logits':
                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            if self.probs_type == 'raw':
                # beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(self.dtypeFloat)
                beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size(), dtype=self.dtypeFloat, device=self.device)
            elif self.probs_type == 'logits':
                # beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size()).type(self.dtypeFloat)
                beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size(), dtype=self.dtypeFloat, device=self.device)

        # Apply constraints if necessary
        if self.constraint_type == 'box':
            for batch_idx in range(self.batch_size):
                for beam_idx in range(self.beam_size):
                    current_node = self.next_nodes[-1][batch_idx, beam_idx]  # Last selected node
                    invalid_mask = torch.tensor(self.constraint[batch_idx, current_node.cpu().numpy()], 
                                                device=self.device, dtype=self.dtypeFloat)
                    beam_lk[batch_idx, beam_idx, :] = beam_lk[batch_idx, beam_idx, :] - (invalid_mask * 1e20)

        elif self.constraint_type == 'path':
            for batch_idx in range(self.batch_size):
                for beam_idx in range(self.beam_size):
                    current_node = self.next_nodes[-1][batch_idx, beam_idx]  # Last selected node
                    constraint_pairs = self.constraint[batch_idx]

                    for i in range(0, len(constraint_pairs), 2):
                        node1 = int(constraint_pairs[i])
                        node2 = int(constraint_pairs[i + 1])
                        
                        if node1 == node2:
                            continue
                        
                        if current_node == node1:
                            beam_lk[batch_idx, beam_idx, :] = -1e20
                            beam_lk[batch_idx, beam_idx, node2] = trans_probs[batch_idx, beam_idx, node2]
                        elif current_node == node2:
                            beam_lk[batch_idx, beam_idx, :] = -1e20
                            beam_lk[batch_idx, beam_idx, node1] = trans_probs[batch_idx, beam_idx, node1] 
                        else:
                            pass
                        
        elif self.constraint_type == 'cluster':
            for batch_idx in range(self.batch_size):
                for beam_idx in range(self.beam_size):
                    current_node = int(self.next_nodes[-1][batch_idx, beam_idx])
                    current_cluster = int(self.constraint[batch_idx, current_node])
                    
                    for next_node in range(self.num_nodes):
                        next_cluster = int(self.constraint[batch_idx, next_node])
                        
                        if current_cluster != next_cluster:
                            if self.cluster_interconnection_count[batch_idx, beam_idx, next_cluster] >= 1:
                                beam_lk[batch_idx, beam_idx, next_node] = -1e10  # Invalidate transition
                            else:
                                # Allow transition, increment the count
                                self.cluster_interconnection_count[batch_idx, beam_idx, next_cluster] += 1

        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = bestScoresId // self.num_nodes
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, perm_mask.to(self.dtypeLong))
        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        """Sets new_nodes to zero in mask.
        """
        arr = (torch.arange(0, self.num_nodes, device=self.device).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(self.dtypeLong))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask).to(self.device)
        update_mask = 1 - torch.eq(arr, new_nodes).type(self.dtypeFloat)
        self.mask = self.mask * update_mask

        if self.probs_type == 'logits':
            self.mask[self.mask == 0] = 1e20

    def sort_best(self):
        """Sort the beam.
        """
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score and index of the best hypothesis in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis.

        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
        assert self.num_nodes == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes, dtype=self.dtypeLong, device=self.device)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k) #k: batch_size * 1
            
        return hyp
