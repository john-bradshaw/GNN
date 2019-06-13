import torch
from torch import nn as nn

from graph_neural_networks.core import data_types
from graph_neural_networks.node_stack_pattern import stacked_nodes


class GraphFeaturesFromStackedNodeFeaturesBase(nn.Module):
    """
    Attention weighted sum using the computed features.
    The trickiness in performing these operations is that we need to do a sum over nodes. For different graphs we
    have different numbers of nodes and so batching is difficult. The children of this class try doing this in different
    ways.

    Base class for modules that take in the stacked node feature matrix [v*, h] and produce embeddings of graphs
     [g, h']. These are called aggregation functions by Johnson (2017).

    Johnson DD (2017) Learning Graphical State Transitions. In: ICLR, 2017.

    Li Y, Vinyals O, Dyer C, et al. (2018) Learning Deep Generative Models of Graphs.
    arXiv [cs.LG]. Available at: http://arxiv.org/abs/1803.03324.
    """
    def __init__(self, mlp_project_up, mlp_gate, mlp_func, cuda_details=None):
        super().__init__()
        self.mlp_project_up = mlp_project_up  # net that goes from [None, h'] to [None, j] with j>h usually
        self.mlp_gate = mlp_gate  # net that goes from [None, h'] to [None, 1]
        self.mlp_func = mlp_func  # net that goes from [None, j] to [None, q]
        self.cuda_details = cuda_details


class GraphFeaturesStackCS(GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum via a cumsum and then indexing and then doing difference.

    THE NODE FEATURES MUST COME IN ORDER (IE ALL NODES OF A GRAPH ARE GROUPED TOGETHER)
    """
    def forward(self, node_features,  node_grp_start_with_end):
        """
        :param node_features: [v*, h]
        :param node_grp_start_with_end: [g +1] index into node features for where each new graph starts
         with concatenated on the end the index of the next imaginary graph ie  node_features.shape[0]
         so that when you take off one you index the last node
        """
        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]

        weighted_sum = torch.cumsum(gate * proj_up, dim=0)   # [v*, j]
        #todo: check whether cumsumming is a good idea especially when it comes to backprop

        indx_before = node_grp_start_with_end[:-1]
        indx_after = node_grp_start_with_end[1:] - 1

        graph_sums = weighted_sum[indx_after, :] - weighted_sum[indx_before, :]  # [g, j]

        result = self.mlp_func(graph_sums)  # [g, q]
        return result


class GraphFeaturesStackPad(GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum by putting everything into a padded structure and then summing over one of the fimensions

    THE NODE FEATURES MUST COME IN ORDER (IE ALL NODES OF A GRAPH ARE GROUPED TOGETHER)
    """

    def forward(self, node_features, node_grp_start_with_end, max_size):
        """
        :param node_features: [v*, h]
        :param node_grp_start_with_end: [g +1] index into node features for where each new graph starts
         with concatenated on the end the index of the next imaginary graph ie  node_features.shape[0]
         so that when you take off one you index the last node
        :param max_size: the largest number of nodes in a graph.
        """
        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]
        gated_vals = gate * proj_up

        padded = torch.zeros(node_grp_start_with_end.shape[0] - 1, max_size, *proj_up.shape[1:],
                             device=self.cuda_details.device_str, dtype=data_types.TORCH_FLT)

        for i, (start, end) in enumerate(zip(node_grp_start_with_end[:-1],
                                             node_grp_start_with_end[1:] - 1)):
            padded[i, :end-start, ...] = gated_vals[start:end, ...]

        graph_sums = torch.sum(padded, dim=1)  # [g,j]

        result = self.mlp_func(graph_sums)  # [g, q]
        return result


class GraphFeaturesStackIndexAdd(GraphFeaturesFromStackedNodeFeaturesBase):
    """
    Do the sum by Pytorch's index_add method.
    """
    def forward(self, node_features, node_to_graph_id):
        """
        :param node_features: [v*, h]
        :param node_to_graph_id:  for each node index the graph it belongs to [v*]
        """

        proj_up = self.mlp_project_up(node_features)  # [v*, j]
        gate_logit = self.mlp_gate(node_features)  # [v*, 1]
        gate = torch.sigmoid(gate_logit)  # [v*, j]
        gated_vals = gate * proj_up

        num_graphs = node_to_graph_id.max() + 1
        graph_sums = torch.zeros(num_graphs, gated_vals.shape[1],
                                                    device=self.cuda_details.device_str,
                                                    dtype=data_types.TORCH_FLT)  # [g, j]
        graph_sums.index_add_(0, node_to_graph_id, gated_vals)

        result = self.mlp_func(graph_sums)  # [g, q]
        return result

