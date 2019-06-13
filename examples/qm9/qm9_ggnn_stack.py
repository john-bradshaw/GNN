
import numpy as np
import torch
from torch import nn

from graph_neural_networks.node_stack_pattern import ggnn_stack, node_groups, stacked_nodes
from graph_neural_networks.datasets import qm9
from graph_neural_networks.core import utils
from graph_neural_networks.ggnn_general import ggnn_base
from graph_neural_networks.ggnn_general import graph_tops
from graph_neural_networks.core import mlp
from graph_neural_networks.example_trainers import qm9_regression


class DatasetTransform(object):
    def __init__(self, hidden_layer_size):
        self.neigh_relations = qm9.EdgeListToNeighRelations(bond_types=[1, 2, 3, 4])
        self.nf_em = qm9.NodeFeaturesEmbedder(hidden_layer_size)

    def __call__(self, edge, node_features):
        node_features = self.nf_em(node_features)

        neigh_relation_dict = self.neigh_relations(edge)
        neigh_relation_dict = {f"edge_{k}": node_groups.NeighbourGrpList(v.T) for k, v in neigh_relation_dict.items()}

        nd_grps = node_groups.NodeGroup(np.array([node_features.shape[0]], dtype=int),
                                        np.zeros(node_features.shape[0], dtype=int))

        stacked_nds = stacked_nodes.StackedNodes(node_features, nd_grps, neigh_relation_dict)
        return stacked_nds


def collate_function(batch):
    #todo: will not be able to pin memory at the moment.

    stacked_nds = [elem[0] for elem in batch]
    targets = [elem[1] for elem in batch]

    stacked_nds_catted = stacked_nds[0].concatenate(stacked_nds)

    stacked_nds_catted.to_torch(cuda_details=utils.CudaDetails(use_cuda=False))
    targets = torch.from_numpy(np.array(targets))

    return stacked_nds_catted, targets


class GGNNModel(nn.Module):
    def __init__(self, hidden_layer_size, edge_names, cuda_details, T):
        super().__init__()
        self.ggnn = ggnn_stack.GGNNStackedLine(
            ggnn_base.GGNNParams(hidden_layer_size, edge_names,
                                       cuda_details, T))

        mlp_project_up = mlp.MLP(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_gate = mlp.MLP(mlp.MlpParams(hidden_layer_size, 1, []))
        mlp_down = lambda x: x

        self.ggnn_top = graph_tops.GraphFeaturesStackCS(mlp_project_up, mlp_gate, mlp_down, cuda_details)

    def forward(self, graphs: stacked_nodes.StackedNodes):
        graphs_out = self.ggnn(graphs)
        graph_feats = self.ggnn_top(graphs_out.node_features, graphs_out.graph_groups.node_grp_start_with_end)
        return graph_feats


class StackParts(qm9_regression.ExperimentParts):
    def create_model(self):
        return GGNNModel(self.exp_params.hidden_layer_size, self.exp_params.edge_names,
                         self.exp_params.cuda_details, self.exp_params.T)

    def create_transform(self):
        return DatasetTransform(self.exp_params.hidden_layer_size)

    def create_collate_function(self):
        return collate_function

    def data_split_and_cudify_func(self, data):
        graphs, targets = data
        graphs.to_cuda(cuda_details=self.exp_params.cuda_details)
        targets = self.exp_params.cuda_details.return_cudafied(targets)
        return (graphs,), targets


def main():
    exp_params = qm9_regression.ExperimentParams()
    exp_parts = StackParts(exp_params)
    qm9_regression.main_runner(exp_parts)


if __name__ == '__main__':
    print("Starting...")
    main()
    print('Completed!')



