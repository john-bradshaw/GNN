



import torch

from ..core import data_types
from ..ggnn_general import ggnn_base
from . import graph_as_adj_list


class GGNNSparse(ggnn_base.GGNNBase):
    def forward(self, graphs: graph_as_adj_list.GraphAsAdjList):

        hidden = graphs.node_features
        num_nodes = hidden.shape[0]

        for t in range(self.params.num_layers):
            message = torch.zeros(num_nodes, self.params.hlayer_size,
                                                    device=self.params.cuda_details.device_str,
                                                    dtype=data_types.TORCH_FLT)

            for edge_name, projection in self.get_edge_names_and_projections():
                adj_list = graphs.edge_type_to_adjacency_list_directed_map[edge_name]
                if adj_list is None:
                    continue  # no edges of this type
                projected_feats = projection(hidden)
                #todo: potentially wasteful doing this projection on all nodes (ie many may not
                # be connected by all kinds of edge)
                message.index_add_(0, adj_list[0], projected_feats.index_select(0, adj_list[1]))

            hidden = self.GRU_hidden(message, hidden)

        return graph_as_adj_list.GraphAsAdjList(hidden, graphs.edge_type_to_adjacency_list_map, graphs.node_to_graph_id)
