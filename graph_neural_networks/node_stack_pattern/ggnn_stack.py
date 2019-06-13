import torch


from . import stacked_nodes
from . import node_groups
from ..core import data_types
from ..ggnn_general import ggnn_base


class GGNNStackedLine(ggnn_base.GGNNBase):
    def forward(self, graphs: stacked_nodes.StackedNodes):

        hidden = graphs.node_features
        num_nodes = hidden.shape[0]

        for t in range(self.params.num_layers):
            message = torch.zeros(num_nodes, self.params.hlayer_size,
                                                    device=self.params.cuda_details.device_str,
                                                    dtype=data_types.TORCH_FLT)

            # We collect the messages from each edge type.
            for edge_name, projection in self.get_edge_names_and_projections():
                nd_group: node_groups.NeighbourGrpList = graphs.node_grp_dict[edge_name]

                projected_feats = projection(hidden)
                #todo: ^ potentially wasteful doing this projection on all the nodes.

                # We accumulate features from neighbours.
                for neigh_idx in range(nd_group.neigh_relations_rows.shape[0]):
                    # ^ there is a maximum possible degree for which we consider.

                    # First we collect up the indices that we are going to be updating (computed ahead of time)
                    indcs = nd_group.indcs_of_active_neighbrs[neigh_idx]
                    # we ignore the locations which are just padding.
                    if (indcs.shape[0] == 0):
                        break

                    # We then collect up the indices of the neighbours we want to add at these indices
                    index_locs = graphs.node_neighs_offsets[edge_name][neigh_idx]

                    # We add these message vectors to our current message at the indices we are updating.
                    message.index_add_(0, indcs, projected_feats.index_select(0, index_locs))
            # Now the node representations can be updated after taking the message into account.
            hidden = self.GRU_hidden(message, hidden)
        return stacked_nodes.StackedNodes(hidden, graph_group=graphs.graph_groups, neigh_relation_dict=graphs.node_grp_dict)
