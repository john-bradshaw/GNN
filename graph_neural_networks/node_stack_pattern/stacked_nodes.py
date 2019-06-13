
import typing
import itertools
import collections

import numpy as np
import torch

from ..core import nd_ten_ops
from . import node_groups
from ..core import utils


class StackedNodes(object):
    def __init__(self, node_features: nd_ten_ops.Nd_Ten, graph_group: node_groups.NodeGroup,
                 neigh_relation_dict: typing.Mapping[str, node_groups.NeighbourGrpList]):
        """
        :param node_features: [v*, h] the node features or embeddings
        :param neigh_relation_dict: Assignments of nodes to different groups.
        """
        self.node_features: nd_ten_ops.Nd_Ten = node_features
        self.graph_groups: node_groups.NodeGroup = graph_group
        self.node_grp_dict: typing.Mapping[typing.Any, node_groups.NeighbourGrpList] = neigh_relation_dict

        # The following properties are only calculated lazily
        self.node_neighs_offsets = None
        # ^ this is a dictionary of lists. The dictionary keys mirror node_grp_dict ie we have one list for every
        # kind of node grping. The lists then store the true indexes within that group. Taking the lcoal relative ones
        # from the corresponding node_grp_dict value and the global ones from graph groups.

    def do_lazy_ops(self):
        for ngrp in self.node_grp_dict.values():
            ngrp.do_lazy_ops()
        if self.node_neighs_offsets is None:
            if self.variant is nd_ten_ops.NdTensor.NUMPY:
                final_dict = collections.defaultdict(list)
                for key, neigh_grp in self.node_grp_dict.items():
                    for i in range(neigh_grp.neigh_relations_rows.shape[0]):
                        # we work out which are the neighbour indexes of each node
                        # there are two main subtleties.
                        # (1) we need to take off 1 due to the fact we start indexing at -1
                        # (2) We need to add the offset so that we index into the correct graph.
                        active_indcs = neigh_grp.indcs_of_active_neighbrs[i]
                        index_locs_rel_neighs = neigh_grp.neigh_relations_rows[i][active_indcs]
                        index_locs_global_offset = self.graph_groups.node_grp_start[
                            self.graph_groups.node_to_graph_id[active_indcs]
                        ] - 1
                        index_locs = index_locs_rel_neighs + index_locs_global_offset
                        final_dict[key].append(index_locs)

                self.node_neighs_offsets = final_dict
            else:
                raise NotImplementedError

    @property
    def variant(self) -> nd_ten_ops.NdTensor:
        return nd_ten_ops.work_out_nd_or_tensor(self.node_features)

    @classmethod
    def concatenate(cls, grp):

        # 1. Do the node features
        node_feats_list = [elem.node_features for elem in grp]
        new_node_feats = nd_ten_ops.concatenate(node_feats_list)

        # 2. Do the graph groups
        graph_group_list = [elem.graph_groups for elem in grp]
        new_graph_group = graph_group_list[0].concatenate(graph_group_list)

        # 3. Do the node grp dict
        node_grp_dict_list = [elem.node_grp_dict for elem in grp]

        # We first out what node group  we have and so which ones need to be concatenated.
        all_keys = set(itertools.chain(*[g.keys() for g in node_grp_dict_list]))

        # For each of the node groups we now concatenate them together.
        new_node_dict = {}
        for key in all_keys:
            node_grp_for_particular_key_list = [dict_[key] for dict_ in node_grp_dict_list]
            all_node_grp_classes = set([type(elem) for elem in node_grp_for_particular_key_list])
            assert len(all_node_grp_classes) == 1, "inconsistent node group class tuples"
            all_node_grp_classes = all_node_grp_classes.pop()

            new_node_dict[key] = all_node_grp_classes.concatenate(
                node_grp_for_particular_key_list
            )
        return cls(new_node_feats, new_graph_group, new_node_dict)

    def to_torch(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()

        self._map_node_neigh_offsets(lambda x: utils.from_np_to_cuda(x, cuda_details))
        self.node_features = utils.from_np_to_cuda(self.node_features, cuda_details)
        self.node_grp_dict = {k: v.to_torch(cuda_details) for k, v in self.node_grp_dict.items()}
        self.graph_groups = self.graph_groups.to_torch(cuda_details)

    def to_cuda(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()

        self._map_node_neigh_offsets(cuda_details.return_cudafied)
        self.node_features = cuda_details.return_cudafied(self.node_features)
        self.node_grp_dict = {k: v.to_cuda(cuda_details) for k,v in self.node_grp_dict.items()}
        self.graph_groups = self.graph_groups.to_cuda(cuda_details)

    def _map_node_neigh_offsets(self, func):
        self.node_neighs_offsets = {k: [func(elem) for elem in v] for k, v in self.node_neighs_offsets.items()}

