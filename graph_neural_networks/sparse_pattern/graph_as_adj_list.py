

import typing

import numpy as np

from ..core import nd_ten_ops
from ..core import utils


class GraphAsAdjList(object):
    def __init__(self, node_features: nd_ten_ops.Nd_Ten, edge_type_to_adjacency_list_map: typing.Mapping[str, nd_ten_ops.Nd_Ten],
                 node_to_graph_id: nd_ten_ops.Nd_Ten):
        """
        :param node_features: [v*, h]. Matrix of node features
        :param edge_type_to_adjacency_list_map: dictionary which for evcery edge type (represented by keys) we have an
        [2, E*] matrix, each row in here represents the bond. Indexing into the node features matrix
        :param node_to_graph_id: [v*] (should be in order ie nodes in the same graph should be consecutive).
         We have one for each node.For each node it indexes into the node for which they belong.
        """
        self.node_features = node_features
        self.edge_type_to_adjacency_list_map = edge_type_to_adjacency_list_map
        # ^ in Numpy mode then we have zero dim arrays. If in Torch mode then we will have None instead.
        self.node_to_graph_id = node_to_graph_id

        self.max_num_graphs = self.node_to_graph_id.max() + 1  # plus one to deal with fact that index from zero.

        # These we compute lazily:
        self.edge_type_to_adjacency_list_directed_map = None  # will simply repeat edges in both directions

    def do_lazy_ops(self):
        if self.edge_type_to_adjacency_list_directed_map is None:
            new = {}
            for key, value in self.edge_type_to_adjacency_list_map.items():
                if value.shape[0] == 0:
                    new[key] = None
                else:
                    new[key] = nd_ten_ops.concatenate([value, value[::-1]], axis=1)
            self.edge_type_to_adjacency_list_directed_map = new

    @property
    def variant(self) -> nd_ten_ops.NdTensor:
        """
        Works out whether underlying stored in Pytorch Tensors or Numpy arrays.
        :return:
        """
        return nd_ten_ops.work_out_nd_or_tensor(self.node_features)

    @classmethod
    def concatenate(cls, grps):

        # Set up the lists will be used to store the new components
        node_features_new = []

        all_keys = set([frozenset(g.edge_type_to_adjacency_list_map.keys()) for g in grps])
        assert len(all_keys) == 1, "inconsistent edges among graph groups"
        adjacency_list_for_all_edges_new = {k: [] for k in all_keys.pop()}
        node_to_graph_id_new = []

        # Now go through and add the respective matrices nodes to these groups
        max_node_index_so_far = 0
        max_num_grps_so_far = 0
        # ^ the indices will all need to be shifted up as add more graphs this variable will record by how much
        for g in grps:

            node_features_new.append(g.node_features)

            for k, v in g.edge_type_to_adjacency_list_map.items():
                if v.shape[0] == 0:
                    continue
                    # sometimes it is empty (ie not every graph has to contain every edge type) so skip these
                adjacency_list_for_all_edges_new[k].append(v + max_node_index_so_far)
            node_to_graph_id_new.append(g.node_to_graph_id + max_num_grps_so_far)

            max_num_grps_so_far += g.max_num_graphs
            max_node_index_so_far += g.node_features.shape[0]

            if g.edge_type_to_adjacency_list_directed_map is not None:
                raise NotImplementedError

        # Now concatenate together
        node_features_new = nd_ten_ops.concatenate(node_features_new, axis=0)
        def cat_or_set_zero(v):
            return np.array([[], []], dtype=np.int64) if len(v) == 0 else nd_ten_ops.concatenate(v, axis=1)
        adjacency_list_for_all_edges_new = {k: cat_or_set_zero(v) for k, v in
                                            adjacency_list_for_all_edges_new.items()}
        node_to_graph_id_new = nd_ten_ops.concatenate(node_to_graph_id_new)

        return GraphAsAdjList(node_features_new, adjacency_list_for_all_edges_new, node_to_graph_id_new)

    def to_torch(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()
        def func_to_map(x):
            return None if (x is None or x.size == 0) else utils.from_np_to_cuda(x, cuda_details)
        self._map_all_props(func_to_map)
        return self

    def to_cuda(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()
        def func_to_map(x):
            return x if x is None else cuda_details.return_cudafied(x)
        self._map_all_props(func_to_map)
        return self

    def _map_all_props(self, func):
        self.node_features = func(self.node_features)
        self.node_to_graph_id = func(self.node_to_graph_id)
        self._map_over_adjacency_list_for_all_edges_both_directed_and_undirected(func)

    def _map_over_adjacency_list_for_all_edges_both_directed_and_undirected(self, func):
        self.edge_type_to_adjacency_list_map = {k: func(v) for k, v in self.edge_type_to_adjacency_list_map.items()}
        self.edge_type_to_adjacency_list_directed_map = {k: func(v) for k, v in
                                                         self.edge_type_to_adjacency_list_directed_map.items()}

