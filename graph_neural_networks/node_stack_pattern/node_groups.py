
import typing

import numpy as np
import torch

from ..core import nd_ten_ops
from ..core import utils


class NodeGroup(object):
    """
    Assigns each node to a particular groups.
    This allows one eg to group nodes into sets which can for instance represent a graph.
    The nodes should be lined up according to their group and the groups should appear in order.
    """
    def __init__(self,
                 node_nums_per_grp: nd_ten_ops.Nd_Ten,
                 node_to_graph_id: nd_ten_ops.Nd_Ten,
                 node_grp_representation: typing.Optional[torch.Tensor]=None):
        self.node_nums_per_grp = node_nums_per_grp  # [g] The number of nodes collected into each group
        self.node_to_graph_id = node_to_graph_id  # [v*] a mapping from each node into its respective group
        self.node_grp_representation = node_grp_representation
        # ^ initially empty variable that can be used to store a representation for each group.

        self.node_grp_start_with_end = nd_ten_ops.add_zero_at_start_of_one_d(nd_ten_ops.cumsum(self.node_nums_per_grp))
        self.node_grp_start = self.node_grp_start_with_end[:-1]

    @property
    def variant(self) -> nd_ten_ops.NdTensor:
        """
        Works out whether underlying stored in Pytorch Tensors or Numpy arrays.
        :return:
        """
        assert isinstance(self.node_nums_per_grp, type(self.node_to_graph_id)), \
            "Node nums and node groupings are inconsistent."
        return nd_ten_ops.work_out_nd_or_tensor(self.node_to_graph_id)

    @classmethod
    def concatenate(cls, grps):
        node_nums_grps = []
        node_grps_list = []
        node_grp_representations = []

        offset = 0
        # ^ the node groupings is a map from nodes to group index. as we stack later nodes, their ids will need to be
        # shifted up by the offset.
        for grp in grps:
            node_nums_grps.append(grp.node_nums_per_grp)
            node_grps_list.append(grp.node_to_graph_id + offset)
            node_grp_representations.append(grp.node_grp_representation)

            offset += grp.node_to_graph_id.max() + 1

        node_nums_per_grp = nd_ten_ops.concatenate(node_nums_grps)
        node_to_graph_id = nd_ten_ops.concatenate(node_grps_list)

        if all([elem is not None for elem in node_grp_representations]):
            node_grp_representations = nd_ten_ops.concatenate(node_grp_representations)
        else:
            node_grp_representations = None

        return NodeGroup(node_nums_per_grp, node_to_graph_id, node_grp_representations)

    def to_torch(self, cuda_details: utils.CudaDetails):
        self.node_nums_per_grp = utils.from_np_to_cuda(self.node_nums_per_grp, cuda_details)
        self.node_to_graph_id = utils.from_np_to_cuda(self.node_to_graph_id, cuda_details)
        self.node_grp_start_with_end = utils.from_np_to_cuda(self.node_grp_start_with_end, cuda_details)
        self.node_grp_start = utils.from_np_to_cuda(self.node_grp_start, cuda_details)
        return self

    def to_cuda(self, cuda_details: utils.CudaDetails):
        self.node_nums_per_grp = cuda_details.return_cudafied(self.node_nums_per_grp)
        self.node_to_graph_id = cuda_details.return_cudafied(self.node_to_graph_id)
        self.node_grp_start_with_end = cuda_details.return_cudafied(self.node_grp_start_with_end)
        self.node_grp_start = cuda_details.return_cudafied(self.node_grp_start)
        self.node_grp_representation = cuda_details.return_cudafied(self.node_grp_representation) \
            if self.node_grp_representation is not None else None
        return self


class NeighbourGrpList(object):
    """
    Contains information about neighbours (this can be withina  group indicated by a seperate NodeGroup class.
    Ie this allows one to store inter group relationships.
    This is useful for instance storing the neighbours of a particular node in a graph.
    """
    def __init__(self, neighbour_relations: nd_ten_ops.Nd_Ten):
        """
        :param neighbour_relations: contains the neighbours of each node. indexes start from 1 with 0 being used to
        indicate no neigbours
        [max_neighs, v*]
        """
        self.neigh_relations_rows = neighbour_relations

        # This stuff is creted lazily
        self.indcs_of_active_neighbrs = None

    def do_lazy_ops(self):
        """
        function to be called after last concatenation. before made into a tensor
        """
        if self.indcs_of_active_neighbrs is None:
            if self.variant is nd_ten_ops.NdTensor.NUMPY:
                self.indcs_of_active_neighbrs = [
                    np.nonzero(self.neigh_relations_rows[i] != 0)[0] for i in
                    range(self.neigh_relations_rows.shape[0])
                ]
            else:
                raise NotImplementedError

    @property
    def variant(self) -> nd_ten_ops.NdTensor:
        """
        Works out whether underlying stored in Pytorch Tensors or Numpy arrays.
        :return:
        """
        return nd_ten_ops.work_out_nd_or_tensor(self.neigh_relations_rows)


    @classmethod
    def concatenate(cls, grps):
        max_neighs = max([g.neigh_relations_rows.shape[0] for g in grps])
        neigh_realtions = []
        for g in grps:
            n_rel = g.neigh_relations_rows
            padded = nd_ten_ops.pad_bottom_2d(n_rel, max_neighs - n_rel.shape[0])
            neigh_realtions.append(padded)

        result = nd_ten_ops.concatenate(neigh_realtions, axis=1)
        if any([g.indcs_of_active_neighbrs is not None for g in grps]):
            raise NotImplementedError

        return NeighbourGrpList(result)

    def to_torch(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()
        self.indcs_of_active_neighbrs = [utils.from_np_to_cuda(elem, cuda_details)
                                         for elem in self.indcs_of_active_neighbrs]

        self.neigh_relations_rows = utils.from_np_to_cuda(self.neigh_relations_rows, cuda_details)
        return self

    def to_cuda(self, cuda_details: utils.CudaDetails):
        self.do_lazy_ops()
        self.neigh_relations_rows = cuda_details.return_cudafied(self.neigh_relations_rows)

        self.indcs_of_active_neighbrs = [cuda_details.return_cudafied(elem)
                                         for elem in self.indcs_of_active_neighbrs]
        return self