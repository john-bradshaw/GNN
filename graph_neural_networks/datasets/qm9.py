"""
Tools to load in the QM9 dataset and transform it.


Works on the json outputs of https://github.com/Microsoft/gated-graph-neural-network-samples/blob/master/get_data.py

We call an edge list a list of (src_node, bond_type, destination_node) tuples whereas an adjacency list just
a list of (src_node, destination_node) tuples
"""
import collections
import json

import numpy as np
from torch.utils import data


class NodeFeaturesEmbedder(object):
    """ This class puts all the node features together
    so that they are stacked in one larger matrix."""
    def __init__(self, total_dims=None, max_nodes=None):
        self.total_dims = total_dims
        self.max_nodes = max_nodes

    def __call__(self, in_features):
        current_length = in_features.shape[1]
        if self.total_dims is None:
            total_dims = in_features.shape[1]
        else:
            total_dims = self.total_dims

        assert current_length <= total_dims
        if self.max_nodes is None:
            max_ = in_features.shape[0]
        else:
            max_ = self.max_nodes
        new_features = np.zeros((max_, total_dims), dtype=np.float32)
        new_features[:in_features.shape[0], :current_length] = in_features
        return new_features


class EdgeListToAdjMat(object):
    """
    This class transforms an edge list to an adjacency matrix structure
    bonds start from 1.

    ie [(1,2,2), (0,1,1)] would turn into

    for bond type 1
    0 1 0
    1 0 0
    0 0 0

    for bond type 2
    0 0 0
    0 0 1
    0 1 0

    (these matrices would be stacked on a new third dimension to result in a three dimensional array being returned)
    """
    NUM_BONDS = 4

    def __init__(self, adj_size=None):
        """
        :param adj_size: set to override the adjacency matrix size.
        """
        self.adj_size = adj_size

    def __call__(self, edges_list):
        if self.adj_size is None:
            edges_list_array = np.array(edges_list)
            max_idx = np.max(edges_list_array[:, [0,2]])
            num_atoms = max_idx + 1  # due to starting from zero indexing
        else:
            num_atoms = self.adj_size

        adj_mat = np.zeros([num_atoms, num_atoms, self.NUM_BONDS], dtype=np.float32)
        for src_idx, bond_type, dest_idx in edges_list:
            # Fill in both diagonal halves of the adjacency matrix. Note that the bond type index starts from 1
            adj_mat[src_idx, dest_idx, bond_type-1] = 1.
            adj_mat[dest_idx, src_idx, bond_type-1] = 1.
        return adj_mat


class EdgeListToNeighRelations(object):
    """
    This converts the edge list
    datastructure into a padded  matrix indicating
    neighbour relations (but indexing from 1 and 0 being pad )

    ie [(1,2,2), (0,1,1), (0,1,2)]] would turn into

    {1:
    [[2, 3],
     [1, 0],
     [1, 0]],

     2:
    [[0],
     [3],
     [2]]}
    """
    def __init__(self, shift_in_index=1, bond_types=None):
        self.shift_in_index = shift_in_index
        self.bond_types = bond_types  # overrides finding it.

    def __call__(self, edges_list):
        involved_indcs = [elem for row in edges_list for elem in (row[0], row[2])]
        max_neighs = collections.Counter(involved_indcs).most_common(1)[0][1]
        num_atms = max(involved_indcs) + 1  # plus one due to zero indexing
        if self.bond_types is None:
            bond_types = sorted(list(set([row[1] for row in edges_list])))
        else:
            bond_types = self.bond_types

        neigh_array_dict = {k: np.zeros((num_atms, max_neighs), dtype=int) for k in bond_types}

        for src_idx, bond_type, dest_idx in edges_list:

            neigh_array = neigh_array_dict[bond_type]

            for j in range(max_neighs):
                if neigh_array[src_idx, j] == 0:
                    neigh_array[src_idx, j] = dest_idx + self.shift_in_index
                    break
            else:
                raise RuntimeError

            for j in range(max_neighs):
                if neigh_array[dest_idx, j] == 0:
                    neigh_array[dest_idx, j] = src_idx + self.shift_in_index
                    break
            else:
                raise RuntimeError

        return neigh_array_dict


class EdgeListToAdjList(object):
    """
    This transforms an edge list to an adjacency list

        ie [(1,2,2), (0,1,1), (0,1,2)]] would turn into
    {
        1:[(0, 1), (0, 2)],
        2:[(1, 2)]
    }
    """
    def __init__(self, bond_types):
        self.bond_types = bond_types

    def __call__(self, edges_list):
        results_dict = {k: [] for k in self.bond_types}
        for src_idx, bond_type, dest_idx in edges_list:
            results_dict[bond_type].append((src_idx, dest_idx))
        return results_dict


class Qm9Dataset(data.Dataset):
    def __init__(self, json_file_loc, transform_x=None):
        with open(json_file_loc, 'rb') as fo:
            self.data = json.load(fo)
        self.transform_x = transform_x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = self.data[idx]

        target = np.array([result["targets"][0][0]], dtype=np.float32)
        node_features = np.array(result['node_features'], dtype=np.float32)
        edges = result['graph']

        if self.transform_x is not None:
            x = self.transform_x(edges, node_features)
        else:
            x = (edges, node_features)

        return x, target
