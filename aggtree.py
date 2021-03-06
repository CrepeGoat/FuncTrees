import functools

import numpy as np


class AggregationArray:
    @staticmethod
    def _parent(index):
        return index >> 1

    @staticmethod
    def _lchild(index):
        return index << 1

    @staticmethod
    def _rchild(index):
        return (index << 1) + 1

    @staticmethod
    def _is_lchild(index):
        return np.bitwise_and(index, 1) == 0

    @staticmethod
    def _is_rchild(index):
        return np.bitwise_and(index, 1) == 1

    @staticmethod
    def _node_level(index):
        return int(index).bit_length()

    def __init__(self, length, dtype, agg_func, agg_identity=None):
        self._agg_func = agg_func
        if agg_identity is not None:
            self._agg_func.identity = agg_identity

        array_len = 2*length  # 0th element unused
        self._arraytree = np.full((array_len,), agg_func.identity,
                                  dtype=np.dtype(dtype))
        self.index_dtype = np.int64

        self._make_leaf_indices()

    def _make_leaf_indices(self):
        level_start = 1 << (self._node_level(len(self._arraytree)-1)-1)
        level_stop = len(self._arraytree)
        level_length = level_stop - level_start

        indices = np.empty(len(self), dtype=self.index_dtype)
        indices[:level_length] = np.arange(level_start, level_stop)
        indices[level_length:] = np.arange(self._parent(level_stop), level_start)
        self._leaf_indices = indices

    def _root_path_factory(make_ufunc):
        ufunc, dtype = make_ufunc()

        @functools.wraps(make_ufunc)
        def wrapper(self, leaf, maxlevels=None):
            # TODO make array case more efficient
            # - currently O(k*log(k)))?, can be O(k)
            no_of_levels = self._node_level(len(self._arraytree)-1)
            if maxlevels is not None:
                no_of_levels = min(no_of_levels, maxlevels)

            leaf = np.asarray(leaf)
            indices = np.ones((no_of_levels,) + leaf.shape,
                              dtype=self.index_dtype)
            indices[0] = leaf

            if dtype is None:
                return ufunc.accumulate(indices, axis=0)
            else:
                return (ufunc
                        .accumulate(indices, axis=0, dtype=dtype)
                        .astype(self.index_dtype))

        return wrapper

    @_root_path_factory
    def _root_path_indices():
        """ Returns the indices in `arraytree` corresponding to the ancestors
            of the item at the given leaf index.
        """
        return np.right_shift, None

    @_root_path_factory
    def _left_root_path_indices():
        def move_up(leaf, _):  # `_` is a stub to use accumulate
            return np.right_shift(np.subtract(leaf, 1), 1)
        return np.frompyfunc(move_up, 2, 1), np.object

    @_root_path_factory
    def _right_root_path_indices():
        def move_up(leaf, _):  # `_` is a stub to use accumulate
            return np.right_shift(np.add(leaf, 1), 1)
        return np.frompyfunc(move_up, 2, 1), np.object

    def __len__(self):
        return len(self._arraytree) // 2

    def __getitem__(self, index):
        return self._arraytree[self._leaf_indices[index]]

    def __setitem__(self, index, value):
        leaf = self._leaf_indices[index]
        self._arraytree[leaf] = value

        nodes = np.unique(self._root_path_indices(leaf)[1:])[::-1]
        if nodes[-1] == 0:
            nodes = nodes[:-1]

        for i in nodes:
            self._arraytree[i] = self._agg_func([
                self._arraytree[self._lchild(i)],
                self._arraytree[self._rchild(i)],
            ])

    def aggregate(self, left_index, right_index):
        result = self._agg_func.identity
        if left_index >= right_index:
            return result

        maxlevels = self._node_level(right_index - left_index)
        right_index -= 1  # make indices inclusive
        left_leaf, right_leaf = self._leaf_indices[[left_index, right_index]]

        leaves_are_on_same_level = (left_leaf <= right_leaf)
        if not leaves_are_on_same_level:
            if self._is_rchild(left_leaf):
                result = self._arraytree[left_leaf]
                left_leaf = self._parent(left_leaf+1)
            else:
                left_leaf = self._parent(left_leaf)

        left_path = self._right_root_path_indices(left_leaf, maxlevels)
        right_path = self._left_root_path_indices(right_leaf, maxlevels)

        last_level = np.flatnonzero(left_path >= right_path)[0]
        if left_path[last_level] == right_path[last_level]:
            last_level += 1
        left_path = left_path[:last_level]
        right_path = right_path[:last_level]

        left_nodes = left_path[self._is_rchild(left_path)]
        right_nodes = right_path[self._is_lchild(right_path)]

        elements = [result]
        if len(left_nodes):
            elements.append(self._agg_func(self._arraytree[left_nodes]))
        if len(right_nodes):
            elements.append(self._agg_func(self._arraytree[right_nodes]))

        return self._agg_func(elements)
