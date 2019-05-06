import pytest
from unittest import mock

from aggtree import *
import numpy as np


def test_AggregationTree_make_leaf_indices():
    aggtree = mock.Mock(spec=AggregationArray)
    aggtree.configure_mock(
        _arraytree=np.empty(34),
        __len__=lambda self: 17,
        index_dtype=np.int64,
        _parent=AggregationArray._parent,
    )

    AggregationArray._make_leaf_indices(aggtree)

    assert aggtree._leaf_indices[0] == 32
    assert aggtree._leaf_indices[1] == 33
    assert aggtree._leaf_indices[2] == 17
    assert aggtree._leaf_indices[-1] == 31


@pytest.mark.skip("unimplemented")
def test_AggregationTree_init():
    aggtree = mock.Mock(spec=AggregationArray)
    AggregationArray.__init__(aggtree, length=17, dtype=np.float64,
                             agg_func=np.add)
    pass


def test_AggregationTree_root_path_indices():
    aggtree = mock.Mock(spec=AggregationArray)
    aggtree.configure_mock(
        index_dtype=np.int64,
    )

    assert (AggregationArray._root_path_indices(aggtree, 5)
            == np.array([5, 2, 1])).all()
    assert (AggregationArray._root_path_indices(aggtree, 6)
            == np.array([6, 3, 1])).all()
    assert (AggregationArray._root_path_indices(aggtree, 14)
            == np.array([14, 7, 3, 1])).all()


def test_AggregationTree_all():
    a = AggregationArray(13, np.float, np.sum, 0)
    assert np.all(a[:] == 0)
    assert len(a) == 13

    for i in range(len(a)):
        a[i] = i
        assert a[i] == i

    for j in range(1, len(a)):
        for i in range(j):
            assert a.aggregate(i, j) == np.sum(a[i:j])
