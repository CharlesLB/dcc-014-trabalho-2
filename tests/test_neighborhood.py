import numpy as np
import pytest

from mst.neighborhood import (ADD, REMOVE, SWAP, Move, is_tabu, moves, neighborhood_size,
                              update_memory)


def test_move_apply():
    S = frozenset({1, 2})
    assert Move(ADD, inn=0).apply(S) == {0, 1, 2}
    assert Move(REMOVE, out=1).apply(S) == {2}
    assert Move(SWAP, out=1, inn=3).apply(S) == {2, 3}


def test_move_apply_unknown():
    with pytest.raises(ValueError):
        Move("x").apply(frozenset())


def test_move_as_dict():
    assert Move(SWAP, out=1, inn=2).as_dict() == {"tipo": "swap", "sai": 1, "entra": 2}


@pytest.mark.parametrize("m,k", [(3, 1), (3, 2), (5, 1), (5, 3), (10, 4), (22, 7), (6, 6)])
def test_neighborhood_size_matches_generation(m, k):
    S = frozenset(range(k))
    assert len(list(moves(S, m))) == neighborhood_size(k, m)


@pytest.mark.parametrize("m,k", [(5, 2), (10, 3), (67, 10)])
def test_size_is_m_plus_k_m_minus_k_when_removals_allowed(m, k):
    assert neighborhood_size(k, m) == m + k * (m - k)


def test_no_removal_when_single_variable():
    ms = list(moves(frozenset({2}), 4))
    assert all(mv.kind != REMOVE for mv in ms)
    assert len(ms) == 3 + 3


def test_min_size_zero_allows_removal_to_empty():
    ms = list(moves(frozenset({2}), 4, min_size=0))
    assert any(mv.kind == REMOVE for mv in ms)


def test_order_is_adds_removes_swaps():
    kinds = [mv.kind for mv in moves(frozenset({1, 3}), 5)]
    first_rem = kinds.index(REMOVE)
    first_swap = kinds.index(SWAP)
    assert all(k == ADD for k in kinds[:first_rem])
    assert all(k == REMOVE for k in kinds[first_rem:first_swap])
    assert all(k == SWAP for k in kinds[first_swap:])


def test_all_neighbors_distinct_and_differ_from_S():
    S = frozenset({0, 4, 7})
    nbrs = [mv.apply(S) for mv in moves(S, 9)]
    assert len(set(nbrs)) == len(nbrs)
    assert S not in nbrs


def test_neighbors_have_hamming_distance_1_or_2():
    S = frozenset({0, 4, 7})
    for mv in moves(S, 9):
        d = len(S ^ mv.apply(S))
        assert d == (2 if mv.kind == SWAP else 1)


def test_full_set_has_no_adds_or_swaps():
    S = frozenset(range(4))
    ms = list(moves(S, 4))
    assert all(mv.kind == REMOVE for mv in ms) and len(ms) == 4


def _mem(m, tenure):
    return np.full(m, -tenure), np.full(m, -tenure)


def test_report_example_reinsertion_window():
    # saiu na iteração 12, tenure 4: reinserção tabu de 13 a 16
    vin, vout = _mem(5, 4)
    update_memory(Move(REMOVE, out=2), 12, vin, vout)
    mv = Move(ADD, inn=2)
    assert [is_tabu(mv, it, vin, vout, 4) for it in range(13, 18)] == [True] * 4 + [False]


def test_exit_tabu_after_entry():
    vin, vout = _mem(5, 3)
    update_memory(Move(ADD, inn=1), 5, vin, vout)
    mv = Move(REMOVE, out=1)
    assert is_tabu(mv, 8, vin, vout, 3)
    assert not is_tabu(mv, 9, vin, vout, 3)


def test_swap_tabu_if_either_part_tabu():
    vin, vout = _mem(5, 2)
    update_memory(Move(REMOVE, out=3), 10, vin, vout)  # entrada de 3 tabu
    assert is_tabu(Move(SWAP, out=0, inn=3), 11, vin, vout, 2)
    update_memory(Move(ADD, inn=0), 10, vin, vout)  # saída de 0 tabu
    assert is_tabu(Move(SWAP, out=0, inn=4), 11, vin, vout, 2)
    assert not is_tabu(Move(SWAP, out=1, inn=4), 11, vin, vout, 2)


def test_initial_memory_not_tabu():
    vin, vout = _mem(3, 5)
    for mv in [Move(ADD, inn=0), Move(REMOVE, out=1), Move(SWAP, out=1, inn=2)]:
        assert not is_tabu(mv, 1, vin, vout, 5)


def test_tenure_zero_never_tabu_next_iteration():
    vin, vout = _mem(3, 0)
    update_memory(Move(SWAP, out=0, inn=1), 4, vin, vout)
    assert not is_tabu(Move(SWAP, out=1, inn=0), 5, vin, vout, 0)


def test_update_memory_swap_sets_both():
    vin, vout = _mem(4, 1)
    update_memory(Move(SWAP, out=2, inn=3), 7, vin, vout)
    assert vin[3] == 7 and vout[2] == 7
