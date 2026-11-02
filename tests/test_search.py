import pytest

from utils import (
    astar, bfs, dfs, gbfs, ucs,
    haversine_km
)

# Small five-state graph for controlled testing
GRAPH = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 1), ("D", 5)],
    "C": [("D", 1), ("E", 7)],
    "D": [("E", 1)],
    "E": []
}

HEURISTIC_VALUES = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
    "E": 0
}

def heuristic(node):
    return HEURISTIC_VALUES[node]


def test_haversine_zero_distance():
    assert haversine_km(0, 0, 0, 0) == 0


def test_haversine_symmetry():
    distance_1 = haversine_km(3, 101, 4, 102)
    distance_2 = haversine_km(4, 102, 3, 101)
    assert distance_1 == pytest.approx(distance_2)


def test_bfs_finds_shortest_hop_path():
    result = bfs(GRAPH, "A", "E")
    assert result.found
    assert result.hops == 2
    assert result.path == ["A", "C", "E"]


def test_dfs_finds_valid_path():
    result = dfs(GRAPH, "A", "E")
    assert result.found
    assert result.path[0] == "A"
    assert result.path[-1] == "E"


def test_ucs_finds_lowest_cost_path():
    result = ucs(GRAPH, "A", "E")
    assert result.found
    assert result.path_cost_km == pytest.approx(4)


def test_gbfs_finds_a_route():
    result = gbfs(GRAPH, "A", "E", heuristic)
    assert result.found
    assert result.path[0] == "A"
    assert result.path[-1] == "E"


def test_astar_matches_ucs():
    ucs_result = ucs(GRAPH, "A", "E")
    astar_result = astar(GRAPH, "A", "E", heuristic)
    assert astar_result.path_cost_km == pytest.approx(
        ucs_result.path_cost_km
    )


def test_ucs_rejects_unknown_airport():
    with pytest.raises(ValueError):
        ucs(GRAPH, "X", "E")