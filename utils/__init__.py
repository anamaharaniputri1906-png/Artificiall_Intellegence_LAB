from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import radians, sin, cos, asin, sqrt
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass
class SearchResult:
    found: bool
    path: list
    hops: int
    path_cost_km: float
    expanded: int
    runtime_ms: float

    def as_dict(self):
        return {
            "found": self.found,
            "hops": self.hops,
            "path_cost_km": self.path_cost_km,
            "expanded": self.expanded,
            "runtime_ms": self.runtime_ms,
        }


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates, in kilometres."""
    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371.0088 * 2 * asin(sqrt(a))


def load_openflights(airports_path, routes_path):
    """Load OpenFlights CSV files and construct a directed weighted graph."""

    # Airport file has no header.
    airports = pd.read_csv(airports_path, header=None, dtype={4: str})

    airports = airports.rename(columns={
        0: "airport_id",
        1: "name",
        2: "city",
        3: "country",
        4: "iata",
        6: "latitude",
        7: "longitude",
    })

    airports["iata"] = airports["iata"].str.strip().str.upper()
    airports["latitude"] = pd.to_numeric(airports["latitude"], errors="coerce")
    airports["longitude"] = pd.to_numeric(airports["longitude"], errors="coerce")

    airports = airports.dropna(subset=["iata", "latitude", "longitude"])
    airports = airports[
        airports["iata"].str.len() == 3
    ].copy()

    coords = {
        row.iata: {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        for row in airports.itertuples()
    }

    airport_names = dict(zip(airports["iata"], airports["name"]))
    graph = {code: [] for code in coords}

    # Route file has a header, but some column names contain spaces/typos.
    routes = pd.read_csv(routes_path)
    routes.columns = routes.columns.str.strip().str.lower()

    source_col = "source airport"
    destination_col = "destination apirport"

    if destination_col not in routes.columns:
        destination_col = "destination airport"

    routes[source_col] = routes[source_col].astype("string").str.strip().str.upper()
    routes[destination_col] = (
        routes[destination_col].astype("string").str.strip().str.upper()
    )

    # Avoid duplicate edges between the same pair of airports.
    edges = {}

    for source, destination in zip(
        routes[source_col], routes[destination_col]
    ):
        if source not in coords or destination not in coords:
            continue

        distance = haversine_km(
            coords[source]["latitude"],
            coords[source]["longitude"],
            coords[destination]["latitude"],
            coords[destination]["longitude"],
        )

        edges[(source, destination)] = distance

    for (source, destination), distance in edges.items():
        graph[source].append((destination, distance))

    return airports, routes, coords, graph, airport_names


def _search(graph, start, goal, heuristic=None, mode="ucs"):
    started = perf_counter()

    if start not in graph or goal not in graph:
        raise ValueError(f"Unknown airport: {start if start not in graph else goal}")

    if heuristic is None:
        heuristic = lambda node: 0.0

    if start == goal:
        return SearchResult(True, [start], 0, 0.0, 0,
                            (perf_counter() - started) * 1000)

    tie = count()
    best_g = {start: 0.0}
    parent = {}
    closed = set()

    initial_priority = heuristic(start) if mode == "gbfs" else 0.0
    frontier = [(initial_priority, next(tie), 0.0, start)]
    expanded = 0

    while frontier:
        priority, _, current_g, current = heappop(frontier)

        if mode == "gbfs":
            if current in closed:
                continue
        elif current_g > best_g.get(current, np.inf) + 1e-10:
            continue

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()

            return SearchResult(
                True,
                path,
                len(path) - 1,
                current_g,
                expanded,
                (perf_counter() - started) * 1000,
            )

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        for neighbour, edge_cost in graph[current]:
            new_g = current_g + edge_cost

            if new_g + 1e-10 >= best_g.get(neighbour, np.inf):
                continue

            best_g[neighbour] = new_g
            parent[neighbour] = current

            if mode == "gbfs":
                new_priority = heuristic(neighbour)
            elif mode == "astar":
                new_priority = new_g + heuristic(neighbour)
            else:
                new_priority = new_g

            # Allow reopening a state if a better path is found.
            closed.discard(neighbour)

            heappush(
                frontier,
                (new_priority, next(tie), new_g, neighbour)
            )

    return SearchResult(
        False, [], 0, float("inf"), expanded,
        (perf_counter() - started) * 1000
    )


def ucs(graph, start, goal):
    return _search(graph, start, goal, mode="ucs")


def gbfs(graph, start, goal, heuristic):
    return _search(graph, start, goal, heuristic, mode="gbfs")


def astar(graph, start, goal, heuristic):
    return _search(graph, start, goal, heuristic, mode="astar")

from collections import deque


def bfs(graph, start, goal):
    started = perf_counter()

    if start not in graph or goal not in graph:
        raise ValueError(f"Unknown airport: {start if start not in graph else goal}")

    queue = deque([(start, [start], 0.0)])
    visited = {start}
    expanded = 0

    while queue:
        current, path, cost = queue.popleft()

        if current == goal:
            return SearchResult(
                True, path, len(path) - 1, cost,
                expanded, (perf_counter() - started) * 1000
            )

        expanded += 1

        for neighbour, edge_cost in graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((
                    neighbour,
                    path + [neighbour],
                    cost + edge_cost
                ))

    return SearchResult(
        False, [], 0, float("inf"), expanded,
        (perf_counter() - started) * 1000
    )


def dfs(graph, start, goal):
    started = perf_counter()

    if start not in graph or goal not in graph:
        raise ValueError(f"Unknown airport: {start if start not in graph else goal}")

    stack = [(start, [start], 0.0)]
    visited = {start}
    expanded = 0

    while stack:
        current, path, cost = stack.pop()

        if current == goal:
            return SearchResult(
                True, path, len(path) - 1, cost,
                expanded, (perf_counter() - started) * 1000
            )

        expanded += 1

        for neighbour, edge_cost in reversed(graph[current]):
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append((
                    neighbour,
                    path + [neighbour],
                    cost + edge_cost
                ))

    return SearchResult(
        False, [], 0, float("inf"), expanded,
        (perf_counter() - started) * 1000
    )

__all__ = [
    "SearchResult",
    "haversine_km",
    "load_openflights",
    "bfs",
    "dfs",
    "ucs",
    "gbfs",
    "astar",
]