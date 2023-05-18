from __future__ import annotations

import itertools
from typing import Iterable, NamedTuple, Sequence

import networkx as nx
import numpy as np


def shortest_path_length_catch_no_path(graph: nx.Graph, source, target) -> int | float:
    """Same as nx.shortest_path_length but return inf if no path exists"""
    try:
        return nx.shortest_path_length(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return float("inf")


def shortest_path_length_multi(
    graph: nx.Graph, sources: Iterable[int], targets: Iterable[int]
):
    """Shortest path for source to reach any of targets from any of the sources.
    If no connection exists, returns inf. If only target is source itself, returns 0.
    """
    if set(sources) == set(targets):
        return 0
    targets = set(targets) - set(sources)
    return min(
        [
            shortest_path_length_catch_no_path(graph, source=source, target=target)
            for source, target in itertools.product(sources, targets)
        ]
    )


def get_n_reachable(graph: nx.Graph, source: int, targets: Sequence[int]) -> int:
    """Get the number of targets that are reachable from source. The source node itself
    will not be counted!
    """
    targets = set(targets) - {source}
    return sum([nx.has_path(graph, source=source, target=target) for target in targets])


class TrackGraphInfo(NamedTuple):
    """Information about how well connected the hits of a track are in the graph.

    Here, "component" means connected component of the graph.
    "segment" means connected component of the graph that only contains hits of the
    track with the given particle ID.

    Attributes:
        pid: The particle ID of the track.
        n_hits: The number of hits in the track.
        n_segments: The number of segments of the track.
        n_hits_largest_segment: The number of hits in the largest segment of the track.
        distance_largest_segments: The shortest path length between the two largest
            segments
        biggest_component: The number of hits of the track of the biggest component of
            the track.
    """

    pid: int
    n_hits: int
    n_segments: int
    n_hits_largest_segment: int
    distance_largest_segments: int
    biggest_component: int


def get_track_graph_info(
    graph: nx.Graph, particle_ids: Sequence[int], pid: int
) -> TrackGraphInfo:
    hits_for_pid = np.where(particle_ids == pid)[0]
    assert len(hits_for_pid) > 0
    sg = graph.subgraph(hits_for_pid).to_undirected()
    segments: list[Sequence[int]] = sorted(  # type: ignore
        nx.connected_components(sg), key=len, reverse=True
    )
    if len(segments) == 1:
        biggest_component = len(hits_for_pid)
    else:
        # We could also iterate over all PIDs, but that would be slower.
        # we already know that the segments are connected, so it's enough to
        # use one of the nodes from each one.
        biggest_component = 1 + max(
            get_n_reachable(graph, next(iter(segment)), hits_for_pid)
            for segment in segments
        )
    distance_largest_segments = 0
    if len(segments) > 1:
        distance_largest_segments = shortest_path_length_multi(
            graph, sources=segments[0], targets=segments[1]
        )
    return TrackGraphInfo(
        pid=pid,
        n_hits=len(hits_for_pid),
        n_segments=len(segments),
        n_hits_largest_segment=len(segments[0]),
        distance_largest_segments=distance_largest_segments,
        biggest_component=biggest_component,
    )
