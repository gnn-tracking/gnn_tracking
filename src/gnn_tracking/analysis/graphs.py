from __future__ import annotations

import itertools
from typing import Iterable, NamedTuple, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data

from gnn_tracking.utils.graph_masks import edge_subgraph


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
        n_hits_largest_component: The number of hits of the track of the biggest
            component of the track.
    """

    pid: int
    n_hits: int
    n_segments: int
    n_hits_largest_segment: int
    distance_largest_segments: int
    n_hits_largest_component: int


def get_track_graph_info(
    graph: nx.Graph, particle_ids: np.ndarray, pid: int
) -> TrackGraphInfo:
    """Get information about how well connected the hits of a single particle are in the
    graph.

    Args:
        graph: networkx graph of the data
        particle_ids: The particle IDs of the hits.
        pid: Particle ID of the true track

    Returns:
        `TrackGraphInfo`
    """
    hits_for_pid = np.where(particle_ids == pid)[0]
    n_hits = len(hits_for_pid)
    assert n_hits > 0
    segment_subgraph = graph.subgraph(hits_for_pid)
    segments: list[Sequence[int]] = sorted(  # type: ignore
        nx.connected_components(segment_subgraph), key=len, reverse=True
    )
    if len(segments) == 1:
        n_hits_largest_component = len(hits_for_pid)
    else:
        # We could also iterate over all PIDs, but that would be slower.
        # we already know that the segments are connected, so it's enough to
        # use one of the nodes from each one.
        component_sizes = [
            1 + get_n_reachable(graph, next(iter(segment)), hits_for_pid)
            for segment in segments
        ]
        n_hits_largest_component = max(component_sizes)
    distance_largest_segments = 0
    if len(segments) > 1:
        distance_largest_segments = shortest_path_length_multi(
            graph, sources=segments[0], targets=segments[1]
        )
        assert distance_largest_segments > 0
    n_hits_largest_segment = len(segments[0])
    assert sum(map(len, segments)) == n_hits
    assert n_hits >= n_hits_largest_component >= n_hits_largest_segment > 0, (
        n_hits,
        n_hits_largest_component,
        n_hits_largest_segment,
    )
    return TrackGraphInfo(
        pid=pid,
        n_hits=n_hits,
        n_segments=len(segments),
        n_hits_largest_segment=n_hits_largest_segment,
        distance_largest_segments=distance_largest_segments,
        n_hits_largest_component=n_hits_largest_component,
    )


def get_track_graph_info_from_data(
    data: Data, w: Tensor, pt_thld=0.9, threshold: float | None = None
) -> pd.DataFrame:
    """Get DataFrame of track graph information for every particle ID in the data.
    This function basically applies `get_track_graph_info` to every particle ID.

    Args:
        data:
        model: Edge classifier model
        pt_thld: pt threshold for particle IDs to consider
        threshold: Edge classification cutoff

    Returns:
        DataFrame with columns as in `TrackGraphInfo`
    """
    assert not torch.isnan(w).any()
    edge_mask = (w > threshold).squeeze()
    data_subgraph = edge_subgraph(data, edge_mask)
    if threshold <= 0:
        assert edge_mask.all()
        assert data_subgraph.num_edges == data.num_edges
    gx = torch_geometric.utils.convert.to_networkx(data_subgraph, to_undirected=True)
    assert gx.number_of_nodes() == data_subgraph.num_nodes, (
        gx.number_of_nodes(),
        data_subgraph.num_nodes,
    )
    # Can't check edge numbers because of directed vs undirected edges
    particle_ids = data.particle_id[
        (data.particle_id > 0) & (data.pt > pt_thld)
    ].unique()
    results = []
    for pid in particle_ids:
        results.append(get_track_graph_info(gx, data.particle_id, pid.item())._asdict())
    return pd.DataFrame.from_records(results)


def summarize_track_graph_info(tgi: pd.DataFrame) -> dict[str, float]:
    """Summarize track graph information returned by
    `get_track_graph_info_from_data`.
    """
    return dict(
        frac_segment100=sum((tgi.n_hits_largest_segment / tgi.n_hits) == 1) / len(tgi),
        frac_component100=sum((tgi.n_hits_largest_component / tgi.n_hits) == 1)
        / len(tgi),
        frac_segment50=sum((tgi.n_hits_largest_segment / tgi.n_hits) >= 0.50)
        / len(tgi),
        frac_component50=sum((tgi.n_hits_largest_component / tgi.n_hits) >= 0.50)
        / len(tgi),
        frac_segment75=sum((tgi.n_hits_largest_segment / tgi.n_hits) >= 0.75)
        / len(tgi),
        frac_component75=sum((tgi.n_hits_largest_component / tgi.n_hits) >= 0.75)
        / len(tgi),
        n_segments=tgi.n_segments.mean(),
        frac_hits_largest_segment=(tgi.n_hits_largest_segment / tgi.n_hits).mean(),
        frac_hits_largest_component=(tgi.n_hits_largest_component / tgi.n_hits).mean(),
    )
