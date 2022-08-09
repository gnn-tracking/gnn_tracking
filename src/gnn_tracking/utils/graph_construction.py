import os
import sys

sys.path.append("../")
import logging
from collections import Counter
from os.path import join

import numpy as np
import pandas as pd

from gnn_tracking.utils.preprocessing import relabel_pids as relabel_pid_func


def initialize_logger(verbose=False):
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Initializing")


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi
    return dphi


def calc_eta(r, z):
    """Computes pseudorapidity
    (https://en.wikipedia.org/wiki/Pseudorapidity)
    """
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def empty_graph(s):
    singles = {k: np.array([]) for k in ["x", "hit_id", "particle_id", "y"]}
    doubles = {k: np.array([[], []]) for k in ["edge_index", "edge_hit_id"]}
    quadrouples = {"edge_attr": np.array([[], [], [], []])}
    graph = {**singles, **doubles, **quadrouples}
    graph["s"] = s
    graph["n_incorrect"] = 0
    return graph


def split_detector_sectors(
    hits, phi_edges, eta_edges, verbose=False, phi_overlap=0.1, eta_overlaps=0.1
):

    """Split hits according to provided phi and eta boundaries."""
    hits_sectors = {}
    sector_info = {}
    for i in range(len(phi_edges) - 1):
        phi_min = phi_edges[i]
        phi_max = phi_edges[i + 1]

        # simple check that we're in the phi sector
        in_phi_sector = (hits.phi > phi_min) & (hits.phi < phi_max)

        # select hits in this phi sector
        phi_hits = hits[in_phi_sector]
        phi_hits = phi_hits.assign(old_phi=phi_hits.phi)

        # deterime upper and lower overlap regions
        lower_overlap = (hits.phi > (phi_min - phi_overlap)) & (hits.phi < phi_min)
        upper_overlap = (hits.phi < (phi_max + phi_overlap)) & (hits.phi > phi_max)

        # interior overlap accounts for where other sectors will intrude
        interior_overlap = (
            (phi_hits.phi > phi_min) & (phi_hits.phi < (phi_min + phi_overlap))
        ) | ((phi_hits.phi > (phi_max - phi_overlap)) & (phi_hits.phi < phi_max))

        # special case: regions bounded at +/-pi branch cut
        if abs(phi_min - (-np.pi)) < 0.01:
            lower_overlap = (hits.phi > (np.pi - phi_overlap)) & (hits.phi < np.pi)
            upper_overlap = (hits.phi < (phi_max + phi_overlap)) & (hits.phi > phi_max)
        if abs(phi_max - np.pi) < 0.01:
            lower_overlap = (hits.phi > (phi_min - phi_overlap)) & (hits.phi < phi_min)
            upper_overlap = (hits.phi > -np.pi) & (hits.phi < (-np.pi + phi_overlap))

        # select hits in overlapped region
        lower_overlap_hits = hits[lower_overlap]
        upper_overlap_hits = hits[upper_overlap]
        lower_phi = lower_overlap_hits.phi
        lower_overlap_hits = lower_overlap_hits.assign(old_phi=lower_phi)
        upper_phi = upper_overlap_hits.phi
        upper_overlap_hits = upper_overlap_hits.assign(old_phi=upper_phi)

        # adjust values across +/-pi branch cut
        if abs(phi_min - (-np.pi)) < 0.01:
            new_lower_phi = -2 * np.pi + lower_overlap_hits.phi
            lower_overlap_hits = lower_overlap_hits.assign(phi=new_lower_phi)
        if abs(phi_max - np.pi) < 0.01:
            new_upper_phi = 2 * np.pi + upper_overlap_hits.phi
            upper_overlap_hits = upper_overlap_hits.assign(phi=new_upper_phi)

        # center these hits on phi=0
        phi_hits = phi_hits.assign(overlapped=np.zeros(len(phi_hits), dtype=bool))
        phi_hits.loc[interior_overlap, "overlapped"] = True
        lower_overlap_hits = lower_overlap_hits.assign(overlapped=True)
        upper_overlap_hits = upper_overlap_hits.assign(overlapped=True)
        phi_hits = phi_hits.append(lower_overlap_hits)
        phi_hits = phi_hits.append(upper_overlap_hits)
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2.0
        phi_hits = phi_hits.assign(phi=centered_phi, phi_sector=i)

        # loop over eta ranges
        for j in range(len(eta_edges) - 1):
            eta_min = eta_edges[j]
            eta_max = eta_edges[j + 1]
            eta_overlap = eta_overlaps

            # select hits in this eta sector
            eta = calc_eta(phi_hits.r, phi_hits.z)
            in_eta_overlap = ((eta > eta_min - eta_overlap) & (eta < eta_min)) | (
                (eta < eta_max + eta_overlap) & (eta > eta_max)
            )
            interior_eta_overlap = (
                (eta > eta_min) & (eta < (eta_min + eta_overlap))
            ) | ((eta > (eta_max - eta_overlap)) & (eta < eta_max))
            sec_hits = phi_hits[(in_eta_overlap | ((eta > eta_min) & (eta < eta_max)))]
            sec_hits = sec_hits.assign(
                overlapped=(
                    sec_hits.overlapped & (in_eta_overlap | interior_eta_overlap)
                )
            )

            # label hits by tuple s = (eta_sector, phi_sector)
            hits_sectors[(j, i)] = sec_hits.assign(eta_sector=j)

            # store eta and phi ranges per sector
            sector_info[(j, i)] = {
                "eta_range": [eta_min, eta_max],
                "phi_range": [phi_min, phi_max],
                "eta_overlap": eta_overlap,
                "phi_overlap": phi_overlap,
            }
            #'nhits': len(sec_hits)}

            if verbose:
                logging.info(
                    f"Sector ({i},{j}):\n"
                    + f"...eta_range=({eta_min:.3f},{eta_max:.3f})\n"
                    f"...phi_range=({phi_min:.3f},{phi_max:.3f})"
                )

    return hits_sectors, sector_info


def select_hits(
    hits: pd.DataFrame,
    truth: pd.DataFrame,
    particles: pd.DataFrame,
    pt_min=0,
    endcaps=False,
    remove_noise=False,
    remove_duplicates=False,
    relabel_pids=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Barrel volume and layer ids
    vlids = [(8, 2), (8, 4), (8, 6), (8, 8)]
    if endcaps:
        vlids.extend(
            [
                (7, 14),
                (7, 12),
                (7, 10),
                (7, 8),
                (7, 6),
                (7, 4),
                (7, 2),
                (9, 2),
                (9, 4),
                (9, 6),
                (9, 8),
                (9, 10),
                (9, 12),
                (9, 14),
            ]
        )
    n_det_layers = len(vlids)

    # select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(["volume_id", "layer_id"])
    hits = pd.concat(
        [vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)]
    )

    # calculate particle transverse momentum
    particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
    particles["eta_pt"] = calc_eta(particles.pt, particles.pz)

    # true particle selection.
    particles = particles[particles.pt > pt_min]
    truth_noise = truth[["hit_id", "particle_id"]][truth.particle_id == 0]
    truth_noise["pt"] = 0
    truth = truth[["hit_id", "particle_id"]].merge(
        particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]], on="particle_id"
    )

    # optionally add noise
    if not remove_noise:
        truth = truth.append(truth_noise)

    # calculate derived hits variables
    hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
    hits["phi"] = np.arctan2(hits.y, hits.x)
    hits["eta"] = calc_eta(hits.r, hits.z)
    hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

    # select the data columns we need
    hits = hits[
        ["hit_id", "r", "phi", "eta", "x", "y", "z", "u", "v", "layer", "module_id"]
    ].merge(
        truth[["hit_id", "particle_id", "pt", "eta_pt", "q", "vx", "vy"]], on="hit_id"
    )

    # optionally remove duplicates
    if remove_duplicates:
        noise_hits = hits[hits.particle_id == 0]
        particle_hits = hits[hits.particle_id != 0]
        particle_hits = particle_hits.loc[
            particle_hits.groupby(["particle_id", "layer"]).r.idxmin()
        ]
        hits = particle_hits.append(noise_hits)

    if relabel_pids:
        hits, particles = relabel_pid_func(hits, particles)
    return hits, particles


def select_edges(
    hits1,
    hits2,
    layer1,
    layer2,
    phi_slope_max,
    z0_max,
    dR_max=0.5,
    uv_approach_max=100,
    module_map=[],
    use_module_map=False,
):

    # start with all possible pairs of hits
    keys = ["evtid", "r", "phi", "z", "u", "v", "module_id", "overlapped"]
    hit_pairs = (
        hits1[keys]
        .reset_index()
        .merge(hits2[keys].reset_index(), on="evtid", suffixes=("_1", "_2"))
    )

    # compute geometric features of the line through each hit pair
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    eta_1 = calc_eta(hit_pairs.r_1, hit_pairs.z_1)
    eta_2 = calc_eta(hit_pairs.r_2, hit_pairs.z_2)
    deta = eta_2 - eta_1
    dR = np.sqrt(deta**2 + dphi**2)
    du = hit_pairs.u_2 - hit_pairs.u_1
    dv = hit_pairs.v_2 - hit_pairs.v_1
    m = dv / du
    b = hit_pairs.v_1 - m * hit_pairs.u_1
    u_approach = -m * b / (1 + m**2)
    v_approach = u_approach * m + b
    uv_approach = np.sqrt(u_approach**2 + v_approach**2)

    # phi_slope and z0 used to filter spurious edges
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

    # apply the intersecting line cut
    intersected_layer = dr.abs() < -1
    # 0th barrel layer to left EC or right EC
    if (layer1 == 0) and (layer2 == 11 or layer2 == 4):
        z_coord = 71.56298065185547 * dz / dr + z0
        intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)
    # 1st barrel layer to the left EC or right EC
    if (layer1 == 1) and (layer2 == 11 or layer2 == 4):
        z_coord = 115.37811279296875 * dz / dr + z0
        intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)

    # mask edges not in the module map
    if use_module_map:
        mid1 = hit_pairs.module_id_1.values
        mid2 = hit_pairs.module_id_2.values
        in_module_map = module_map[mid1, mid2]
    else:
        in_module_map = np.ones(z0.shape, dtype=bool)

    # filter edges according to selection criteria
    good_edge_mask = (
        (phi_slope.abs() < phi_slope_max)
        & (z0.abs() < z0_max)  # geometric
        & (dR < dR_max)
        & (uv_approach < uv_approach_max)
        & (intersected_layer == False)
        & (in_module_map)
    )  # data-driven

    # store edges (in COO format) and geometric edge features
    selected_edges = pd.DataFrame(
        {
            "index_1": hit_pairs["index_1"][good_edge_mask],
            "index_2": hit_pairs["index_2"][good_edge_mask],
            "dr": dr[good_edge_mask],
            "dphi": dphi[good_edge_mask],
            "dz": dz[good_edge_mask],
            "dR": dR[good_edge_mask],
        }
    )

    return selected_edges


def correct_truth_labels(hits, edges, y, particle_ids):
    """Corrects for extra edges surviving the barrel intersection
    cut, i.e. for each particle counts the number of extra
    "transition edges" crossing from a barrel layer to an
    innermost endcap slayer; the sum is n_incorrect
    - [edges] = n_edges x 2
    - [y] = n_edges
    - [particle_ids] = n_edges
    """
    logging.info(f"Initially {int(np.sum(y))}/{len(y)} true edges.")

    # layer indices for barrel-to-endcap edges
    barrel_to_endcaps = {
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),  # barrel to l-EC
        (0, 11),
        (1, 11),
        (2, 11),
        (3, 11),
    }  # barrel to r-EC
    precedence = {
        (0, 4): 0,
        (1, 4): 1,
        (2, 4): 2,
        (3, 4): 3,
        (0, 11): 0,
        (1, 11): 1,
        (2, 11): 2,
        (3, 11): 3,
    }

    # group hits by particle id, get layer indices
    hits_by_particle = hits.groupby("particle_id")
    layers_1 = hits.layer.loc[edges.index_1].values
    layers_2 = hits.layer.loc[edges.index_2].values

    # loop over particle_id, particle_hits,
    # count extra transition edges as n_incorrect
    n_corrected = 0
    for p, particle_hits in hits_by_particle:
        if p == 0:
            continue
        particle_hit_ids = particle_hits["hit_id"].values

        # grab true segment indices for particle p
        relevant_indices = (particle_ids == p) & (y == 1)

        # get layers connected by particle's edges
        particle_l1 = layers_1[relevant_indices]
        particle_l2 = layers_2[relevant_indices]
        layer_pairs = set(zip(particle_l1, particle_l2))

        # count the number of transition edges between barrel/endcaps
        transition_edges = layer_pairs.intersection(barrel_to_endcaps)
        if len(transition_edges) > 1:
            transition_edges = list(transition_edges)
            edge_precedence = np.array([precedence[e] for e in transition_edges])
            max_precedence = np.amax(edge_precedence)
            to_relabel = np.array(transition_edges)[(edge_precedence < max_precedence)]
            for l1, l2 in to_relabel:
                relabel = (layers_1 == l1) & (layers_2 == l2) & relevant_indices
                relabel_idx = np.where(relabel == True)[0]
                y[relabel_idx] = 0
                n_corrected += len(relabel_idx)

    logging.info(f"Relabeled {n_corrected} edges crossing from barrel to endcaps.")
    logging.info(f"Updated y has {int(np.sum(y))}/{len(y)} true edges.")
    return y, n_corrected


def get_particle_properties(particle_id, df):
    return df[df.particle_id == particle_id].squeeze()


def get_n_track_segs(hits_by_particle, valid_connections):
    """Calculates the number of track segments present in
    a subset of hits generated by a particle
    (used for analyzing efficiency per sector)
    """
    # loop over particle_ids and corresponding particle hits
    n_track_segs = {}
    for particle_id, particle_hits in hits_by_particle:

        # noise doesn't produce true edges
        if particle_id == 0:
            n_track_segs[particle_id] = 0
            continue

        # store hit multiplicity per layer
        layers_hit = particle_hits.layer.values
        hits_per_layer = Counter(layers_hit)
        layers = np.unique(layers_hit)

        # single-hits don't produce truth edges
        if len(layers) == 1:
            n_track_segs[particle_id] = 0
            continue

        # all edges must be valid for a reconstructable particle
        layer_pairs = set(zip(layers[:-1], layers[1:]))

        # total number of true edges produced by particle
        good_layer_pairs = layer_pairs.intersection(valid_connections)
        count = 0
        for good_lp in good_layer_pairs:
            count += hits_per_layer[good_lp[0]] * hits_per_layer[good_lp[1]]
        n_track_segs[particle_id] = count

    return n_track_segs


def graph_summary(
    evtid, sectors, particle_properties, sector_info, print_per_layer=False
):
    """Calculates per-sector and per-graph summary stats
    and returns a dictionary for subsequent analysis
     - total_track_segs: # track segments (true edges) possible
     - total_nodes: # nodes present in graph / sector
     - total_edges: # edges present in graph / sector
     - total_true: # true edges present in graph / sector
     - total_false # false edges present in graph / sector
     - boundary_fraction: fraction of track segs lost between sectors

    """
    # truth number of track segments possible
    track_segs = particle_properties["n_track_segs"].values()
    total_track_segs = np.sum(list(track_segs))
    total_track_segs_sectored = 0

    # reconstructed quantities per graph
    total_nodes, total_edges = 0, 0
    total_true, total_false = 0, 0

    # helper function for division by 0
    def div(a, b):
        return float(a) / b if b else 0

    # loop over graph sectors and compile statistics
    sector_stats = {}
    total_possible_per_s = 0
    all_edges, all_truth = [], []
    for i, sector in enumerate(sectors):

        # get information about the graph's sector
        s = sector["s"]  # s = sector label
        sector_ranges = sector_info[s]

        # catalogue all edges by id
        all_edges.extend(np.transpose(sector["edge_hit_id"]).tolist())
        all_truth.extend(sector["y"].tolist())

        # calculate graph properties
        n_nodes = sector["x"].shape[0]
        total_nodes += n_nodes
        # correct n_edges for multiple transition edges
        # (see check_truth_labels())
        n_true = np.sum(sector["y"]) - sector["n_incorrect"]
        total_true += n_true
        n_false = np.sum(sector["y"] == 0)
        total_false += n_false
        n_edges = len(sector["y"])
        total_edges += n_edges

        # calculate track segments in sector
        n_track_segs_per_pid = particle_properties["n_track_segs_per_s"][s]
        n_track_segs = np.sum(list(n_track_segs_per_pid.values()))
        total_track_segs_sectored += n_track_segs

        # estimate purity in each sector
        sector_stats[i] = {
            "eta_range": sector_ranges["eta_range"],
            "phi_range": sector_ranges["phi_range"],
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "purity": div(n_true, n_edges),
            "efficiency": div(n_true, n_track_segs),
        }

    # count duplicated true and false edges
    all_true_edges = [
        frozenset(ae) for i, ae in enumerate(all_edges) if all_truth[i] == 1
    ]
    all_false_edges = [
        frozenset(ae) for i, ae in enumerate(all_edges) if all_truth[i] == 0
    ]
    te_counts = np.array(list(Counter(all_true_edges).values()))
    fe_counts = np.array(list(Counter(all_false_edges).values()))
    te_excess = np.sum(te_counts[te_counts > 1] - 1)
    fe_excess = np.sum(fe_counts[fe_counts > 1] - 1)

    # proportion of true edges to all possible track segments
    efficiency = div(total_true - te_excess, total_track_segs)
    # proportion of true edges to total reconstructed edges
    purity = div(total_true, total_edges)
    purity_corr = div(total_true - te_excess, total_edges - te_excess - fe_excess)
    # proportion of true track segments lost in sector boundaries
    boundary_fraction = div(
        total_track_segs - total_track_segs_sectored, total_track_segs
    )

    logging.info(
        f"Event {evtid}, graph summary statistics\n"
        + f"...total nodes: {total_nodes}\n"
        + f"...total edges: {total_edges}\n"
        + f"...efficiency: {efficiency:.5f}\n"
        + f"...purity: {purity:.5f}\n"
        + f"...purity_corr: {purity_corr:.5f}\n"
        + f"...duplicated true edges: {te_excess}\n"
        + f"...duplicated false edges: {fe_excess}\n"
        + f"...boundary edge fraction: {boundary_fraction:.5f}"
    )

    return {
        "n_nodes": total_nodes,
        "n_edges": total_edges,
        "efficiency": efficiency,
        "purity": purity,
        "boundary_fraction": boundary_fraction,
        "sector_stats": sector_stats,
    }
