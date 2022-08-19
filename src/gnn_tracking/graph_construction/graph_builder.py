from __future__ import annotations

import collections
import os
from os.path import join as join

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class GraphBuilder:
    def __init__(
        self,
        indir,
        outdir,
        pixel_only=True,
        redo=True,
        phi_slope_max=0.005,
        z0_max=200,
        dR_max=1.7,
        uv_approach_max=0.0015,
        feature_names=["r", "phi", "z", "eta_rz", "u", "v"],
        feature_scale=np.array([1000.0, np.pi, 1000.0, 1, 1 / 1000.0, 1 / 1000.0]),
        directed=False,
        measurement_mode=False,
    ):
        self.indir = indir
        self.outdir = outdir
        self.pixel_only = pixel_only
        self.redo = redo
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.dR_max = dR_max
        self.uv_approach_max = uv_approach_max
        self.feature_names = feature_names
        self.feature_scale = feature_scale
        self.data_list = []
        self.outfiles = os.listdir(outdir)
        self.directed = directed
        self.measurement_mode = measurement_mode

    def calc_dphi(self, phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Computes phi2-phi1 given in range [-pi,pi]"""
        dphi = phi2 - phi1
        dphi[dphi > np.pi] -= 2 * np.pi
        dphi[dphi < -np.pi] += 2 * np.pi
        return dphi

    def calc_eta(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes pseudorapidity
        (https://en.wikipedia.org/wiki/Pseudorapidity)
        """
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def get_dataframe(self, evt, evtid):
        to_df = {"evtid": evtid}
        for i, n in enumerate(self.feature_names):
            to_df[n] = evt.x[:, i]
        to_df["layer"] = evt.layer
        to_df["pt"] = evt.pt
        to_df["particle_id"] = evt.particle_id
        return pd.DataFrame(to_df)

    def select_edges(self, hits1, hits2, layer1, layer2):
        hit_pairs = hits1.reset_index().merge(
            hits2.reset_index(), on="evtid", suffixes=("_1", "_2")
        )

        # define various geometric quantities
        dphi = self.calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        eta_1 = self.calc_eta(hit_pairs.r_1, hit_pairs.z_1)
        eta_2 = self.calc_eta(hit_pairs.r_2, hit_pairs.z_2)
        deta = eta_2 - eta_1
        dR = np.sqrt(deta**2 + dphi**2)
        du = hit_pairs.u_2 - hit_pairs.u_1
        dv = hit_pairs.v_2 - hit_pairs.v_1
        m = dv / du
        b = hit_pairs.v_1 - m * hit_pairs.u_1
        u_approach = -m * b / (1 + m**2)
        v_approach = u_approach * m + b

        # restrict the distance of closest approach in the uv plane
        uv_approach = np.sqrt(u_approach**2 + v_approach**2)

        # restrict phi_slope and z0
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

        # filter edges according to selection criteria
        good_edge_mask = (
            (phi_slope.abs() < self.phi_slope_max)
            & (z0.abs() < self.z0_max)  # geometric
            & (dR < self.dR_max)
            & (uv_approach < self.uv_approach_max)
            & (~intersected_layer)
        )

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

    def correct_truth_labels(self, hits, edges, y, particle_ids):
        """Corrects for extra edges surviving the barrel intersection
        cut, i.e. for each particle counts the number of extra
        "transition edges" crossing from a barrel layer to an
        innermost endcap slayer; the sum is n_incorrect
        - [edges] = n_edges x 2
        - [y] = n_edges
        - [particle_ids] = n_edges
        """
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
            # particle_hit_ids = np.arange(len(hits))  # = particle_hits["hit_id"].values

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
                to_relabel = np.array(transition_edges)[
                    (edge_precedence < max_precedence)
                ]
                for l1, l2 in to_relabel:
                    relabel = (layers_1 == l1) & (layers_2 == l2) & relevant_indices
                    relabel_idx = np.where(relabel)[0]
                    y[relabel_idx] = 0
                    n_corrected += len(relabel_idx)

        if n_corrected > 0:
            print(f"Relabeled {n_corrected} edges crossing from barrel to endcaps.")
            print(f"Updated y has {int(np.sum(y))}/{len(y)} true edges.")
        return y, n_corrected

    def build_edges(self, hits):
        if self.pixel_only:
            layer_pairs = [
                (0, 1),
                (1, 2),
                (2, 3),  # barrel-barrel
                (0, 4),
                (1, 4),
                (2, 4),
                (3, 4),  # barrel-LEC
                (0, 11),
                (1, 11),
                (2, 11),
                (3, 11),  # barrel-REC
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),  # LEC-LEC
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),  # REC-REC
            ]
        else:
            layer_pairs = []
        groups = hits.groupby("layer")
        edges = []
        for (layer1, layer2) in layer_pairs:
            try:
                hits1 = groups.get_group(layer1)
                hits2 = groups.get_group(layer2)
            except KeyError:
                continue

            edges_layer_pair = self.select_edges(
                hits1,
                hits2,
                layer1,
                layer2,
            )
            edges.append(edges_layer_pair)
        edges = pd.concat(edges)
        edge_attr = np.stack(
            (
                edges.dr.values / self.feature_scale[0],
                edges.dphi.values / self.feature_scale[1],
                edges.dz.values / self.feature_scale[2],
                edges.dR.values,
            )
        )
        node_idx = np.arange(len(hits["r"]))
        node_idx = pd.Series(node_idx, index=node_idx)
        edge_start = node_idx.loc[edges.index_1].values
        edge_end = node_idx.loc[edges.index_2].values
        edge_index = np.stack((edge_start, edge_end))

        pid1 = hits.particle_id.loc[edges.index_1].values
        pid2 = hits.particle_id.loc[edges.index_2].values
        y = np.zeros(len(pid1))
        y[:] = (pid1 == pid2) & (pid1 > 0) & (pid2 > 0)
        y, n_corrected = self.correct_truth_labels(
            hits, edges[["index_1", "index_2"]], y, pid1
        )
        edge_pt = hits.pt.loc[edges.index_1].values
        return edge_index, edge_attr, y, edge_pt

    def to_pyg_data(self, graph, edge_index, edge_attr, y, evtid=-1, s=-1):
        x = torch.from_numpy(graph.x / self.feature_scale).float()
        edge_index = torch.tensor(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr).float()
        pt = torch.from_numpy(graph.pt).float()
        particle_id = torch.from_numpy(graph.particle_id).long()
        y = torch.from_numpy(y).float()
        reconstructable = torch.from_numpy(graph.reconstructable).long()
        sector = torch.from_numpy(graph.sector).long()
        evtid = torch.tensor([evtid]).long()  # event label
        s = torch.tensor([s]).long()  # sector label

        if not self.directed:
            row, col = edge_index[0], edge_index[1]
            edge_index = torch.stack(
                [torch.cat([row, col]), torch.cat([col, row])], dim=0
            )
            negate = torch.tensor([[-1], [-1], [-1], [1]]).float()
            edge_attr = torch.cat([edge_attr, negate * edge_attr], dim=1)
            y = torch.cat([y, y])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pt=pt,
            particle_id=particle_id,
            y=y,
            reconstructable=reconstructable,
            sector=sector,
            evtid=evtid,
            s=s,
        )
        data.edge_attr = data.edge_attr.T
        return data

    def get_n_truth_edges(self, df):
        grouped = df[["particle_id", "layer", "pt"]].groupby("particle_id")
        n_truth_edges = {0: 0, 0.1: 0, 0.5: 0, 0.9: 0, 1.0: 0}
        for pid, group in grouped:
            if pid == 0:
                continue
            layer = group.layer.values
            unique, counts = np.unique(layer, return_counts=True)
            n_segs = sum(counts[1:] * counts[:-1])
            for pt_thld in n_truth_edges.keys():
                if group.pt.values[0] > pt_thld:
                    n_truth_edges[pt_thld] += n_segs
        return n_truth_edges

    @staticmethod
    def get_event_id_sector_from_str(name: str) -> tuple[int, int]:
        """
        Returns:
            Event id, sector Id
        """
        evtid_s = name.split(".")[0][4:]
        evtid = int(evtid_s[:5])
        s = int(evtid_s.split("_s")[-1])
        return evtid, s

    def process(self, n=10**6, verbose=False):
        infiles = os.listdir(self.indir)
        self.edge_purities = []
        self.edge_efficiencies = collections.defaultdict(list)
        for f in infiles[:, n]:
            name = f.split("/")[-1]
            if f in self.outfiles and not self.redo:
                graph = torch.load(join(self.outdir, name))
                self.data_list.append(graph)
            else:
                try:
                    evtid, s = self.get_event_id_sector_from_str(name)
                except (ValueError, KeyError) as e:
                    raise ValueError(f"{name} is not a valid file name") from e
                if verbose:
                    print(f"Processing {f}")
                f = join(self.indir, f)
                graph = torch.load(f)
                df = self.get_dataframe(graph, evtid)
                edge_index, edge_attr, y, edge_pt = self.build_edges(df)

                if self.measurement_mode:
                    n_truth_edges = self.get_n_truth_edges(df)
                    edge_purity = sum(y) / len(y)
                    self.edge_purities.append(edge_purity)
                    for pt, denominator in n_truth_edges.items():
                        numerator = sum(y[edge_pt > pt])
                        self.edge_efficiencies[pt].append(numerator / denominator)

                graph = self.to_pyg_data(
                    graph, edge_index, edge_attr, y, evtid=evtid, s=s
                )
                outfile = join(self.outdir, name)
                if verbose:
                    print(f"Writing {outfile}")
                torch.save(graph, outfile)
                self.data_list.append(graph)

        print("Summary Statistics:")
        print(
            f" - Edge Purity: {np.mean(self.edge_purities)} "
            + f"+/- {np.std(self.edge_purities)}"
        )
        for pt, eff in self.edge_efficiencies.items():
            print(
                f" - Edge Efficiency (pt > {pt} GeV): "
                + f"{np.mean(self.edge_efficiencies[pt])} +/- "
                + f"{np.std(self.edge_efficiencies[pt])}"
            )
