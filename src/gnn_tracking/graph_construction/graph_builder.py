import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import ndarray as A
from pandas import DataFrame as DF
from torch_geometric.data import Data
from tqdm import tqdm

from gnn_tracking.preprocessing.point_cloud_builder import DEFAULT_FEATURES
from gnn_tracking.utils.log import get_logger, logger

# ruff: noqa: G004


def get_two_hop_tuples(tuples: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Given a list of tuples `(a, b)`, returns the set of tuples `(x, y)`
    where `(x, t)` and `(t, y)` are in the input list.
    """
    additions = set()
    for a, b in tuples:
        for c, d in tuples:
            if b != c:
                continue
            additions.add((a, d))
    return additions


# todo: This class needs refactoring: Loading should be done separately; many of the
#   methods are actually static and might better be extracted; some of the __init__
#   arguments are better for the process method; internal methods should be marked
#   as private; pathlib should be used instead of os.path/string manipulations;
#   typing is incomplete; log level should just be that of the logging module
class GraphBuilder:
    def __init__(
        self,
        indir: str | os.PathLike,
        outdir: str | os.PathLike,
        *,
        pixel_only=True,
        redo=True,
        phi_slope_max=0.005,
        z0_max=200,
        dR_max=1.7,
        remove_intersecting=True,
        directed=False,
        measurement_mode=False,
        write_output=True,
        log_level=0,
        collect_data=True,
        edge_augmentation: str | None = None,
    ):
        """Build graphs out of the input data.

        Args:
            indir:
            outdir:
            pixel_only: Only consider pixel detector
            redo:
            phi_slope_max:
            z0_max:
            dR_max:
            remove_intersecting: Remove "ambiguous" edges, see Fig. 3 in
                "Charged particle tracking via edge-classifying interaction networks"
                http://arxiv.org/abs/2103.16701 and mark the remaining ones as
                incorrect edges
            directed: Build directed edges
            measurement_mode:
            write_output: Save graphs?
            log_level:
            collect_data: Deprecated: Directly load the data into memory
            edge_augmentation: Add more edges (e.g., adding next-neighbor connections).
                Needs ``remove_intersecting`` to be false
        """
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.pixel_only = pixel_only
        self.redo = redo
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.dR_max = dR_max
        #: Name/meaning of the node features
        self.feature_names = DEFAULT_FEATURES
        #: Scaling of node features
        self.feature_scale = np.array(
            [1000.0, np.pi, 1000.0, 1, 1 / 1000.0, 1 / 1000.0]
            + [1.0] * (len(DEFAULT_FEATURES) - 6)
        )
        self._data_list = []
        self.directed = directed
        self.measurement_mode = measurement_mode
        self.write_output = write_output
        self.measurements = []
        level = logging.DEBUG if log_level > 0 else logging.INFO
        self.logger = get_logger("GraphBuilder", level)
        self._collect_data = collect_data
        if self._collect_data:
            self.logger.warning(
                "Collecting data is deprecated. Please use graph_builder.load_data "
                "instead."
            )
        self._remove_intersecting = remove_intersecting
        self._edge_augmentation = edge_augmentation
        if edge_augmentation and remove_intersecting:
            _ = "Edge autmentation currently requires remove_intersecting==False"
            raise ValueError(_)

    @property
    def data_list(self):
        logger.warning(
            "Using GraphBuilder to load data is depcreacted. Please use "
            "graph_builder.load_data instead."
        )
        return self._data_list

    def get_measurements(self):
        measurements = pd.DataFrame(self.measurements)
        means = measurements.mean()
        stds = measurements.std()
        output = {}
        for var in means.index:
            output[var] = means[var]
            output[var + "_err"] = stds[var]
        return output

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

    def get_dataframe(self, evt: Data, evtid: int) -> DF:
        """Converts pytorch geometric data object to pandas dataframe

        Args:
            evt: pytorch geometric data object
            evtid: event id

        Returns:
            pandas dataframe
        """
        to_df = {"evtid": evtid}
        for i, n in enumerate(self.feature_names):
            to_df[n] = evt.x[:, i]
        to_df["layer"] = evt.layer
        to_df["pt"] = evt.pt
        to_df["particle_id"] = evt.particle_id
        return pd.DataFrame(to_df)

    def select_edges(self, hits1: DF, hits2: DF, layer1: int, layer2: int) -> DF:
        """Build edges between two layers

        Args:
            hits1: Information about hit 1
            hits2: Information about hit 2
            layer1: Layer number for hit 1
            layer2: Layer number for hit 2

        Returns:
            Dictionary containing edge indices and extra information
        """
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

        # restrict phi_slope and z0
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

        # apply the intersecting line cut
        if (layer1 == 7) and (layer2 in [6, 11]):
            # 0th barrel layer to left EC or right EC
            # We calculate the z coordinate that the edge passes through when it is on
            # the r height of layer 1. We check if it is between the z range of layer
            # 1 to see if the edge intersects layer 1
            layer_1_r = 71.56298065185547
            z_coord = layer_1_r * dz / dr + z0
            intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)
        elif (layer1 == 8) and (layer2 in [6, 11]):
            # 1st barrel layer to the left EC or right EC
            layer_2_r = 115.37811279296875
            z_coord = layer_2_r * dz / dr + z0
            intersected_layer = np.logical_and(z_coord > -490.975, z_coord < 490.975)
        else:
            intersected_layer = dr.abs() < -1

        # filter edges according to selection criteria
        good_edge_mask = (
            (phi_slope.abs() < self.phi_slope_max)
            & (z0.abs() < self.z0_max)  # geometric
            & (dR < self.dR_max)
        )
        if self._remove_intersecting:
            good_edge_mask &= ~intersected_layer

        # store edges (in COO format) and geometric edge features
        return pd.DataFrame(
            {
                "index_1": hit_pairs["index_1"][good_edge_mask],
                "index_2": hit_pairs["index_2"][good_edge_mask],
                "dr": dr[good_edge_mask],
                "dphi": dphi[good_edge_mask],
                "dz": dz[good_edge_mask],
                "dR": dR[good_edge_mask],
            }
        )

    def correct_truth_labels(
        self, hits: DF, edges: DF, y: A, particle_ids: A
    ) -> tuple[A, int]:
        """Corrects for extra edges surviving the barrel intersection
        cut, i.e. for each particle counts the number of extra
        "transition edges" crossing from a barrel layer to an
        innermost endcap layer; the sum is n_incorrect
        - [edges] = n_edges x 2
        - [y] = n_edges
        - [particle_ids] = n_edges

        Returns:
            corrected truth labels, number of incorrect edges
        """
        # layer indices for barrel-to-endcap edges
        barrel_to_endcaps = {
            (7, 6),
            (8, 6),
            (9, 6),
            (10, 6),  # barrel to l-EC
            (7, 11),
            (8, 11),
            (9, 11),
            (10, 11),
        }  # barrel to r-EC
        precedence = {
            (7, 6): 0,
            (8, 6): 1,
            (9, 6): 2,
            (10, 6): 3,
            (7, 11): 0,
            (8, 11): 1,
            (9, 11): 2,
            (10, 11): 3,
        }

        # group hits by particle id, get layer indices
        hits_by_particle = hits.groupby("particle_id")
        layers_1 = hits.layer.loc[edges.index_1].to_numpy()
        layers_2 = hits.layer.loc[edges.index_2].to_numpy()

        # loop over particle_id, particle_hits,
        # count extra transition edges as n_incorrect
        n_corrected = 0
        for particle_id, _ in hits_by_particle:
            if particle_id == 0:
                # noise
                continue

            # grab true segment indices for particle
            relevant_indices = (particle_ids == particle_id) & (y == 1)

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
            self.logger.debug(
                "Relabeled %d edges crossing from barrel to endcaps.", n_corrected
            )
            self.logger.debug("Updated y has %d/%d true edges.", int(np.sum(y)), len(y))
        return y, n_corrected

    def build_edges(self, hits: DF) -> tuple[A, A, A, A]:
        """Build edges between hits

        Args:
            hits: Point cloud dataframe

        Returns:
            edge_index (2 x num edges), edge_attr (edge features x num edges),
            y (truth label, shape = num edges), edge_pt (pt of track belong to first
            hit)
        """
        if self.pixel_only:
            layer_pairs = [
                (7, 8),
                (8, 9),
                (9, 10),  # barrel-barrel
                (7, 6),
                (8, 6),
                (9, 6),
                (10, 6),  # barrel-LEC
                (7, 11),
                (8, 11),
                (9, 11),
                (10, 11),  # barrel-REC
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),  # LEC-LEC
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),  # REC-REC
            ]
        else:
            layer_pairs = []
        if self._edge_augmentation is None:
            pass
        elif self._edge_augmentation == "add_two_hop":
            layer_pairs.extend(list(get_two_hop_tuples(layer_pairs)))
        else:
            _ = f"Invalid augmentation mode: {self._edge_augmentation}"
            raise ValueError(_)
        groups = hits.groupby("layer")
        edges = []
        for layer1, layer2 in layer_pairs:
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
                edges.dr.to_numpy() / self.feature_scale[0],
                edges.dphi.to_numpy() / self.feature_scale[1],
                edges.dz.to_numpy() / self.feature_scale[2],
                edges.dR.to_numpy(),
            )
        )
        node_idx = np.arange(len(hits["r"]))
        node_idx = pd.Series(node_idx, index=node_idx)
        edge_start = node_idx.loc[edges.index_1].to_numpy()
        edge_end = node_idx.loc[edges.index_2].to_numpy()
        edge_index = np.stack((edge_start, edge_end))

        pid1 = hits.particle_id.loc[edges.index_1].to_numpy()
        pid2 = hits.particle_id.loc[edges.index_2].to_numpy()
        y = np.zeros(len(pid1))
        y[:] = (pid1 == pid2) & (pid1 > 0)
        if self._remove_intersecting:
            y, _ = self.correct_truth_labels(
                hits, edges[["index_1", "index_2"]], y, pid1
            )
        edge_pt = hits.pt.loc[edges.index_1].to_numpy()
        return edge_index, edge_attr, y, edge_pt

    def to_pyg_data(
        self,
        point_cloud,
        edge_index: A,
        edge_attr: A,
        y: A,
        evtid: int = -1,
        s: int = -1,
    ) -> Data:
        """Convert hit dataframe and edges to pytorch geometric data object

        Args:
            point_cloud: Hit dataframe, see `get_dataframe`
            edge_index: See `build_edges`
            edge_attr: See `build_edges`
            y: See `build_edges`
            evtid: Event ID
            s: Sector

        Returns:
            Pytorch geometric data object
        """
        x = (point_cloud.x.clone() / self.feature_scale).float()
        edge_index = torch.tensor(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr).float()
        pt = point_cloud.pt.clone().float()
        particle_id = point_cloud.particle_id.clone().long()
        y = torch.tensor(y).float()
        reconstructable = point_cloud.reconstructable.clone().long()
        sector = point_cloud.sector.clone().long()
        evtid = torch.tensor([evtid]).long()  # event label
        s = torch.tensor([s]).long()  # sector label
        eta = point_cloud.eta.clone().float()
        layer = point_cloud.layer.clone().long()

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
            eta=eta,
            layer=layer,
        )
        data.edge_attr = data.edge_attr.T
        return data

    def get_n_truth_edges(self, df: DF) -> dict[float, int]:
        grouped = df[["particle_id", "layer", "pt"]].groupby("particle_id")
        n_truth_edges = {0: 0, 0.1: 0, 0.5: 0, 0.9: 0, 1.0: 0}
        for pid, group in grouped:
            if pid == 0:
                continue
            layer = group.layer.to_numpy()
            unique, counts = np.unique(layer, return_counts=True)
            n_segs = sum(counts[1:] * counts[:-1])
            for pt_thld in n_truth_edges:
                if group.pt.to_numpy()[0] > pt_thld:
                    n_truth_edges[pt_thld] += n_segs
        return n_truth_edges

    @staticmethod
    def get_event_id_sector_from_str(name: str) -> tuple[int, int]:
        """Parses input file names.

        Args:
            name: Input file name

        Returns:
            Event id, sector Id
        """
        number_s = name.split(".")[0][len("data") :]
        evtid_s, sectorid_s = number_s.split("_s")
        evtid = int(evtid_s)
        sectorid = int(sectorid_s)
        return evtid, sectorid

    def process(self, start=0, stop=1, *, only_sector: int = -1, progressbar=False):
        """Main processing loop

        Args:
            start:
            stop:
            only_sector: Only process files for this sector. If < 0 (default): process
                all sectors.

        Returns:

        """
        available_files = sorted(self.indir.iterdir())
        outfiles = sorted([child.name for child in self.outdir.iterdir()])
        considered_files = available_files[start:stop]
        logger.info(
            "Processing %d graphs (out of %d available).",
            len(considered_files),
            len(available_files),
        )
        iterator = considered_files
        if progressbar:
            iterator = tqdm(iterator)
        for f in iterator:
            if f.suffix != ".pt":
                continue
            try:
                evtid, sector = self.get_event_id_sector_from_str(f.name)
            except (ValueError, KeyError) as e:
                msg = f"{f.name} is not a valid file name"
                raise ValueError(msg) from e
            if 0 <= only_sector != sector:
                continue
            if f.name in outfiles and not self.redo:
                if self._collect_data:
                    # Deprecated, remove soon
                    graph = torch.load(self.outdir / f.name)
                    self._data_list.append(graph)
                continue
            self.logger.debug(f"Processing {f.name}")
            point_cloud = torch.load(f)
            df = self.get_dataframe(point_cloud, evtid)
            edge_index, edge_attr, y, edge_pt = self.build_edges(df)

            if self.measurement_mode:
                n_truth_edges = self.get_n_truth_edges(df)
                edge_purity = sum(y) / len(y)
                edge_efficiencies = {}
                for pt, denominator in n_truth_edges.items():
                    numerator = sum(y[edge_pt > pt])
                    edge_efficiencies[f"edge_efficiency_{pt}"] = numerator / denominator
                n_truth_edges = {
                    f"n_truth_edge_{pt}": n for pt, n in n_truth_edges.items()
                }
                measurements = {
                    "n_edges": len(y),
                    "n_true_edges": sum(y),
                    "n_false_edges": len(y) - sum(y),
                    **n_truth_edges,
                    "edge_purity": edge_purity,
                    **edge_efficiencies,
                }
                self.measurements.append(measurements)

            graph = self.to_pyg_data(
                point_cloud, edge_index, edge_attr, y, evtid=evtid, s=sector
            )
            outfile = self.outdir / f.name
            self.logger.debug(f"Writing {outfile}")
            if self.write_output:
                torch.save(graph, outfile)
            if self._collect_data:
                self._data_list.append(graph)

        if self.measurement_mode:
            self.logger.info(self.get_measurements())
