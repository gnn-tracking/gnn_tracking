from __future__ import annotations

import collections
import logging
import os
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from trackml.dataset import load_event

from gnn_tracking.utils.log import get_logger

pd.options.mode.chained_assignment = None  # default='warn'


class PointCloudBuilder:
    def __init__(
        self,
        outdir: str | PurePath,
        indir: str | PurePath,
        *,
        n_sectors: int,
        redo=True,
        pixel_only=True,
        sector_di=0.0001,
        sector_ds=1.1,
        measurement_mode=False,
        thld=0.5,
        remove_noise=False,
        write_output=True,
        log_level=logging.INFO,
        collect_data=True,
    ):
        """

        Args:
            outdir: Directory for the output files
            indir: Directory of input files
            n_sectors: Total number of sectors
            redo: Force recompute of point cloud building
            pixel_only: Construct tracks only from pixel layers
            sector_di:
            sector_ds:
            feature_names: Names of features that are passed on to pyg data
            feature_scale: Scaling of features given by ``feature_names``
            measurement_mode:
            thld: pt threshold for reconstructable particles
            remove_noise:
            write_output:
            log_level:
            collect_data: Collect data in memory
        """
        # create outdir if necessary
        os.makedirs(outdir, exist_ok=True)
        self.outdir = PurePath(outdir)

        self.indir = PurePath(indir)
        self.n_sectors = n_sectors
        self.redo = redo
        self.pixel_only = pixel_only
        self.sector_di = sector_di
        self.sector_ds = sector_ds
        self.feature_names = ["r", "phi", "z", "eta_rz", "u", "v"]
        self.feature_scale = np.array([1, 1, 1, 1, 1, 1])
        self.measurement_mode = measurement_mode
        self.thld = thld
        self.stats = {}
        self.remove_noise = remove_noise
        self.measurements: list[dict[str, Any]] = []
        self.write_output = write_output

        suffix = "-hits.csv.gz"
        self.prefixes: list[str] = []
        #: Does an output file for a given key exist?
        self.exists: dict[str, bool] = {}
        outfiles = os.listdir(outdir)
        for p in os.listdir(self.indir):
            if str(p).endswith(suffix):
                prefix = str(p).replace(suffix, "")
                evtid = int(prefix[-9:])
                for s in range(self.n_sectors):
                    key = f"data{evtid}_s{s}.pt"
                    self.exists[key] = key in outfiles
                self.prefixes.append(self.indir / prefix)

        self.data_list: list[Data] = []
        self.logger = get_logger("PointCloudBuilder", level=log_level)
        self._collect_data = collect_data

    def calc_eta(self, r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def restrict_to_subdetectors(self, hits: pd.DataFrame) -> pd.DataFrame:
        # select barrel layers and assign convenient layer number [0-9]
        if self.pixel_only:
            pixel_barrel = [(8, 2), (8, 4), (8, 6), (8, 8)]
            pixel_LEC = [(7, 14), (7, 12), (7, 10), (7, 8), (7, 6), (7, 4), (7, 2)]
            pixel_REC = [(9, 2), (9, 4), (9, 6), (9, 8), (9, 10), (9, 12), (9, 14)]
            allowed_layers = pixel_barrel + pixel_REC + pixel_LEC
        else:
            allowed_layers = None

        hit_layer_groups = hits.groupby(["volume_id", "layer_id"])
        if allowed_layers is not None:
            available_allowed_layers = sorted(
                set(hit_layer_groups.groups.keys()) & set(allowed_layers)
            )
        else:
            available_allowed_layers = sorted(hit_layer_groups.groups.keys())
        hits = pd.concat(
            [
                hit_layer_groups.get_group(layer).assign(layer=i)
                for i, layer in enumerate(available_allowed_layers)
            ]
        )
        return hits

    def append_features(
        self, hits: pd.DataFrame, particles: pd.DataFrame, truth: pd.DataFrame
    ) -> pd.DataFrame:
        """Add additional features to the hits dataframe and return it."""
        particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
        particles["eta_pt"] = self.calc_eta(particles.pt, particles.pz)

        # handle noise
        truth_noise = truth[["hit_id", "particle_id"]][truth.particle_id == 0]
        truth_noise["pt"] = 0
        truth = truth[["hit_id", "particle_id"]].merge(
            particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]],
            on="particle_id",
        )

        # optionally add noise
        if not self.remove_noise:
            truth = pd.concat([truth, truth_noise])

        hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
        hits["phi"] = np.arctan2(hits.y, hits.x)
        hits["eta_rz"] = self.calc_eta(hits.r, hits.z)
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits = hits[
            [
                "hit_id",
                "r",
                "phi",
                "eta_rz",
                "x",
                "y",
                "z",
                "u",
                "v",
                "volume_id",
                "layer",
            ]
        ].merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")
        return hits

    def sector_hits(
        self, hits: pd.DataFrame, s, particle_id_counts: dict[int, int]
    ) -> pd.DataFrame:
        if self.n_sectors == 1:
            hits["sector"] = 0
            return hits
        # build sectors in each 2*np.pi/self.n_sectors window
        theta = np.pi / self.n_sectors
        slope = np.arctan(theta)
        hits["ur"] = hits["u"] * np.cos(2 * s * theta) - hits["v"] * np.sin(
            2 * s * theta
        )
        hits["vr"] = hits["u"] * np.sin(2 * s * theta) + hits["v"] * np.cos(
            2 * s * theta
        )

        sector = hits[
            ((hits.vr > -slope * hits.ur) & (hits.vr < slope * hits.ur) & (hits.ur > 0))
        ]

        # assign when the majority of the particle's hits are in a sector
        particle_id_sectors = collections.defaultdict(lambda: -1)
        for pid in np.unique(sector.particle_id.values):
            if pid == 0:
                continue
            hits_in_sector = len(sector[sector.particle_id == pid])
            hits_for_pid = particle_id_counts[pid]
            if (hits_in_sector / hits_for_pid) > 0.5:
                particle_id_sectors[pid] = s

        lower_bound = -self.sector_ds * slope * hits.ur - self.sector_di
        upper_bound = self.sector_ds * slope * hits.ur + self.sector_di
        extended_sector = hits[
            ((hits.vr > lower_bound) & (hits.vr < upper_bound) & (hits.ur > 0))
        ]
        extended_sector["sector"] = extended_sector["particle_id"].map(
            particle_id_sectors
        )

        measurements = {}
        if self.measurement_mode:
            measurements["n_hits"] = len(sector)
            measurements["n_hits_ext"] = len(extended_sector)
            if len(sector) > 0:
                measurements["n_hits_ratio"] = len(extended_sector) / len(sector)
            else:
                measurements["n_hits_ratio"] = 0

            measurements["n_unique_pids"] = len(
                np.unique(extended_sector.particle_id.values)
            )

            majority_contained = []
            for pid in np.unique(extended_sector.particle_id.values):
                if pid == 0:
                    continue
                group = hits[hits.particle_id == pid]
                in_sector = (
                    (group.vr < slope * group.ur)
                    & (group.vr > -slope * group.ur)
                    & (group.pt >= self.thld)
                )
                n_total = particle_id_counts[pid]
                if sum(in_sector) / n_total < 0.5:
                    continue
                in_ext_sector = (
                    (group.vr < (self.sector_ds * slope * group.ur + self.sector_di))
                    & (group.vr > (-self.sector_ds * slope * group.ur - self.sector_di))
                    & (group.pt > self.thld)
                )
                majority_contained.append(sum(in_ext_sector) == n_total)
                efficiency = sum(majority_contained) / len(majority_contained)
                measurements["majority_contained"] = efficiency
                self.measurements.append(measurements)

        return extended_sector

    def to_pyg_data(self, hits: pd.DataFrame) -> Data:
        """Build the output data structure"""
        data = Data(
            x=torch.from_numpy(
                hits[self.feature_names].values / self.feature_scale
            ).float(),
            layer=torch.from_numpy(hits.layer.values).long(),
            particle_id=torch.from_numpy(hits["particle_id"].values).long(),
            pt=torch.from_numpy(hits["pt"].values).float(),
            reconstructable=torch.from_numpy(hits["reconstructable"].values).long(),
            sector=torch.from_numpy(hits["sector"].values).long(),
        )
        return data

    def get_measurements(self):
        measurements = pd.DataFrame(self.measurements)
        means = measurements.mean()
        stds = measurements.std()
        output = {}
        for var in means.index:
            output[var] = means[var]
            output[var + "_err"] = stds[var]
        return output

    def process(self, start: int | None = None, stop: int | None = None):
        """Process input files from self.input_files and write output files to
        self.output_files

        Args:
            n: Number of events to process
            verbose:

        Returns:

        """
        if stop is not None and stop > len(self.prefixes):
            # to avoid tracking wrong hyperparameters
            raise ValueError(
                f"stop={stop} is larger than the number of files ({len(self.prefixes)})"
            )
        for f in self.prefixes[start:stop]:
            self.logger.debug(f"Processing {f}")

            evtid = int(f[-9:])
            hits, particles, truth = load_event(f, parts=["hits", "particles", "truth"])

            hits = self.restrict_to_subdetectors(hits)
            hits = self.append_features(hits, particles, truth)
            hits_by_pid = hits.groupby("particle_id")

            particle_id_counts = {pid: len(hit_group) for pid, hit_group in hits_by_pid}
            pid_layers_hit = {
                pid: len(np.unique(hit_group.layer)) for pid, hit_group in hits_by_pid
            }
            self.reconstructable = {
                pid: ((counts >= 3) and (pid > 0))
                for pid, counts in pid_layers_hit.items()
            }
            hits["reconstructable"] = hits.particle_id.map(self.reconstructable)

            n_particles = len(np.unique(hits.particle_id.values))
            n_hits = len(hits)
            n_noise = len(hits[hits.particle_id == 0])
            n_sector_hits = 0  # total quantities appearing in sectored graph
            n_sector_particles = 0
            for s in range(self.n_sectors):
                name = f"data{evtid}_s{s}.pt"
                if self.exists[name] and not self.redo:
                    if self._collect_data:
                        data = torch.load(self.outdir / name)
                        self.data_list.append(data)
                    self.logger.debug(f"skipping {name}")
                    continue
                else:
                    sector = self.sector_hits(
                        hits, s, particle_id_counts=particle_id_counts
                    )
                    n_sector_hits += len(sector)
                    n_sector_particles += len(np.unique(sector.particle_id.values))
                    sector = self.to_pyg_data(sector)
                    outfile = self.outdir / name
                    if self.write_output:
                        torch.save(sector, outfile)
                    self.data_list.append(sector)
                    self.logger.debug(f"wrote {outfile}")

            self.stats[evtid] = {
                "n_hits": n_hits,
                "n_particles": n_particles,
                "n_noise": n_noise,
                "n_sector_hits": n_sector_hits,
                "n_sector_particles": n_sector_particles,
            }

        self.logger.debug("Output statistics:", self.stats[evtid])
        if self.measurement_mode:
            measurements = pd.DataFrame(self.measurements)
            means = measurements.mean()
            stds = measurements.std()
            for var in stds.index:
                self.logger.debug(f"{var}: {means[var]:.4f}+/-{stds[var]:.4f}")
