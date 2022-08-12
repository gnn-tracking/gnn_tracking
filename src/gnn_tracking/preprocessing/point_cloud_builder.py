import os
from os.path import join

import numpy as np
import torch
from torch_geometric.data import Data
from trackml.dataset import load_event


class GraphSectors:
    def __init__(
            self,
            n_sectors,
            ds=None,
            di=None
    ):

class PointCloudBuilder:
    def __init__(
        self,
        outdir: str,
        indir: str,
        n_sectors: int,
        redo=False,
    ):
        self.outdir = outdir
        self.indir = indir
        self.n_sectors = n_sectors
        self.redo = redo
        self.idx_dict = {}
        counter = 0
        for i in range(1000):
            for j in range(self.n_sectors):
                self.idx_dict[counter] = (i, j)

        suffix = "-hits.csv.gz"
        self.prefixes, self.exists = [], {}
        for p in os.listdir(self.indir):
            if str(p).endswith(suffix):
                prefix = str(p).replace(suffix, "")
                evtid = int(prefix[-9:])
                if f"data{evtid}_s0.pt" in os.listdir(outdir):
                    self.exists[evtid] = True
                else:
                    self.exists[evtid] = False
                self.prefixes.append(join(indir, prefix))

        self.data_list = []
        self.process()

    def calc_eta(self, r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def append_features(self, hits, particles, truth):
        particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
        particles["eta_pt"] = self.calc_eta(particles.pt, particles.pz)
        truth = truth[["hit_id", "particle_id"]].merge(
            particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]],
            on="particle_id",
        )
        hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
        hits["phi"] = np.arctan2(hits.y, hits.x)
        hits["eta_rz"] = self.calc_eta(hits.r, hits.z)
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits = hits[
            ["hit_id", "r", "phi", "eta_rz", "x", "y", "z", "u", "v", "volume_id"]
        ].merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")
        feature_names = ["r", "phi", "z", "eta_rz", "u", "v"]
        feature_scale = np.array([1000.0, np.pi, 1000.0, 1, 1 / 1000.0, 1 / 1000.0])
        data = Data(
            x=hits[feature_names].values / feature_scale,
            particle_id=hits["particle_id"].values,
            pt=hits["pt"].values,
        )
        return data

    def process(self):
        for i, f in enumerate(self.prefixes):
            s = 0
            evtid = int(f[-9:])
            name = f"data{evtid}_s{s}.pt"
            if self.exists[evtid] and not self.redo:
                data = torch.load(join(self.outdir, name))
                self.data_list.append(data)
            else:
                hits, particles, truth = load_event(
                    f, parts=["hits", "particles", "truth"]
                )
                data = self.append_features(hits, particles, truth)
                torch.save(data, join(self.outdir, name))
                self.data_list.append(data)
