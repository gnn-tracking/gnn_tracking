from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml
from torch_geometric.data import Data

def open_yaml(infile, task=0):
    with open(infile) as f:
        config = yaml.load(f, yaml.FullLoader)
    if task == 0:
        logging.info(f"Configuration: {config}")
    return config


def map_pt(pt, to_str=True):
    num_to_str = {
        0: "0p0",
        0.1: "0p1",
        0.2: "0p2",
        0.3: "0p3",
        0.4: "0p4",
        0.5: "0p5",
        0.6: "0p6",
        0.7: "0p7",
        0.8: "0p8",
        0.9: "0p9",
        1: "1",
        1.1: "1p1",
        1.2: "1p2",
        1.3: "1p3",
        1.4: "1p4",
        1.5: "1p5",
        1.6: "1p6",
        1.7: "1p7",
        1.8: "1p8",
        1.9: "1p9",
        2.0: "2",
    }
    if to_str:
        return num_to_str[pt]
    str_to_num = {v: k for k, v in num_to_str.items()}
    return str_to_num[pt]


def relabel_pids(hits, particles):
    particles = particles[particles.particle_id.isin(pd.unique(hits.particle_id))]
    particle_id_map = {p: i + 1 for i, p in enumerate(particles["particle_id"].values)}
    particle_id_map[0] = 0
    particles = particles.assign(
        particle_id=particles["particle_id"].map(particle_id_map)
    )
    hits = hits.assign(particle_id=hits["particle_id"].map(particle_id_map))
    return hits, particles


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def append_features(hits, particles, truth):
    particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
    particles["eta_pt"] = calc_eta(particles.pt, particles.pz)
    truth = truth[["hit_id", "particle_id"]].merge(
        particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]], on="particle_id"
    )
    hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
    hits["phi"] = np.arctan2(hits.y, hits.x)
    hits["eta_rz"] = calc_eta(hits.r, hits.z)
    hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits = hits[
        ["hit_id", "r", "phi", "eta_rz", "x", "y", "z", "u", "v", "volume_id"]
    ].merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")
    data = Data(
        x=hits[["x", "y", "z", "r", "phi", "eta_rz", "u", "v"]].values,
        particle_id=hits["particle_id"].values,
        pt=hits["pt"].values,
    )
    return data
