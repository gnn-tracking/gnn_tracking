import os
import sys
import logging
from os.path import join

import yaml
import numpy as np


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


def get_trackml_prefixes(
    indir, trackml=True, evtid_min=0, evtid_max=8000, n_tasks=1, task=0, codalab=True
):
    all_files = os.listdir(indir)
    suffix = "-hits.csv.gz"
    if not codalab:
        suffix = "-hits.csv"
    file_prefixes = sorted(
        join(indir, f.replace(suffix, "")) for f in all_files if f.endswith(suffix)
    )
    evtids = [int(prefix[-9:]) for prefix in file_prefixes]
    if evtid_min < np.min(evtids):
        evtid_min = np.min(evtids)
    if evtid_max > np.max(evtids):
        evtid_max = np.max(evtids)
    file_prefixes = [
        prefix
        for prefix in file_prefixes
        if (
            (int(prefix.split("0000")[-1]) >= evtid_min)
            and (int(prefix.split("0000")[-1]) <= evtid_max)
        )
    ]
    file_prefixes = np.array_split(file_prefixes, n_tasks)[task]
    return file_prefixes


def load_module_map(path):
    module_map = np.load(path)
    module_map = {key: item.astype(bool) for key, item in module_maps.items()}
    return module_map


def filter_file_prefixes(file_prefixes, outdir):
    existing_files = os.listdir(outdir)
    existing_evtids = [f.split("_")[0].split("00000")[-1] for f in existing_files]
    existing_evtids = np.unique(existing_evtids)
    logging.info(f"Requested {len(file_prefixes)} new graphs.")
    file_prefixes = [
        f for f in file_prefixes if f.split("00000")[-1] not in existing_evtids
    ]
    logging.info(
        "Skipping pre-existing graphs, calculating "
        + f"{len(file_prefixes)} new graphs"
    )
    return file_prefixes


def relabel_pids(hits, particles):
    particles = particles[particles.particle_id.isin(pd.unique(hits.particle_id))]
    particle_id_map = {p: i + 1 for i, p in enumerate(particles["particle_id"].values)}
    particle_id_map[0] = 0
    particles = particles.assign(
        particle_id=particles["particle_id"].map(particle_id_map)
    )
    hits = hits.assign(particle_id=hits["particle_id"].map(particle_id_map))
    return hits, particles
