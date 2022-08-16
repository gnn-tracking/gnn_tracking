from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from collections import Counter, OrderedDict
from functools import partial
from os.path import join

import numpy as np
import pandas as pd
import trackml.dataset
from matplotlib import pyplot as plt
from scipy import optimize
from utils.graph_construction import initialize_logger, select_hits


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("prepare.py")
    add_arg = parser.add_argument
    add_arg("-v", "--verbose", action="store_true")
    add_arg("-i", "--input-dir", type=str, default="/tigress/jdezoort/codalab/train_1")
    add_arg("-o", "--output-dir", type=str, default="particle_properties")
    add_arg("--n-workers", type=int, default=1)
    add_arg("--redo", type=bool, default=False)
    add_arg("--n-tasks", type=int, default=1)
    add_arg("--task", type=int, default=0)
    return parser.parse_args(args)


def calc_radii(xc: float, yc: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def radii_diffs(c: tuple[float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radii = calc_radii(*c, x=x, y=y)
    return radii - np.mean(radii)


def calc_circle_pt(R: float) -> float:
    # todo: Name magic constant
    return 0.0003 * 2 * R


def calc_circle_d0(xc: float, yc: float, R: float) -> float:
    return R - np.sqrt(xc**2 + yc**2)


def radius_error(x: np.ndarray, y: np.ndarray, xc: float, yc: float, R: float) -> float:
    angles = np.arctan2(y - yc, x - xc)
    cx = xc + R * np.cos(angles)
    cy = yc + R * np.sin(angles)
    sum2_errs = (x - cx) ** 2 + (y - cy) ** 2
    return np.sqrt(np.sum(sum2_errs))


def rotate(
    u: np.ndarray, v: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot_mat = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    uv = np.stack((u, v), axis=1).T
    uv = rot_mat @ uv
    ur, vr = uv[0], uv[1]
    return ur, vr


def regress(x, y, quadratic=False):
    X = np.stack((np.ones(len(x)), x), axis=1)
    if quadratic:
        X = np.stack((np.ones(len(x)), x, x**2), axis=1)
    sol = np.linalg.inv(X.T @ X) @ X.T @ y
    sol = sol.reshape(len(sol))
    return sol


def calc_conformal_pt(fit):
    b = 1.0 / (2 * fit[0])
    a = -fit[1] * b
    R = np.sqrt(a**2 + b**2)
    return 0.0003 * 2 * R


def calc_conformal_d0(fit):
    b = 1.0 / (2 * fit[0])
    a = -fit[1] * b
    return -fit[2] * (b / np.sqrt(a**2 + b**2)) ** 3


def track_fit_plot(x, y, u, v, conformal_fit, xc, yc, R, label=""):
    fig, axs = plt.subplots(ncols=2, dpi=100, figsize=(16, 8))
    axs[0].scatter(x, y, label=label, marker=".", s=40)
    phi = np.linspace(0, 2 * np.pi, 1000)
    axs[0].plot(R * np.cos(phi) + xc, R * np.sin(phi) + yc, lw=0.5, ls="-", marker="")
    axs[1].scatter(u, v, label=label, marker=".", s=40)
    s = np.linspace(-0.06, 0.06, 1000)
    axs[1].plot(
        s,
        conformal_fit[2] * s**2 + conformal_fit[1] * s + conformal_fit[0],
        lw=0.5,
        ls="-",
        marker="",
    )
    plt.legend(loc="best")
    axs[1].set_xlabel("$x$ [m]")
    axs[0].set_ylabel("$y$ [m]")
    axs[0].set_xlim([-150, 150])
    axs[0].set_ylim([-150, 150])
    axs[0].plot(0, 0, ms=20, marker="+", color="black")
    axs[1].set_xlim([-0.06, 0.06])
    axs[1].set_ylim([-0.06, 0.06])
    axs[1].set_xlabel("$u$ [1/mm]")
    axs[1].set_ylabel("$v$ [1/mm]")
    plt.tight_layout()
    plt.show()


def make_df(
    prefix,
    output_dir,
    endcaps=True,
    remove_noise=False,
    remove_duplicates=False,
    n_layers_fit=4,
):

    # define valid layer-layer connections
    layer_pairs = [(0, 1), (1, 2), (2, 3)]  # barrel-barrel
    layer_pairs.extend(
        [
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
            (6, 7),  # LEC-LEC
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),  # REC-REC
            (14, 15),
            (15, 16),
            (16, 17),
        ]
    )
    valid_connections = set(layer_pairs)

    # load the data
    evtid = int(prefix[-9:])
    logging.info("Event %i, loading data" % evtid)
    hits, particles, truth = trackml.dataset.load_event(
        prefix, parts=["hits", "particles", "truth"]
    )
    hits = hits.assign(evtid=evtid)

    # apply hit selection
    logging.info("Event %i, selecting hits" % evtid)
    hits, particles = select_hits(
        hits, truth, particles, 0, endcaps, remove_noise, remove_duplicates
    )

    # get truth information for each particle
    hits_by_particle = hits.groupby("particle_id")
    df_properties = []
    for i, (particle_id, particle_hits) in enumerate(hits_by_particle):
        properties = pd.DataFrame(
            {
                "particle_id": particle_id,
                "pt": 0,
                "eta_pt": 0,
                "d0": 0,
                "q": 0,
                "n_track_segs": 0,
                "n_layers_hit": 0,
                "n_hits": len(particle_hits),
                "reconstructable": True,
                "skips_layer": False,
                "good_fit": False,
                "anomalous": False,
                "pt_err": 0,
            },
            index=[i],
        )

        # explicit noise case
        if particle_id == 0:
            # properties['reconstructable'] = False
            # df_properties.append(properties)
            continue

        # fill in properties of real particles
        properties["pt"] = particle_hits["pt"].values[0]
        properties["eta_pt"] = particle_hits["eta_pt"].values[0]
        properties["q"] = particle_hits["q"].values[0]
        layers_hit = particle_hits.layer.values
        hits_per_layer = Counter(layers_hit)  # dict of layer: nhits
        hits_per_layer = OrderedDict(hits_per_layer)
        unique_layers_hit = np.unique(layers_hit)
        properties["q"] = particle_hits["q"].values[0]
        properties["n_layers_hit"] = len(unique_layers_hit)

        # implicit noise (single-layer particles)
        if len(unique_layers_hit) == 1:
            properties["reconstructable"] = False
            df_properties.append(properties)
            continue

        # now particles have hit >1 layer
        paired_layers = set(zip(unique_layers_hit[:-1], unique_layers_hit[1:]))
        skips_layer = not paired_layers.issubset(valid_connections)

        # figure out how many possible track segments to capture
        good_layer_pairs = paired_layers.intersection(valid_connections)
        edge_counts = [
            hits_per_layer[lp[0]] * hits_per_layer[lp[1]] for lp in good_layer_pairs
        ]
        edge_count = np.sum(edge_counts)
        properties["n_track_segs"] = edge_count

        # particles that skipped a layer
        if skips_layer:
            properties["skips_layer"] = True
            properties["reconstructable"] = False
            df_properties.append(properties)
            continue

        # two-layer particles
        if len(unique_layers_hit) == 2:
            properties["reconstructable"] = False
            df_properties.append(properties)
            continue

        # coordinates for track fits
        true_pt = properties["pt"].values[0]
        layer_id = particle_hits["layer"].values
        sort_idx = np.argsort(layer_id)
        x = particle_hits["x"].values[sort_idx]
        y = particle_hits["y"].values[sort_idx]
        vx = particle_hits["vx"].values[0]  # force vertex into fit
        vy = particle_hits["vy"].values[0]
        x = np.insert(x, 0, vx)
        y = np.insert(y, 0, vy)
        xy2 = x**2 + y**2
        u, v = x / xy2, y / xy2

        # fit only the first n_layer_fit layers
        cutoff = (
            np.sum(
                [
                    nhits
                    for l, (lid, nhits) in enumerate(hits_per_layer.items())
                    if l < n_layers_fit
                ]
            )
            + 1
        )

        # rotate the conformal coordinates
        theta = np.arctan2(v[:cutoff][-1] - v[0], u[:cutoff][-1] - u[0])
        ur, vr = rotate(u, v, theta)

        # perform conformal fit
        fit = regress(ur[:cutoff], vr[:cutoff], quadratic=True)
        conformal_pt = calc_conformal_pt(fit)
        conformal_pt_err = abs(true_pt - conformal_pt) / true_pt
        conformal_d0 = calc_conformal_d0(fit)

        # use conformal fit to inform circle fit
        yc_est = 1 / (2 * fit[0])
        xc_est = -fit[1] * yc_est
        est = rotate([xc_est], [yc_est], -theta)
        xc_est, yc_est = est[0][0], est[1][0]
        # R_est = true_pt / (2 * 0.0003)
        (xc, yc), ier = optimize.leastsq(radii_diffs, (xc_est, yc_est), args=(x, y))
        R = np.mean(calc_radii(xc, yc, x=x[:cutoff], y=y[:cutoff]))
        circle_pt = calc_circle_pt(R)
        circle_pt_err = abs(true_pt - circle_pt) / true_pt
        circle_d0 = calc_circle_d0(xc, yc, R)
        circle_R_err = radius_error(x, y, xc, yc, R)

        # try the fit again with fewer hits if pt error is bad
        min_err = min(conformal_pt_err, circle_pt_err)
        properties["pt_err"] = min_err

        # there was no hope for these fits
        if circle_R_err > 5:
            properties["reconstructable"] = False
            properties["anomalous"] = True
            df_properties.append(properties)
            continue

        # otherwise we have a good track and maybe a good fit
        if min_err < 0.5:
            properties["good_fit"] = True
        if circle_pt_err < conformal_pt_err:
            properties["d0"] = circle_d0
        else:
            properties["d0"] = conformal_d0
        df_properties.append(properties)

    df = pd.concat(df_properties)
    outfile = join(output_dir, f"{evtid}.csv")
    logging.info(f"Writing {outfile}")
    df.to_csv(outfile, index=False)
    return 1


def main(args):
    initialize_logger()
    args = parse_args(args)
    logging.info(f"Args {args}")
    initialize_logger(verbose=args.verbose)
    input_dir = args.input_dir
    output_dir = args.output_dir
    # train_idx = int(input_dir.split("train_")[-1][0])
    logging.info(f"Running on data from {input_dir}.")
    file_prefixes = get_file_prefixes(
        input_dir, n_tasks=args.n_tasks, task=args.task, evtid_min=0, evtid_max=100000
    )

    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(make_df, output_dir=output_dir)
        pool.map(process_func, file_prefixes)

    logging.info("All done!")


if __name__ == "__main__":
    main(sys.argv[1:])
