from __future__ import annotations

import os
from os.path import join
from typing import Sequence

import mplhep as hep
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import cm
from torch_geometric.data import Data
from trackml.dataset import load_event

plt.style.use(hep.style.CMS)


class EventPlotter:
    def __init__(
        self,
        indir,
    ):
        self.indir = indir
        self.infiles = os.listdir(self.indir)

    def calc_eta(self, r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def append_coordinates(
        self, hits: pd.DataFrame, truth: pd.DataFrame, particles: pd.DataFrame
    ) -> pd.DataFrame:
        particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
        particles["eta_pt"] = self.calc_eta(particles.pt, particles.pz)
        truth = truth[["hit_id", "particle_id"]].merge(
            particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]],
            on="particle_id",
        )
        hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
        hits["phi"] = np.arctan2(hits.y, hits.x)
        hits["eta"] = self.calc_eta(hits.r, hits.z)
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # select the data columns we need
        hits = hits[
            ["hit_id", "r", "phi", "eta", "x", "y", "z", "u", "v", "volume_id"]
        ].merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")
        return hits

    def get_hits(self, evtid=None):
        if evtid is None:
            evtid = 21000 + np.random.randint(999, size=1)[0]
        else:
            evtid = str(evtid)
        prefix = f"event0000{evtid}"
        path = join(self.indir, prefix)
        hits, particles, truth = load_event(path, parts=["hits", "particles", "truth"])
        hits = self.append_coordinates(hits, truth, particles)
        return hits, prefix

    def plot_ep_rv_uv(self, evtid=None):
        hits, prefix = self.get_hits(evtid)
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(24, 8))
        axs[0].plot(hits["eta"], hits["phi"], "b.", lw=0, ms=0.1)
        axs[0].set_xlabel(r"$\eta$")
        axs[0].set_ylabel(r"$\phi$")
        axs[1].plot(hits["z"], hits["r"], "b.", lw=0, ms=0.1)
        axs[1].set_xlabel(r"$z$")
        axs[1].set_ylabel(r"$r$")
        axs[2].plot(hits["u"], hits["v"], "b.", lw=0, ms=0.1)
        axs[2].set_xlabel(r"u")
        axs[2].set_ylabel(r"v")
        axs[1].set_title(prefix)
        plt.tight_layout()
        plt.show()


class PointCloudPlotter:
    def __init__(self, indir, n_sectors=64):
        self.indir = indir
        self.infiles = os.listdir(indir)
        self.n_sectors = n_sectors
        self.colors = cm.prism(np.linspace(0, 1, n_sectors))

    def plot_ep_rv_uv(self, i: int, sector: str, axs: Sequence[plt.Axes], display=True):
        x = torch.load(sector).x
        phi, eta = x[:, 1], x[:, 3]
        r, z = x[:, 0], x[:, 2]
        u, v = x[:, 4], x[:, 5]
        kwargs = {"lw": 0, "ms": 0.1, "color": self.colors[i]}
        axs[0].plot(eta, phi, **kwargs)
        axs[0].set_xlabel(r"$\eta$")
        axs[0].set_ylabel(r"$\phi$")
        axs[1].plot(z, r, **kwargs)
        axs[1].set_xlabel(r"$z$ [mm]")
        axs[1].set_ylabel(r"$r$ [mm]")
        axs[1].set_xlim(-1550, 1550)
        axs[2].plot(u, v, **kwargs)
        axs[2].set_xlabel(r"u [1/mm]")
        axs[2].set_ylabel(r"v [1/mm]")
        if display:
            plt.tight_layout()
            plt.show()

    def plot_ep_rv_uv_all_sectors(self, evtid: int):
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(24, 8))
        sector_files = [join(self.indir, f) for f in self.infiles if str(evtid) in f]
        prefix = f"event{evtid}"
        for i, s in enumerate(sector_files):
            self.plot_ep_rv_uv(i, s, axs=axs, display=False)
        axs[1].set_title(prefix)
        plt.tight_layout()
        plt.show()

    def plot_ep_rv_uv_with_boundary(self, evtid: int, sector: int, di, ds):
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(24, 8))
        f = join(self.indir, f"data{evtid}_s{sector}.pt")
        x = torch.load(f).x
        phi, eta = x[:, 1], x[:, 3]
        r, z = x[:, 0], x[:, 2]
        u, v = x[:, 4], x[:, 5]
        theta = np.pi / self.n_sectors
        slope = np.arctan(theta)
        ur = u * np.cos(2 * sector * theta) - v * np.sin(2 * sector * theta)
        vr = u * np.sin(2 * sector * theta) + v * np.cos(2 * sector * theta)
        axs[0].plot(eta, phi, "b.", lw=0, ms=0.5)
        axs[0].set_xlabel(r"$\eta$")
        axs[0].set_ylabel(r"$\phi$")
        axs[1].plot(z, r, "b.", lw=0, ms=0.5)
        axs[1].set_xlabel(r"$z$ [mm]")
        axs[1].set_ylabel(r"$r$ [mm]")
        axs[2].plot(ur, vr, "b.", lw=0, ms=0.5)
        axs[2].set_xlabel(r"$u_\mathrm{rotated}$ [1/mm]")
        axs[2].set_ylabel(r"$v_\mathrm{rotated}$ [1/mm]")
        axs[1].set_title(f"event{evtid}_s{sector}")
        axs[1].set_xlim([-1550, 1550])
        xr = np.arange(0, 0.035, 0.0001)
        axs[2].plot(xr, slope * xr, "k-", label="Original Sector")
        axs[2].plot(xr, -slope * xr, "k-")
        axs[2].plot(xr, ds * slope * xr + di, "k--", label="Extended Sector")
        axs[2].plot(xr, -ds * slope * xr - di, "k--")
        axs[2].set_xlim([0, 0.035])
        axs[2].set_ylim([-0.002, 0.002])
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


class GraphPlotter:
    def __init__(self, style="seaborn-paper"):
        self.style = style

    def configure_plt(self):
        plt.style.use(self.style)
        rcParams.update({"figure.autolayout": True})

    def plot_rz(
        self,
        graph: Data,
        name: str,
        scale=None,
        savefig=False,
        filename="",
    ):
        x = graph.x[:, :3] / scale
        y = graph.y
        edge_index = graph.edge_index
        feats_o = x[edge_index[0, :]]
        feats_i = x[edge_index[1, :]]
        true_edges_o = feats_o[y > 0.5]
        true_edges_i = feats_i[y > 0.5]
        false_edges_o = feats_o[y < 0.5]
        false_edges_i = feats_i[y < 0.5]

        fig, ax = plt.subplots(dpi=200, figsize=(12, 12))
        for i in range(len(true_edges_o)):
            ax.plot(
                (true_edges_o[i][2], true_edges_i[i][2]),
                (true_edges_o[i][0], true_edges_i[i][0]),
                marker="o",
                ls="-",
                color="blue",
                lw=0.25,
                ms=0.1,
                alpha=1,
            )
        for i in range(len(false_edges_o)):
            ax.plot(
                (false_edges_o[i][2], false_edges_i[i][2]),
                (false_edges_o[i][0], false_edges_i[i][0]),
                marker="o",
                ls="-",
                color="black",
                lw=0.15,
                ms=0.1,
                alpha=0.4,
            )

        ax.set_ylabel("r [m]")
        ax.set_xlabel("z [m]")
        plt.title(name)
        if savefig:
            plt.savefig(filename, dpi=1200)
        plt.tight_layout()
        plt.show()


def plot_rz(X, idxs, y, save_fig=False, filename="rz.png"):
    X = np.array(X)
    feats_o = X[idxs[0, :]]
    feats_i = X[idxs[1, :]]

    for i in range(len(X)):
        plt.scatter(X[i][2], X[i][0], c="silver", linewidths=0, marker="s", s=15)

    track_segs_o = feats_o[y > 0.5]
    track_segs_i = feats_i[y > 0.5]
    for i in range(len(track_segs_o)):
        plt.plot(
            (track_segs_o[i][2], track_segs_i[i][2]),
            (track_segs_o[i][0], track_segs_i[i][0]),
            marker="o",
            ls="-",
            color="blue",
            lw=0.3,
            ms=0.1,
            alpha=1,
        )

    false_edges_o = feats_o[y < 0.5]
    false_edges_i = feats_i[y < 0.5]
    for i in range(len(false_edges_o)):
        plt.plot(
            (false_edges_o[i][2], false_edges_i[i][2]),
            (false_edges_o[i][0], false_edges_i[i][0]),
            marker="o",
            ls="-",
            color="black",
            lw=0.1,
            ms=0.1,
            alpha=0.25,
        )

    plt.ylabel("r [m]")
    plt.xlabel("z [m]")
    # plt.title(f'Sector: ({label[0]}, {label[1]})')
    if save_fig:
        plt.savefig(filename, dpi=1200)
    plt.tight_layout()
    plt.show()


def plot_3d(X, idxs, y, save_fig=False, filename="rz.png"):
    X = np.array(X)
    r, phi, z = X[:, 0], X[:, 1], X[:, 2]
    pred = y
    x, y = r * np.cos(phi), r * np.sin(phi)
    x_o, y_o, z_o = x[idxs[0, :]], y[idxs[0, :]], z[idxs[0, :]]
    x_i, y_i, z_i = x[idxs[1, :]], y[idxs[1, :]], z[idxs[1, :]]

    # feats_o = X[idxs[0,:]]
    # feats_i = X[idxs[1,:]]

    ax = plt.axes(projection="3d")
    for _ in range(len(X)):
        ax.scatter3D(x, y, z, c="silver", marker="s", s=15)
        # plt.scatter(X[i][2], X[i][0], c='silver', linewidths=0, marker='s', s=8)

    xt_o, yt_o, zt_o = x_o[pred > 0.5], y_o[pred > 0.5], z_o[pred > 0.5]
    xt_i, yt_i, zt_i = x_i[pred > 0.5], y_i[pred > 0.5], z_i[pred > 0.5]
    # track_segs_o = feats_o[y>0.5]
    # track_segs_i = feats_i[y>0.5]
    for i in range(len(xt_o)):
        ax.plot3D(
            (xt_o[i], xt_i[i]),
            (yt_o[i], yt_i[i]),
            (zt_o[i], zt_i[i]),
            marker="o",
            ls="-",
            color="blue",
            lw=0.25,
            ms=0,
            alpha=1,
        )

    xf_o, yf_o, zf_o = x_o[pred < 0.5], y_o[pred < 0.5], z_o[pred < 0.5]
    xf_i, yf_i, zf_i = x_i[pred < 0.5], y_i[pred < 0.5], z_i[pred < 0.5]
    # false_edges_o = feats_o[y<0.5]
    # false_edges_i = feats_i[y<0.5]
    for i in range(len(xf_o)):
        ax.plot3D(
            (xf_o[i], xf_i[i]),
            (yf_o[i], yf_i[i]),
            (zf_o[i], zf_i[i]),
            marker="h",
            ls="-",
            color="black",
            lw=0.15,
            ms=0,
            alpha=0.4,
        )

    ax.set_xlabel("x [m]", labelpad=25)
    ax.set_ylabel("y [m]", labelpad=25)
    ax.set_zlabel("z [m]", labelpad=25)
    # plt.title(f'Sector: ({label[0]}, {label[1]})')
    # if (save_fig): plt.savefig(filename, dpi=1200)
    plt.show()
