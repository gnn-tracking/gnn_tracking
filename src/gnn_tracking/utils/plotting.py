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

    def plot_ep_rv_uv(self, evtid=None, savefig=False, filename=""):
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
        if savefig:
            plt.savefig(filename, dpi=1200, format="pdf")
        plt.show()


class PointCloudPlotter:
    def __init__(self, indir, n_sectors=64):
        self.indir = indir
        self.infiles = os.listdir(indir)
        self.n_sectors = n_sectors
        self.colors = cm.prism(np.linspace(0, 1, n_sectors))

    def plot_ep_rv_uv(
        self,
        i: int,
        sector: str,
        axs: Sequence[plt.Axes],
        display=True,
        pixel_only=False,
    ):
        x = torch.load(sector).x
        phi, eta = x[:, 1], x[:, 3]
        r, z = x[:, 0], x[:, 2]
        u, v = x[:, 4], x[:, 5]
        ms = 0.1 if pixel_only else 1.0
        kwargs = {"marker": ".", "lw": 0, "ms": ms, "color": self.colors[i]}
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

    def plot_ep_rv_uv_all_sectors(
        self, evtid: int, savefig=False, filename="", pixel_only=False
    ):
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(24, 8))
        sector_files = [join(self.indir, f) for f in self.infiles if str(evtid) in f]
        prefix = f"event{evtid}"
        for i, s in enumerate(sector_files):
            self.plot_ep_rv_uv(i, s, axs=axs, display=False, pixel_only=pixel_only)
        axs[1].set_title(prefix)
        if savefig:
            plt.savefig(filename, dpi=1200, format="pdf")
        plt.tight_layout()
        plt.show()

    def plot_ep_rv_uv_with_boundary(
        self,
        evtid: int,
        sector: int,
        di,
        ds,
        ulim_low=0,
        ulim_high=0.035,
        vlim_low=-0.004,
        vlim_high=0.004,
        savefig=False,
        filename="",
        pixel_only=False,
    ):
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
        ms = 0.5 if pixel_only else 3
        axs[0].plot(eta, phi, "b.", lw=0, ms=ms)
        axs[0].set_xlabel(r"$\eta$")
        axs[0].set_ylabel(r"$\phi$")
        axs[1].plot(z, r, "b.", lw=0, ms=ms)
        axs[1].set_xlabel(r"$z$ [mm]")
        axs[1].set_ylabel(r"$r$ [mm]")
        axs[2].plot(ur, vr, "b.", lw=0, ms=ms)
        axs[2].set_xlabel(r"$u_\mathrm{rotated}$ [1/mm]")
        axs[2].set_ylabel(r"$v_\mathrm{rotated}$ [1/mm]")
        axs[1].set_title(f"event{evtid}_s{sector}")
        axs[1].set_xlim([-1550, 1550])
        xr = np.arange(0, 0.035, 0.0001)
        axs[2].plot(xr, slope * xr, "k-", label="Original Sector")
        axs[2].plot(xr, -slope * xr, "k-")
        axs[2].plot(xr, ds * slope * xr + di, "k--", label="Extended Sector")
        axs[2].plot(xr, -ds * slope * xr - di, "k--")
        axs[2].set_xlim([ulim_low, ulim_high])
        axs[2].set_ylim([vlim_low, vlim_high])
        plt.legend(loc="best")
        plt.tight_layout()
        if savefig:
            plt.savefig(filename, dpi=1200, format="pdf")
        plt.show()


class GraphPlotter:
    def __init__(
        self,
        indir="",
        n_sectors=64,
    ):
        """Plotter for graph data.

        Args:
            indir: Input directory with graphs (if loading by name)
            n_sectors:
        """
        self.indir = indir
        self.n_sectors = n_sectors

    def configure_plt(self, style="seaborn-paper"):
        plt.style.use(style)
        rcParams.update({"figure.autolayout": True})

    def plot_ep_rz_uv(
        self,
        *,
        graph: Data,
        sector: int,
        name: str = "",
        filename="",
    ):
        """

        Args:
            graph:
            sector:
            name: If ``graph`` is not specified, load from ``self.indir / name``.
            filename: If specified, save figure to file

        Returns:

        """
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(24, 8))
        if graph is None:
            f = join(self.indir, f"{name}.pt")
            graph = torch.load(f)
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        phi, eta = x[:, 1], x[:, 3]
        r, z = x[:, 0], x[:, 2]

        # rotate u, v onto the u axis
        u, v = x[:, 4], x[:, 5]
        theta = np.pi / self.n_sectors
        ur = u * np.cos(2 * sector * theta) - v * np.sin(2 * sector * theta)
        vr = u * np.sin(2 * sector * theta) + v * np.cos(2 * sector * theta)

        # plot single_particles
        particle_id = graph.particle_id
        colors = ["red", "green", "purple", "yellow", "orange"]
        for i in range(len(colors)):
            particle_id = graph.particle_id
            mask = particle_id == np.random.choice(
                particle_id[particle_id != 0], size=None
            )
            kwargs = {"marker": ".", "ms": 8, "zorder": 100, "color": colors[i]}
            axs[0].plot(eta[mask], phi[mask] * np.pi, **kwargs)
            axs[1].plot(z[mask] * 1000.0, r[mask] * 1000.0, **kwargs)
            axs[2].plot(ur[mask] / 1000.0, vr[mask] / 1000.0, **kwargs)

        # plot others
        self.plot_2d(
            np.stack((eta, phi * np.pi), axis=1),
            y,
            edge_index,
            x1_label=r"$\eta$",
            x2_label=r"$\phi$",
            ax=axs[0],
        )
        self.plot_2d(
            np.stack((z * 1000.0, r * 1000.0), axis=1),
            y,
            edge_index,
            x1_label=r"$z$ [mm]",
            x2_label=r"$r$ [mm]",
            ax=axs[1],
        )
        self.plot_2d(
            np.stack((ur / 1000.0, vr / 1000.0), axis=1),
            y,
            edge_index,
            x1_label=r"$u$ [1/mm]",
            x2_label="$v$ [1/mm]",
            ax=axs[2],
        )
        axs[1].set_title(name)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=1200, format="pdf")
        plt.show()

    def plot_2d(
        self,
        X,
        y,
        edge_index,
        name="",
        ax=None,
        x1_label="",
        x2_label="",
        single_particle=False,
    ):
        true_x1_o = X[edge_index[0, :]][(y > 0.5)][:, 0]
        false_x1_o = X[edge_index[0, :]][(y < 0.5)][:, 0]
        true_x1_i = X[edge_index[1, :]][(y > 0.5)][:, 0]
        false_x1_i = X[edge_index[1, :]][(y < 0.5)][:, 0]
        true_x2_o = X[edge_index[0, :]][(y > 0.5)][:, 1]
        false_x2_o = X[edge_index[0, :]][(y < 0.5)][:, 1]
        true_x2_i = X[edge_index[1, :]][(y > 0.5)][:, 1]
        false_x2_i = X[edge_index[1, :]][(y < 0.5)][:, 1]

        show_plot = False
        if ax is None:
            show_plot = True
            fig, ax = plt.subplots(dpi=200, figsize=(12, 12))
        ax.plot(X[:, 0], X[:, 1], "b.", lw=0, ms=0.5)

        # plot true edges
        for i in range(len(true_x1_o)):
            ax.plot(
                (true_x1_o[i], true_x1_i[i]),
                (true_x2_o[i], true_x2_i[i]),
                marker="o",
                ls="-",
                color="blue" if not single_particle else "green",
                lw=0.25 if not single_particle else 1,
                ms=0.1 if not single_particle else 0.5,
                alpha=1,
            )

        # plot false edges
        for i in range(len(false_x1_o)):
            ax.plot(
                (false_x1_o[i], false_x1_i[i]),
                (false_x2_o[i], false_x2_i[i]),
                marker="o",
                ls="-",
                color="black" if not single_particle else "red",
                lw=0.05 if not single_particle else 0.5,
                ms=0.1 if not single_particle else 0.5,
                alpha=0.2,
            )

        ax.set_xlabel(x1_label)
        ax.set_ylabel(x2_label)

        if show_plot:
            plt.title(name)
            plt.tight_layout()
            plt.show()

    def plot_rz(
        self,
        graph: Data,
        name="",
        scale=None,
        savefig=False,
        filename="",
        ax=None,
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

        show_plot = False
        if ax is None:
            show_plot = True
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
        if show_plot:
            plt.tight_layout()
            plt.title(name)
            if savefig:
                plt.savefig(filename, dpi=1200)
            plt.show()


def plot_rz(X, idxs, y, savefig=False, filename="rz.png"):
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
    if savefig:
        plt.savefig(filename, dpi=1200)
    plt.tight_layout()
    plt.show()


def plot_3d(X, idxs, y, savefig=False, filename="rz.png"):
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
