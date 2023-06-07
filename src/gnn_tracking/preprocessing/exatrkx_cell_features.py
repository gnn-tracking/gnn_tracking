from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from pandas import Series as S

from gnn_tracking.utils.log import get_logger

"""These functions have been adapted from ExaTrkx's preprocessing.
The source can be found at
https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/
in the file
Pipelines/TrackML_Example/LightningModules/Processing/utils/detector_utils.py
"""

cf_logger = get_logger("CF")


def load_detector(detector_path: Path) -> tuple[pd.DataFrame, dict]:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above.

    WARNING: This might create a race condition with creating the preprocessed file.
    """
    detector_orig = pd.read_csv(detector_path)
    detector_preproc = detector_path.parent / (detector_path.stem + ".pickle")
    try:
        cf_logger.info("Loading detector...")
        with open(detector_preproc, "rb") as f:
            detector = pickle.load(f)
        cf_logger.info("Detector loaded.")
    except FileNotFoundError:
        cf_logger.info("Failed to load preprocessed detector. Building...")
        detector = preprocess_detector(detector_orig)
        try:
            with open(detector_preproc, "xb") as f:
                pickle.dump(detector, f)
        except FileExistsError:
            cf_logger.warning(
                "Output file created in the meantime. This is because this "
                "function is not thread-safe. Shouldn't be a problem though."
            )
        else:
            cf_logger.info("Detector preprocessed and saved.")
    return detector_orig, detector


def preprocess_detector(detector: DF) -> dict:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""
    thicknesses = DetectorThicknesses(detector).get_thicknesses()
    rotations = DetectorRotations(detector).get_rotations()
    pixel_size = DetectorPixelSize(detector).get_pixel_size()
    det = dict(thicknesses=thicknesses, rotations=rotations, pixel_size=pixel_size)
    return det


def determine_array_size(detector: DF) -> tuple[float, float, float]:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""
    max_v, max_l, max_m = (0, 0, 0)
    unique_vols = detector.volume_id.unique()
    max_v = max(unique_vols) + 1
    for v in unique_vols:
        vol = detector.loc[detector["volume_id"] == v]
        unique_layers = vol.layer_id.unique()
        max_l = max(max_l, max(unique_layers) + 1)
        for l in unique_layers:
            lay = vol.loc[vol["layer_id"] == l]
            unique_modules = lay.module_id.unique()
            max_m = max(max_m, max(unique_modules) + 1)
    return max_v, max_l, max_m


class DetectorRotations:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    def __init__(self, detector: DF):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_rotations(self):
        cf_logger.debug("Extracting rotations...")
        self._init_rotation_array()
        self._extract_all_rotations()
        cf_logger.debug("Done.")
        return self.rot

    def _init_rotation_array(self):
        self.rot = np.zeros((self.max_v, self.max_l, self.max_m, 3, 3))

    def _extract_all_rotations(self):
        for _, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            rot = self._extract_rotation_matrix(r)
            self.rot[v, l, m] = rot

    def _extract_rotation_matrix(self, mod):
        """
        Extract the rotation matrix from module dataframe
        """
        r = np.matrix(
            [
                [mod.rot_xu.item(), mod.rot_xv.item(), mod.rot_xw.item()],
                [mod.rot_yu.item(), mod.rot_yv.item(), mod.rot_yw.item()],
                [mod.rot_zu.item(), mod.rot_zv.item(), mod.rot_zw.item()],
            ]
        )
        return r


class DetectorThicknesses:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    def __init__(self, detector: DF):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_thicknesses(self):
        cf_logger.debug("Extracting thicknesses...")
        self._init_thickness_array()
        self._extract_all_thicknesses()
        cf_logger.debug("Done.")
        return self.all_t

    def _init_thickness_array(self):
        self.all_t = np.zeros((self.max_v, self.max_l, self.max_m))

    def _extract_all_thicknesses(self):
        for _, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_t[v, l, m] = r.module_t


class DetectorPixelSize:
    """Adapted from ExaTrkX's preprocessing. See docstring above."""

    def __init__(self, detector: DF):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_pixel_size(self) -> np.ndarray:
        cf_logger.debug("Extracting thicknesses...")
        self._init_size_array()
        self._extract_all_size()
        return self.all_s

    def _init_size_array(self):
        self.all_s = np.zeros((self.max_v, self.max_l, self.max_m, 2))

    def _extract_all_size(self):
        for _, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_s[v, l, m, 0] = r.pitch_u
            self.all_s[v, l, m, 1] = r.pitch_v


def cartesion_to_spherical(x, y, z):
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r3)
    return r3, theta, phi


def theta_to_eta(theta):
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    return -np.log(np.tan(0.5 * theta))


def get_all_local_angles(hits: DF, cells: DF, detector: dict) -> tuple[S, S, S]:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    direction_count_u = cells.groupby(["hit_id"]).ch0.agg(["min", "max"])
    direction_count_v = cells.groupby(["hit_id"]).ch1.agg(["min", "max"])
    nb_u = direction_count_u["max"] - direction_count_u["min"] + 1
    nb_v = direction_count_v["max"] - direction_count_v["min"] + 1

    vols = hits["volume_id"].to_numpy()
    layers = hits["layer_id"].to_numpy()
    modules = hits["module_id"].to_numpy()

    pitch = detector["pixel_size"]
    thickness = detector["thicknesses"]

    pitch_cells = pitch[vols, layers, modules]
    thickness_cells = thickness[vols, layers, modules]

    l_u = nb_u * pitch_cells[:, 0]
    l_v = nb_v * pitch_cells[:, 1]
    l_w = 2 * thickness_cells
    return l_u, l_v, l_w


def get_all_rotated(hits: DF, detector: dict, l_u: S, l_v: S, l_w: S):
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    vols = hits["volume_id"].to_numpy()
    layers = hits["layer_id"].to_numpy()
    modules = hits["module_id"].to_numpy()
    rotations = detector["rotations"]
    rotations_hits = rotations[vols, layers, modules]
    u = l_u.values.reshape(-1, 1)
    v = l_v.values.reshape(-1, 1)
    w = l_w.reshape(-1, 1)
    dirs = np.concatenate((u, v, w), axis=1)

    dirs = np.expand_dims(dirs, axis=2)
    vecRot = np.matmul(rotations_hits, dirs).squeeze(2)
    return vecRot


def extract_dir_new(hits: DF, cells: DF, detector: dict) -> DF:
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""

    l_u, l_v, l_w = get_all_local_angles(hits, cells, detector)
    g_matrix_all = get_all_rotated(hits, detector, l_u, l_v, l_w)
    hit_ids, cell_counts, cell_vals = (
        hits["hit_id"].to_numpy(),
        hits["cell_count"].to_numpy(),
        hits["cell_val"].to_numpy(),
    )

    l_u, l_v = l_u.to_numpy(), l_v.to_numpy()

    _, g_theta, g_phi = np.vstack(cartesion_to_spherical(*list(g_matrix_all.T)))
    cf_logger.debug("G calc")
    _, l_theta, l_phi = cartesion_to_spherical(l_u, l_v, l_w)
    cf_logger.debug("L calc")
    l_eta = theta_to_eta(l_theta)
    g_eta = theta_to_eta(g_theta)

    angles = np.vstack(
        [hit_ids, cell_counts, cell_vals, l_eta, l_phi, l_u, l_v, l_w, g_eta, g_phi]
    ).T
    cf_logger.debug("Concated")
    df_angles = pd.DataFrame(
        angles,
        columns=[
            "hit_id",
            "cell_count",
            "cell_val",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        ],
    )
    cf_logger.info("DF constructed")

    return df_angles


def augment_hit_features(hits: DF, cells: DF, detector_proc: dict):
    """Adapted/copied from ExaTrkX's preprocessing. See docstring above."""
    hits["cell_count"] = (
        cells.groupby(["hit_id"]).value.count().to_numpy().astype(np.float32)
    )
    hits["cell_val"] = (
        cells.groupby(["hit_id"]).value.sum().to_numpy().astype(np.float32)
    )
    angles = extract_dir_new(hits, cells, detector_proc)
    # Drop duplicate columns before merging to avoid suffixes
    hits = pd.merge(
        hits.drop(columns=["cell_count", "cell_val"]), angles, on="hit_id", how="left"
    )
    assert "cell_count" in hits.columns, hits.columns
    return hits
