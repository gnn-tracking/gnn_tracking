"""Build point clouds.

Expected run time: ~1s / file for 32 sectors and pixel only.
One stream has ~900 files.
"""

from __future__ import annotations

import argparse
import logging
import os

from point_cloud_builder import PointCloudBuilder


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build point clouds")
    parser.add_argument(
        "--indir",
        type=str,
        help="Input directory",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory",
    )
    default_start = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument(
        "--start",
        type=int,
        default=default_start,
        help="We'll start at this value * batch size. Default will be slurm array"
        " index, if available, else 0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files to process. If set to 0: process all.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    pc_builder = PointCloudBuilder(
        indir=args.indir,
        outdir=args.outdir,
        n_sectors=32,
        pixel_only=True,
        redo=True,
        measurement_mode=False,
        sector_di=0,
        sector_ds=1.3,
        thld=0.9,
        log_level=logging.WARNING,
        collect_data=False,
    )
    start = args.start * args.batch_size
    stop = None
    if args.batch_size > 0:
        stop = start + args.batch_size
    pc_builder.process(start=start, stop=stop, ignore_loading_errors=True)
