import argparse
import os

from graph_builder import GraphBuilder


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
    graph_builder = GraphBuilder(
        indir=args.indir,
        outdir=args.outdir,
        redo=True,
        measurement_mode=False,
        phi_slope_max=0.001825,
        z0_max=197.4,
        dR_max=1.797,
        log_level=0,
        collect_data=False,
        remove_intersecting=False,
    )
    start = args.start * args.batch_size
    stop = None
    if args.batch_size > 0:
        stop = start + args.batch_size
    graph_builder.process(start=start, stop=stop)
