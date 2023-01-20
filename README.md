<div align="center">

# GNNs for Charged Particle Tracking

[![Documentation Status](https://readthedocs.org/projects/gnn-tracking/badge/?version=latest)](https://gnn-tracking.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gnn-tracking/gnn_tracking/main.svg)](https://results.pre-commit.ci/latest/github/gnn-tracking/gnn_tracking/main)
[![gh actions](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/test.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions)
[![Check Markdown links](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml)
[![codecov](https://codecov.io/gh/gnn-tracking/gnn_tracking/branch/main/graph/badge.svg?token=3MKA387NOH)](https://codecov.io/gh/gnn-tracking/gnn_tracking)


![](readme_assets/banner.jpg)

</div>

## Setup and testing

1. Set up a conda environment with one of the `environment/*.yml` files
2. Run `pip3 install -e '.[testing]'`
3. Run `pytest` to check if everything worked

## Development setup

Install the pre-commit hooks with

```bash
pip3 install -e '.[testing,dev]'
pre-commit install
```

If you open a PR and pre-commit fails for formatting,, comment`pre-commit.ci run`
to trigger a fixup commit from `pre-commit`.
