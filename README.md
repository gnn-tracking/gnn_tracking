# GNNs for Charged Particle Tracking

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GageDeZoort/gnn_tracking/main.svg)](https://results.pre-commit.ci/latest/github/GageDeZoort/gnn_tracking/main)
[![gh actions](https://github.com/GageDeZoort/gnn_tracking/actions/workflows/test.yaml/badge.svg)](https://github.com/GageDeZoort/gnn_tracking/actions)
[![codecov](https://codecov.io/gh/GageDeZoort/gnn_tracking/branch/main/graph/badge.svg?token=3MKA387NOH)](https://codecov.io/gh/GageDeZoort/gnn_tracking)


![](readme_assets/banner.jpg)

## Setup and testing

1. Set up a conda environment with one of the `environment/*.yml` files
2. Run `pip3 install -e '.[testing]'`
3. Run `pytest` to check if everything worked

## Development setup

Install the pre-commit hooks with

```bash
pip3 install pre-commit
# cd to this directory
pre-commit install
```
