<div align="center">

# GNNs for Charged Particle Tracking

[![Documentation Status](https://readthedocs.org/projects/gnn-tracking/badge/?version=latest)](https://gnn-tracking.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gnn-tracking/gnn_tracking/main.svg)](https://results.pre-commit.ci/latest/github/gnn-tracking/gnn_tracking/main)
[![gh actions](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/test.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions)
[![Check Markdown links](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml)
[![codecov](https://codecov.io/gh/gnn-tracking/gnn_tracking/branch/main/graph/badge.svg?token=3MKA387NOH)](https://codecov.io/gh/gnn-tracking/gnn_tracking)


![](readme_assets/banner.jpg)

</div>

This repository holds the main python package for the GNN Tracking project.
See the [readme of the organization][organization-readme] for an overview of the task.

## 🔥 Install & run

1. Set up a conda environment with one of the `environment/*.yml` files
2. Run `pip3 install -e '.[testing]'`
3. Run `pytest` to check if everything worked

A good place to start are the [demo notebooks][demo]

## 🧰 Development setup

Install the pre-commit hooks with

```bash
pip3 install -e '.[testing,dev]'
pre-commit install
```

If you open a PR and pre-commit fails for formatting, comment`pre-commit.ci run`
to trigger a fixup commit from `pre-commit`.

## 💚 Contributing

A good place to start are the [issues marked with 'good first issue'][gfi]. It is always best to have the issue assigned to you before starting to work on it. You can also [reach us per mail][ml].

## 🖋️ Contact

[Write to our mailing list.][ml]

[organization-readme]: https://github.com/gnn-tracking
[gfi]: https://github.com/gnn-tracking/gnn_tracking/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22
[ml]: mailto:gnn-tracking@googlegroups.com
[demo]: https://github.com/gnn-tracking/tutorials
