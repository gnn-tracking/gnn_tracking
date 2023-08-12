<div align="center">

# GNNs for Charged Particle Tracking

[![DOI](https://zenodo.org/badge/516883615.svg)](https://zenodo.org/badge/latestdoi/516883615)
[![CalVer YY.0M.MICRO](https://img.shields.io/badge/calver-YY.0M.MICRO-22bfda.svg)][calver]
[![Documentation Status](https://readthedocs.org/projects/gnn-tracking/badge/?version=latest)](https://gnn-tracking.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gnn-tracking/gnn_tracking/main.svg)](https://results.pre-commit.ci/latest/github/gnn-tracking/gnn_tracking/main)
[![gh actions](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/test.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions)
[![Check Markdown links](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml/badge.svg)](https://github.com/gnn-tracking/gnn_tracking/actions/workflows/check-links.yaml)
[![codecov](https://codecov.io/gh/gnn-tracking/gnn_tracking/branch/main/graph/badge.svg?token=3MKA387NOH)](https://codecov.io/gh/gnn-tracking/gnn_tracking)


![](readme_assets/banner.jpg)

</div>

This repository holds the main python package for the GNN Tracking project.
See the [readme of the organization][organization-readme] for an overview of the task.

* 🔋 Batteries included: This repository implements a hole pipeline: from preprocessing to models,
  to the evaluation of the final performance metrics.
* ⚡ Built around [pytorch lightning][], our models are easy to train and to restore. By using
  hooks and callbacks, everything remains modular and maintainable.
* ✅ Tested: Most of the code is guaranteed to run

[pytorch lightning]: lightning.ai/

## 🔥 Installation

1. Install mamba or micromamba ([installation instructions][mamba install]).
   Conda works as well, but will be slow to solve the environment, so it's not
   recommended.
2. Set up your environment with one of the `environment/*.yml` files (see the
   readme in that folder)
3. Run `pip3 install -e '.[testing,dev]'` from this directory.
4. Run `pytest` from this directory to check if everything worked
5. For development: Install [pre-commit][] hooks: `pre-commit install` (from this directory)

A good place to get started are the [demo notebooks][demo].
This package is versioned as [![CalVer YY.0M.MICRO](https://img.shields.io/badge/calver-YY.0M.MICRO-22bfda.svg)][calver].

[mamba install]: https://mamba.readthedocs.io/en/latest/installation.html
[demo]: https://github.com/gnn-tracking/tutorials
[pre-commit]: https://pre-commit.com
[calver]: https://calver.org/

## 🧰 Development guidelines

If you open a PR and pre-commit fails for formatting, comment`pre-commit.ci autofix`
to trigger a fixup commit from `pre-commit`.

To skip the slowest tests with `pytest`, run `pytest --no-slow`.

## 💚 Contributing

A good place to start are the [issues marked with 'good first issue'][gfi]. It is always best to have the issue assigned to you before starting to work on it. You can also [reach us per mail][ml].

Core developers ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GageDeZoort"><img src="https://avatars.githubusercontent.com/u/19605692?v=4?s=100" width="100px;" alt="Gage DeZoort"/><br /><sub><b>Gage DeZoort</b></sub></a><br /><a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=GageDeZoort" title="Code">💻</a> <a href="#ideas-GageDeZoort" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.lieret.net/"><img src="https://avatars.githubusercontent.com/u/13602468?v=4?s=100" width="100px;" alt="Kilian Lieret"/><br /><sub><b>Kilian Lieret</b></sub></a><br /><a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=klieret" title="Code">💻</a> <a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=klieret" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

Thanks also goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shubhanshu02"><img src="https://avatars.githubusercontent.com/u/54344426?v=4?s=100" width="100px;" alt="Shubhanshu Saxena"/><br /><sub><b>Shubhanshu Saxena</b></sub></a><br /><a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=shubhanshu02" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://kingjuno.github.io/"><img src="https://avatars.githubusercontent.com/u/69108486?v=4?s=100" width="100px;" alt="Geo Jolly"/><br /><sub><b>Geo Jolly</b></sub></a><br /><a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=kingjuno" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jnpark3"><img src="https://avatars.githubusercontent.com/u/49174255?v=4?s=100" width="100px;" alt="Jian Park"/><br /><sub><b>Jian Park</b></sub></a><br /><a href="https://github.com/gnn-tracking/gnn_tracking/commits?author=jnpark3" title="Code">💻</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## 🖋️ Contact

[Write to our mailing list.][ml]

[organization-readme]: https://github.com/gnn-tracking
[gfi]: https://github.com/gnn-tracking/gnn_tracking/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22
[ml]: mailto:gnn-tracking@googlegroups.com
