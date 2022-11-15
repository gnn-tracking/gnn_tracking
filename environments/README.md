# Environments

> **Warning**
> Always install from the `.conda-lock.yml` files.

> **Warning**
> Do not apply changes to the `.conda-lock.yml` files, they are auto-generated with
> `conda-lock`.

## Pick the right file

* `testing.yml`: The environment that is being used for testing with github
   actions
* `default.yml`: Full environment including helper repositories for GPU.
   Should work for Linux and Windows
* `macos.yml`: Full environment including helper repositories for CPU for
   Macos.

## Install it

```bash
conda install -c conda-forge conda-lock
conda-lock install --name gnn  default.conda-lock.yml
```
