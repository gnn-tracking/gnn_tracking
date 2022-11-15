# Environments

> **Warning**
> Always install from the `.conda-lock.yml` files.

> **Warning**
> Do not apply changes to the `.conda-lock.yml` files, they are auto-generated with
> `conda-lock`.

## Pick the right file

* `minimal.yml`: The environment that is being used for testing with github
   actions
* `default.yml`: Full environment including setup for helper repositories for GPU.
   Should work for Linux and Windows
* `macos.yml`: Full environment including setup helper repositories for CPU for
   Macos.

## Install it

```bash
conda install -c conda-forge conda-lock
conda-lock install --name gnn  default.conda-lock.yml
```

## Updating lock files

Run

```bash
make
```

If you run into [this issue](https://github.com/conda-incubator/conda-lock/issues/283),
you might have to remove the lock files first and run without the `--lockfile`
argument.
