# Environments

## Pick the right file

* `default`: Full environment including setup for helper repositories for GPU.
   Should work for Linux and Windows
* `minimal`: The environment that is being used for testing with github
   actions

## Install it

```bash
micromamba env create --name gnn --file default.yml
```
