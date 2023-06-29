# Environments

## Pick the right file

* `default`: Full environment including setup for helper repositories for GPU.
   Should work for Linux and Windows
* `minimal`: The environment that is being used for testing with github
   actions
* `macos`: For MacOS. See additional instructions below.

## Install it

```bash
micromamba create --name gnn --file default.yml
```

## MacOS

You need to pip-install pyg and friends:

```bash
pip install torch_geometric
pip install torch_scatter  # takes a while
```

## Explanations for lower limits

* `pytorch > 1.12`: https://github.com/pytorch/pytorch/issues/80809 (though
  the fix should already be available for fixup versions of 1.12)
