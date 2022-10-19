from __future__ import annotations

from os import PathLike
from types import ModuleType

from gnn_tracking.utils.log import logger


# todo: This will fail in a non-editable install
def get_commit_hash(module: None | ModuleType | str | PathLike = None) -> str:
    """Get the git commit hash of the current module."""
    import git

    if module is None:
        import gnn_tracking

        module = gnn_tracking
    if isinstance(module, ModuleType):
        base_path = module.__path__[0]
    else:
        base_path = module
    repo = git.Repo(path=base_path, search_parent_directories=True)
    if repo.is_dirty():
        logger.warning("Repository is dirty, commit hash may not be accurate.")
    return repo.head.object.hexsha
