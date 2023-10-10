from pathlib import Path

def get_repo_root() -> Path:
    """
    Returns a pathlib.Path object representing the root directory of this repo.
    """
    return Path(__file__).parent.parent.parent.absolute()
