from pathlib import Path
import os

def resolve_root(root: Path | None, create: bool = False) -> Path:
    """
    Resolve the model root directory.

    Precedence:
      1) Use the provided `root` argument if not None.
      2) Else, use the `PRT_MODEL_ROOT` environment variable if set and non-empty.
      3) Else, default to `~/models`.

    Parameters
    ----------
    root : Path | None
        Candidate root directory.
    create : bool, optional (default: False)
        If True, create the directory (with parents) when it does not exist.

    Returns
    -------
    Path
        Absolute, user-expanded path.

    Raises
    ------
    NotADirectoryError
        If the resolved path exists and is not a directory.
    """
    if root is not None:
        path = Path(root).expanduser()
    else:
        env_val = os.getenv("PRT_MODEL_ROOT", "").strip()
        path = Path(env_val).expanduser() if env_val else (Path.home() / "models")

    if create:
        path.mkdir(parents=True, exist_ok=True)

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Resolved path exists but is not a directory: {path}")

    return path.resolve()