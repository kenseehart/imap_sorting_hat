"""Bootstrap fish on a RunPod network volume.

Runnable as ``fish runpod-setup`` once the package is importable, or as
``python3 src/fish/runpod_setup.py`` from a checkout on ``/workspace``
(first session, before the venv exists).

Does not install torch — the RunPod image already has CUDA PyTorch; a venv
with ``--system-site-packages`` reuses it. Does not touch canonical fish.db
on gcp-e2-mcp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

RUNPOD_VOLUME_ROOT = Path("/workspace")
RUNPOD_DATA_DIR = RUNPOD_VOLUME_ROOT / "fish"
VENV_DIR = RUNPOD_DATA_DIR / ".venv"

# pyproject runtime deps needed to train — not MCP (fastmcp/mcp/ken-mcp), not torch.
PIP_PACKAGES = (
    "beautifulsoup4",
    "cryptography",
    "imapclient",
    "numpy",
    "openai",
    "python-dotenv",
    "pyyaml",
    "scikit-learn",
    "qdrant-client",
    "tiktoken",
    "tqdm",
)

EDITABLE_PACKAGES = (
    ("shared/cmdline", "cmdline"),
    ("compute", "compute"),
    ("util", "util"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root(repo: Path) -> Path:
    return repo.parent


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, check=check, text=True)


def _ensure_rsync() -> dict[str, Any]:
    if shutil.which("rsync"):
        return {"rsync": shutil.which("rsync"), "installed": False}
    if shutil.which("apt-get") is None:
        raise RuntimeError(
            "rsync is missing and apt-get is not available — install rsync "
            "before compute sync can use it."
        )
    _run(["apt-get", "update", "-qq"])
    _run(["apt-get", "install", "-y", "-qq", "rsync"])
    path = shutil.which("rsync")
    if not path:
        raise RuntimeError("apt-get install rsync succeeded but rsync is still not on PATH")
    return {"rsync": path, "installed": True}


def _ensure_venv() -> Path:
    python = Path(sys.executable)
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run([str(python), "-m", "venv", "--system-site-packages", str(VENV_DIR)])
    if not venv_python.is_file():
        raise RuntimeError(f"venv created at {VENV_DIR} but {venv_python} is missing")
    return venv_python


def _pip(venv_python: Path, args: list[str]) -> None:
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", *args])


def _on_runpod() -> bool:
    return bool(os.getenv("RUNPOD_POD_ID", "").strip())


def _write_fish_env(data_dir: Path) -> Path:
    """Stdlib-only so first-session setup does not need python-dotenv."""
    config_dir = Path.home() / ".config" / "fish"
    config_dir.mkdir(parents=True, exist_ok=True)
    env_path = config_dir / "fish.env"
    lines = env_path.read_text().splitlines() if env_path.is_file() else []
    prefix = "FISH_DATA_DIR="
    out = [line for line in lines if not line.startswith(prefix)]
    if out and out[-1].strip():
        out.append("")
    out.append(f"FISH_DATA_DIR={data_dir}")
    env_path.write_text("\n".join(out).rstrip() + "\n")
    os.environ["FISH_DATA_DIR"] = str(data_dir)
    return env_path


def _write_env_script(data_dir: Path, venv_python: Path) -> Path:
    path = data_dir / "env.sh"
    activate = venv_python.parent / "activate"
    path.write_text(
        "# Source on RunPod before fish train/eval.\n"
        f"export FISH_DATA_DIR={data_dir}\n"
        f"export PATH={venv_python.parent}:$PATH\n"
        f"source {activate}\n"
    )
    return path


def setup_runpod(*, allow_non_runpod: bool = False) -> dict[str, Any]:
    """Create persistent venv + env on the RunPod volume. Idempotent."""
    if not _on_runpod() and not allow_non_runpod:
        raise RuntimeError(
            "Not a RunPod container (RUNPOD_POD_ID unset). "
            "Pass --allow-non-runpod only to test the installer."
        )
    if _on_runpod() and not RUNPOD_VOLUME_ROOT.is_dir():
        raise RuntimeError(
            f"{RUNPOD_VOLUME_ROOT} is missing — attach network volume "
            f"daime_prism_volume (or the resource volume_mount) before setup."
        )

    data_dir = RUNPOD_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (data_dir / "imports").mkdir(parents=True, exist_ok=True)

    fish_env = _write_fish_env(data_dir)
    rsync_info = _ensure_rsync()
    venv_python = _ensure_venv()
    _pip(venv_python, ["pip", "wheel"])
    _pip(venv_python, list(PIP_PACKAGES))

    repo = _repo_root()
    ws = _workspace_root(repo)
    editable: dict[str, str] = {}
    missing: list[str] = []
    for rel, name in EDITABLE_PACKAGES:
        src = ws / rel
        if src.is_dir() and (src / "pyproject.toml").is_file():
            _pip(venv_python, ["-e", str(src), "--no-deps"])
            editable[name] = str(src)
        else:
            missing.append(f"{name} ({src})")
    if not (repo / "pyproject.toml").is_file():
        raise RuntimeError(f"fish checkout not found at {repo}")
    if "cmdline" not in editable or "compute" not in editable:
        raise RuntimeError(
            "Need editable cmdline and compute checkouts next to fish "
            f"(looked under {ws}; missing {missing}). "
            f"From the laptop: cd ~/ws && compute sync runpod-l4 --push fish "
            f"&& compute sync runpod-l4 --push compute "
            f"&& compute sync runpod-l4 --push shared/cmdline"
        )
    _pip(venv_python, ["-e", str(repo), "--no-deps"])
    editable["fish"] = str(repo)

    env_script = _write_env_script(data_dir, venv_python)
    return {
        "data_dir": str(data_dir),
        "fish_env": str(fish_env),
        "env_script": str(env_script),
        "venv_python": str(venv_python),
        "rsync": rsync_info,
        "editable": editable,
        "missing_editable": missing,
        "torch": "use image torch via venv --system-site-packages (not pip-installed)",
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    allow = "--allow-non-runpod" in argv
    try:
        result = setup_runpod(allow_non_runpod=allow)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("RunPod fish setup complete.", flush=True)
    for key, value in result.items():
        print(f"  {key}: {value}", flush=True)
    if result.get("missing_editable"):
        print(
            "Missing sibling checkouts (sync ~/ws/{compute,util,shared} onto "
            f"{RUNPOD_VOLUME_ROOT} and re-run): {result['missing_editable']}",
            file=sys.stderr,
        )
    print(
        f"Next: source {result['env_script']} && fish prism-train --config smoke_joint --overfit",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
