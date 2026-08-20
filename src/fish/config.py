from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "fish"
ENV_PATH = CONFIG_DIR / "fish.env"
ACCOUNTS_PATH = CONFIG_DIR / "accounts.yaml"
ACTIONS_LOG = CONFIG_DIR / "actions.log"

# RunPod network volume. Container disk is ephemeral — see docs/cloud.md.
RUNPOD_VOLUME_ROOT = Path("/workspace")
RUNPOD_DATA_DIR = RUNPOD_VOLUME_ROOT / "fish"


def on_runpod() -> bool:
    """True inside a RunPod container (pod id is injected into the environment)."""
    return bool(os.getenv("RUNPOD_POD_ID", "").strip())


def path_is_on_runpod_volume(path: Path, *, volume_root: Path = RUNPOD_VOLUME_ROOT) -> bool:
    """Return whether ``path`` resolves under the RunPod network volume."""
    try:
        path.expanduser().resolve().relative_to(volume_root.expanduser().resolve())
    except ValueError:
        return False
    return True


def require_runpod_persistent_path(path: Path, *, what: str = "FISH_DATA_DIR") -> None:
    """Fail fast on RunPod when a data path would land on ephemeral container disk."""
    if not on_runpod():
        return
    if path_is_on_runpod_volume(path):
        return
    raise RuntimeError(
        f"On RunPod, {what} must be under {RUNPOD_VOLUME_ROOT} so artifacts "
        f"survive pod stop (got {path}). Set FISH_DATA_DIR={RUNPOD_DATA_DIR} "
        f"(and keep FISH_DB_PATH under that tree). Run `fish runpod-setup` "
        f"or `python3 src/fish/runpod_setup.py` from a checkout on the volume."
    )


def data_dir() -> Path:
    """Corpus data root (db, models, imports). Default ~/.config/fish; cloud uses FISH_DATA_DIR."""
    load_env()
    raw = os.getenv("FISH_DATA_DIR", "").strip()
    path = Path(raw).expanduser() if raw else CONFIG_DIR
    require_runpod_persistent_path(path, what="FISH_DATA_DIR")
    path.mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(parents=True, exist_ok=True)
    (path / "imports").mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    load_env()
    raw = os.getenv("FISH_DB_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser()
        require_runpod_persistent_path(path, what="FISH_DB_PATH")
        return path
    return data_dir() / "fish.db"


def models_dir() -> Path:
    return data_dir() / "models"


def imports_dir() -> Path:
    return data_dir() / "imports"


def write_lock_path() -> Path:
    return db_path().with_suffix(".write.lock")
MEMORY_DEDUP_THRESHOLD = 0.95
MEMORY_MERGE_THRESHOLD = 0.85
MEMORY_LLM_MODEL = "gpt-4o-mini"

DEFAULT_SYNC_DAYS = 90
# On-demand sync before MCP/CLI search — keep the window short so tool calls finish.
SEARCH_SYNC_DAYS = 14
SEARCH_SYNC_MAX_AGE_SEC = 300
SEARCH_SYNC_LOCK_TIMEOUT_SEC = 5.0
SEARCH_SYNC_EMBED_BUDGET = 50
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_EMBED_TOKENS = 8192
MAX_EMBED_BODY_CHARS = 32_000
# OpenAI embeddings API limit is 300k tokens/request; stay under with margin.
EMBED_REQUEST_MAX_TOKENS = 250_000

SKIP_FOLDER_PATTERNS = (
    "[Gmail]/Trash",
    "[Gmail]/Spam",
    "Junk",
    "Trash",
)

# Recommended starting point for ~/.config/fish/accounts.yaml ignore_folders.
# Gmail: sync only [Gmail]/All Mail; ignore INBOX and other label mirrors.
RECOMMENDED_IGNORE_FOLDERS = (
    "INBOX",
    "Drafts",
    "[Gmail]/Sent Mail",
    "[Gmail]/Important",
    "[Gmail]/Starred",
    "[Gmail]/Drafts",
    "[Gmail]/Trash",
    "[Gmail]/Spam",
    "Junk",
    "Trash",
    "INBOX.spam",
    "INBOX.Trash",
)


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_env() -> None:
    ensure_config_dir()
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    load_dotenv(override=False)


def upsert_env_var(key: str, value: str) -> None:
    ensure_config_dir()
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    prefix = f"{key}="
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n")
    os.environ[key] = value


def is_plausible_openai_key(key: str) -> bool:
    key = key.strip()
    if not key:
        return False
    if key.startswith("http://") or key.startswith("https://"):
        return False
    return key.startswith("sk-")


def prompt_openai_api_key() -> str:
    print(f"Enter your OpenAI secret key (starts with sk-), not a URL.")
    print("Create one at https://platform.openai.com/api-keys")
    while True:
        key = getpass.getpass("OpenAI API key: ").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding mail")
        if is_plausible_openai_key(key):
            return key
        print(
            "Invalid key format — paste the secret key (sk-...), not the website URL.",
            file=sys.stderr,
        )


def openai_api_key() -> str:
    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY not set in {ENV_PATH}")
    if not is_plausible_openai_key(key):
        raise RuntimeError(
            f"OPENAI_API_KEY in {ENV_PATH} is invalid (expected sk-... secret, not a URL). "
            "Run `fish sync` again to re-enter."
        )
    return key


def ensure_openai_api_key(*, interactive: bool = True, force: bool = False) -> str:
    """Return OpenAI API key, prompting to save one if missing or invalid."""
    from fish.embed import reset_client

    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    if not force and key and is_plausible_openai_key(key):
        return key

    if key and not is_plausible_openai_key(key):
        print(
            f"OPENAI_API_KEY in {ENV_PATH} looks wrong (URL or bad format).",
            file=sys.stderr,
        )
    elif force:
        print("OpenAI rejected the API key (401 unauthorized).", file=sys.stderr)

    if not interactive:
        raise RuntimeError(f"OPENAI_API_KEY not valid in {ENV_PATH}")
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"OPENAI_API_KEY not valid in {ENV_PATH} and stdin is not interactive"
        )

    if not key:
        print(f"OpenAI API key not found in {ENV_PATH}")
    key = prompt_openai_api_key()
    upsert_env_var("OPENAI_API_KEY", key)
    reset_client()
    print(f"Saved OPENAI_API_KEY to {ENV_PATH}")
    return key


def embedding_model() -> str:
    load_env()
    return os.getenv("FISH_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)


def prism_model_path() -> Path | None:
    """Resolve .prz from FISH_PRISM_MODEL (model_id or filename under models/)."""
    load_env()
    raw = os.getenv("FISH_PRISM_MODEL", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = models_dir() / (raw if raw.endswith(".prz") else f"{raw}.prz")
    if not path.exists():
        raise RuntimeError(
            f"FISH_PRISM_MODEL is set to {raw!r} but file not found: {path}"
        )
    return path


def active_prism_model_id() -> str | None:
    """model_id for search: FISH_PRISM_MODEL env, else DB active flag."""
    load_env()
    raw = os.getenv("FISH_PRISM_MODEL", "").strip()
    if raw:
        name = Path(raw).name
        return name[:-4] if name.endswith(".prz") else name
    from fish.store import db_conn, init_db
    from fish.prism.registry import active_prism_model

    init_db()
    with db_conn() as db:
        active = active_prism_model(db)
        return active["model_id"] if active else None


def memory_llm_model() -> str:
    load_env()
    return os.getenv("FISH_MEMORY_LLM_MODEL", MEMORY_LLM_MODEL)
