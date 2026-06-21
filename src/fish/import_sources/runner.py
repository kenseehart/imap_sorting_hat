from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from fish.import_sources.android_sms import import_android_sms
from fish.import_sources.chatgpt import import_chatgpt_export, import_chatgpt_memory_json
from fish.import_sources.claude import import_claude_export
from fish.sync import embed_all_pending


def run_import(
    source: str,
    path: Path,
    *,
    dry_run: bool = False,
    phone_filter: str | None = None,
    embed: bool = True,
) -> dict[str, Any]:
    source = source.lower().replace("_", "-")
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if source == "android-sms":
        stats = import_android_sms(
            path, phone_filter=phone_filter or "8315352442", dry_run=dry_run
        )
    elif source == "chatgpt":
        stats = import_chatgpt_export(path, dry_run=dry_run)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                mem_files = [n for n in zf.namelist() if n.endswith("memory.json")]
            if mem_files:
                import tempfile

                with zipfile.ZipFile(path) as zf, tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False
                ) as tmp:
                    tmp.write(zf.read(mem_files[0]))
                    tmp_path = Path(tmp.name)
                mem_stats = import_chatgpt_memory_json(tmp_path, dry_run=dry_run)
                tmp_path.unlink(missing_ok=True)
                stats["memory_import"] = mem_stats
    elif source == "claude":
        stats = import_claude_export(path, dry_run=dry_run)
    else:
        raise ValueError(
            f"Unknown import source {source!r}. Use android-sms, chatgpt, or claude."
        )

    result: dict[str, Any] = {"source": source, "path": str(path), "stats": stats}
    if embed and not dry_run:
        result["embedded"] = embed_all_pending(show_progress=False)
    return result
