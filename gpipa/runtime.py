from __future__ import annotations

import os
from pathlib import Path

_FFMPEG_ENV = "GPIPA_FFMPEG_BIN"


def configure_windows_ffmpeg() -> Path | None:
    """Register a bundled FFmpeg DLL directory on Windows if available."""
    if os.name != "nt":
        return None

    candidates: list[Path] = []

    env_path = os.environ.get(_FFMPEG_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    workspace_bin = Path(__file__).resolve().parents[1] / "vendor" / "ffmpeg" / "bin"
    candidates.append(workspace_bin)

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue

        resolved = candidate.resolve()
        current_path = os.environ.get("PATH", "")
        resolved_str = str(resolved)

        if resolved_str not in current_path.split(os.pathsep):
            os.environ["PATH"] = resolved_str + os.pathsep + current_path

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            add_dll_directory(resolved_str)

        return resolved

    return None
