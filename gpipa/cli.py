import argparse
import json
import logging
from pathlib import Path
from rich.logging import RichHandler
from gpipa.pipeline import pipeline
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()], force=True)
logger = logging.getLogger(__name__)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GPIPA pipeline for one audio file and print alignment outputs."
    )
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config yaml (default: config.yml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full pipeline result in JSON format",
    )
    return parser


def _print_summary(result: dict) -> None:
    mode = result.get("mode", "unknown")
    segments = result.get("segments", [])

    print(f"Mode: {mode}")
    print(f"Segments: {len(segments)}")

    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start_sec")
        end = seg.get("end_sec")
        duration = seg.get("duration_sec")
        tg = seg.get("textgrid_path", "")

        if start is not None and end is not None and duration is not None:
            print(
                f"[{idx}] {start:.2f}s -> {end:.2f}s "
                f"(dur {duration:.2f}s) | TextGrid: {tg}"
            )
        else:
            print(f"[{idx}] TextGrid: {tg}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio)
    config_path = Path(args.config)

    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    result = pipeline(str(audio_path), config_path=str(config_path))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
