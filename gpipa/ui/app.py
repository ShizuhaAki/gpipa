from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

from gpipa.pipeline import pipeline

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yml"
RUNTIME_ROOT = WORKSPACE_ROOT / "output" / "ui-runs"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_config(config_path: str) -> tuple[str, str]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path

    if not path.exists():
        return "", f"Config not found: {path}"

    return _read_text(path), f"Loaded config from: {path}"


def save_config(config_path: str, config_text: str) -> str:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text, encoding="utf-8")
    return f"Saved config to: {path}"


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _collect_output_files(result: dict[str, Any]) -> list[str]:
    files: list[str] = []

    merged = result.get("merged_textgrid_path")
    if isinstance(merged, str) and Path(merged).exists():
        files.append(merged)

    for seg in result.get("segments", []):
        tg = seg.get("textgrid_path")
        if isinstance(tg, str) and tg and Path(tg).exists():
            files.append(tg)

    return _unique_paths(files)


def _build_segment_rows(result: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, seg in enumerate(result.get("segments", []), start=1):
        rows.append(
            [
                idx,
                seg.get("start_sec"),
                seg.get("end_sec"),
                seg.get("duration_sec"),
                seg.get("asr_text", ""),
                " ".join(seg.get("pinyin_tokens_tone", [])),
                " ".join(seg.get("ipa_tokens", [])),
                seg.get("textgrid_path", ""),
            ]
        )
    return rows


def _build_step_markdown(
    result: dict[str, Any],
    run_audio_path: Path,
    runtime_config_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("## Run Summary")
    lines.append(f"- audio: {run_audio_path}")
    lines.append(f"- runtime_config: {runtime_config_path}")
    lines.append(f"- mode: {result.get('mode', 'unknown')}")
    lines.append(f"- segments: {len(result.get('segments', []))}")

    merged = result.get("merged_textgrid_path")
    if merged:
        lines.append(f"- merged_textgrid: {merged}")

    for idx, seg in enumerate(result.get("segments", []), start=1):
        lines.append("")
        lines.append(f"## Segment {idx}")
        if seg.get("start_sec") is not None and seg.get("end_sec") is not None:
            lines.append(
                f"- span: {seg.get('start_sec'):.2f}s -> {seg.get('end_sec'):.2f}s"
            )
        if seg.get("duration_sec") is not None:
            lines.append(f"- duration: {seg.get('duration_sec'):.2f}s")

        lines.append("- step1_asr:")
        lines.append(f"  - text: {seg.get('asr_text', '').strip()}")

        pinyin_tokens = seg.get("pinyin_tokens_tone", [])
        align_tokens = seg.get("align_tokens_no_tone", [])
        ipa_tokens = seg.get("ipa_tokens", [])
        lines.append("- step2_g2p:")
        lines.append(f"  - pinyin_tone: {' '.join(pinyin_tokens)}")
        lines.append(f"  - align_tokens_no_tone: {' '.join(align_tokens)}")
        lines.append(f"  - ipa_tokens: {' '.join(ipa_tokens)}")

        lines.append("- step3_alignment:")
        lines.append(f"  - skipped: {bool(seg.get('skipped', False))}")
        lines.append(f"  - textgrid: {seg.get('textgrid_path')}")

    return "\n".join(lines)


def run_pipeline_ui(audio_path: str, config_text: str) -> tuple[str, str, list[list[Any]], list[str]]:
    if not audio_path:
        return "Please upload an audio file first.", "", [], []

    try:
        yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        return f"Invalid YAML in config editor:\n\n{exc}", "", [], []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNTIME_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_audio = Path(audio_path)
    run_audio_path = run_dir / source_audio.name
    shutil.copy2(source_audio, run_audio_path)

    runtime_config_path = run_dir / "config.yml"
    runtime_config_path.write_text(config_text, encoding="utf-8")

    try:
        result = pipeline(str(run_audio_path), config_path=str(runtime_config_path))
    except Exception as exc:  # noqa: BLE001
        return f"Pipeline failed:\n\n{exc}", "", [], []

    step_markdown = _build_step_markdown(result, run_audio_path, runtime_config_path)
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    rows = _build_segment_rows(result)
    files = _collect_output_files(result)
    return step_markdown, result_json, rows, files


def build_app() -> gr.Blocks:
    default_config_text = _read_text(DEFAULT_CONFIG_PATH)

    with gr.Blocks(title="GPIPA UI") as demo:
        gr.Markdown("# GPIPA Gradio UI")
        gr.Markdown(
            "Upload an audio file, edit config.yml, then run the pipeline and inspect ASR/G2P/alignment outputs."
        )

        with gr.Row():
            config_path = gr.Textbox(
                value=str(DEFAULT_CONFIG_PATH),
                label="Config path",
                lines=1,
            )
            load_btn = gr.Button("Load config.yml")
            save_btn = gr.Button("Save config.yml")

        config_editor = gr.Code(
            language="yaml",
            label="Config editor",
            value=default_config_text,
        )
        config_status = gr.Markdown()

        audio_input = gr.Audio(
            type="filepath",
            sources=["upload"],
            label="Audio file",
        )
        run_btn = gr.Button("Run pipeline", variant="primary")

        step_output = gr.Markdown(label="Step-by-step results")
        with gr.Accordion("Pipeline JSON", open=False):
            json_output = gr.Code(language="json", label="Result JSON")

        segment_table = gr.Dataframe(
            headers=[
                "segment",
                "start_sec",
                "end_sec",
                "duration_sec",
                "asr_text",
                "pinyin_tone",
                "ipa_tokens",
                "textgrid_path",
            ],
            datatype=["number", "number", "number", "number", "str", "str", "str", "str"],
            label="Segment outputs",
            row_count=(0, "dynamic"),
            col_count=(8, "fixed"),
        )

        file_output = gr.Files(label="Generated TextGrid files")

        load_btn.click(
            fn=load_config,
            inputs=[config_path],
            outputs=[config_editor, config_status],
        )
        save_btn.click(
            fn=save_config,
            inputs=[config_path, config_editor],
            outputs=[config_status],
        )
        run_btn.click(
            fn=run_pipeline_ui,
            inputs=[audio_input, config_editor],
            outputs=[step_output, json_output, segment_table, file_output],
        )

    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
    app = build_app()
    app.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch()
