#!/usr/bin/env python3
r"""
Comfy Metadata Rescue
---------------------
Recover embedded generation metadata from ComfyUI/Krea PNGs (and other image
formats when ExifTool is installed), then export:

- readable_settings.md
- original_workflow.json
- fixed_seed_workflow.json
- api_prompt.json
- extraction_summary.json

No third-party Python packages are required.

Usage:
    python krea_metadata_recover.py "C:\\path\\to\\image.png"
    python krea_metadata_recover.py "C:\path\to\metadata_dump.txt"
    python krea_metadata_recover.py "C:\folder\of\images" --batch

Optional:
    - Install ExifTool to add JPEG/WEBP metadata support:
      winget install -e --id OliverBetz.ExifTool
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
EXIF_DUMP_EXTENSIONS = {".txt"}


@dataclass
class LoraInfo:
    name: str
    strength: Any = None
    strength_clip: Any = None
    enabled: bool = True
    node_id: str | None = None


@dataclass
class SamplerInfo:
    node_id: str
    class_type: str
    seed: Any = None
    seed_mode: str | None = None
    steps: Any = None
    cfg: Any = None
    sampler_name: Any = None
    scheduler: Any = None
    denoise: Any = None


@dataclass
class RecoveredRecipe:
    source_file: str
    metadata_source: str
    image_width: int | None = None
    image_height: int | None = None
    saved_aspect_ratio: str | None = None
    requested_aspect_ratio: str | None = None
    model: str | None = None
    clip: str | None = None
    clip_type: str | None = None
    vae: str | None = None
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    loras: list[LoraInfo] = field(default_factory=list)
    samplers: list[SamplerInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MetadataRecoveryError(RuntimeError):
    """Raised when a source cannot be read or does not contain supported metadata."""


def json_or_none(value: Any) -> Any:
    """Parse JSON-like values while preserving already parsed dicts/lists."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def find_metadata_value(metadata: dict[str, Any], field_name: str) -> Any:
    """Find keys like Prompt, PNG:Prompt, or [PNG] Prompt without guessing groups."""
    wanted = field_name.casefold()
    for key, value in metadata.items():
        normalized = str(key).replace("[", "").replace("]", "").strip()
        tail = normalized.replace(":", " ").split()[-1].casefold() if normalized else ""
        if str(key).casefold() == wanted or tail == wanted:
            return value
    return None


def decode_text(data: bytes) -> str:
    """Decode PNG text as gracefully as possible."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def parse_png_text_chunks(path: Path) -> dict[str, Any]:
    """
    Read standard PNG tEXt / zTXt / iTXt chunks without Pillow or ExifTool.
    This is intentionally narrow: it handles the kinds of embedded metadata used
    by ComfyUI/Krea PNGs and provides actual image dimensions from IHDR.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    with path.open("rb") as handle:
        if handle.read(8) != signature:
            raise MetadataRecoveryError(f"{path.name} is not a valid PNG.")

        metadata: dict[str, Any] = {}
        while True:
            length_raw = handle.read(4)
            if not length_raw:
                break
            if len(length_raw) != 4:
                raise MetadataRecoveryError(f"{path.name} ended in the middle of a PNG chunk.")

            length = struct.unpack(">I", length_raw)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(length)
            crc = handle.read(4)

            if len(chunk_type) != 4 or len(chunk_data) != length or len(crc) != 4:
                raise MetadataRecoveryError(f"{path.name} has a truncated PNG chunk.")

            if chunk_type == b"IHDR" and len(chunk_data) >= 8:
                width, height = struct.unpack(">II", chunk_data[:8])
                metadata["ImageWidth"] = width
                metadata["ImageHeight"] = height

            elif chunk_type == b"tEXt":
                keyword, separator, text = chunk_data.partition(b"\x00")
                if separator:
                    metadata[decode_text(keyword)] = decode_text(text)

            elif chunk_type == b"zTXt":
                keyword, separator, remaining = chunk_data.partition(b"\x00")
                if separator and len(remaining) >= 1:
                    compression_method = remaining[0]
                    compressed_text = remaining[1:]
                    if compression_method == 0:
                        try:
                            metadata[decode_text(keyword)] = decode_text(zlib.decompress(compressed_text))
                        except zlib.error:
                            metadata[f"{decode_text(keyword)}__decompression_error"] = (
                                "Could not decompress zTXt metadata."
                            )

            elif chunk_type == b"iTXt":
                # keyword\0 compression_flag compression_method language_tag\0
                # translated_keyword\0 text
                keyword, separator, remaining = chunk_data.partition(b"\x00")
                if not separator or len(remaining) < 2:
                    continue
                compression_flag = remaining[0]
                compression_method = remaining[1]
                remainder = remaining[2:]
                language_tag, separator, remainder = remainder.partition(b"\x00")
                if not separator:
                    continue
                translated_keyword, separator, text = remainder.partition(b"\x00")
                if not separator:
                    continue
                try:
                    if compression_flag == 1 and compression_method == 0:
                        text = zlib.decompress(text)
                    metadata[decode_text(keyword)] = decode_text(text)
                except zlib.error:
                    metadata[f"{decode_text(keyword)}__decompression_error"] = (
                        "Could not decompress iTXt metadata."
                    )

            elif chunk_type == b"IEND":
                break

    return metadata


def parse_exiftool_dump(path: Path) -> dict[str, Any]:
    """
    Accept the text dump created by:
      exiftool -a -u -G1 -s image.png > metadata_dump.txt
    Windows PowerShell commonly writes this as UTF-16LE with a BOM.
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")

    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        left, value = line.split(":", 1)
        if not left.strip().startswith("["):
            continue
        # Example left field: "[PNG]           Prompt                         "
        group_end = left.find("]")
        if group_end == -1:
            continue
        group = left[1:group_end].strip()
        tag = left[group_end + 1:].strip()
        if not tag:
            continue
        clean_value = value.strip()
        metadata[f"{group}:{tag}"] = clean_value
        metadata.setdefault(tag, clean_value)

    return metadata


def exiftool_path() -> str | None:
    """Return an ExifTool executable if one is available on PATH."""
    for candidate in ("exiftool", "exiftool.exe"):
        discovered = shutil.which(candidate)
        if discovered:
            return discovered
    return None


def parse_with_exiftool(path: Path) -> dict[str, Any]:
    """Ask ExifTool for structured JSON and return the metadata record."""
    executable = exiftool_path()
    if not executable:
        raise FileNotFoundError("ExifTool is not available on PATH.")

    completed = subprocess.run(
        [executable, "-j", "-a", "-u", "-G1", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise MetadataRecoveryError(
            f"ExifTool could not read {path.name}: {completed.stderr.strip() or 'unknown error'}"
        )

    try:
        records = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataRecoveryError(f"ExifTool returned invalid JSON for {path.name}.") from exc

    if not records:
        raise MetadataRecoveryError(f"ExifTool returned no metadata for {path.name}.")
    return records[0]


def extract_metadata(path: Path, allow_exiftool: bool = True) -> tuple[dict[str, Any], str]:
    """
    Read metadata using the least-fragile available approach:
      1) supplied ExifTool text dump
      2) direct native PNG chunk parser
      3) ExifTool (for WEBP/JPEG and fallback cases)
    """
    suffix = path.suffix.casefold()
    if suffix in EXIF_DUMP_EXTENSIONS:
        return parse_exiftool_dump(path), "ExifTool text dump"

    if suffix == ".png":
        try:
            metadata = parse_png_text_chunks(path)
            if find_metadata_value(metadata, "Prompt") or find_metadata_value(metadata, "Workflow"):
                return metadata, "Native PNG text-chunk parser"
            # PNG metadata may be in an unusual location; let ExifTool have a go.
            if not allow_exiftool:
                return metadata, "Native PNG text-chunk parser"
        except MetadataRecoveryError:
            if not allow_exiftool:
                raise

    if allow_exiftool:
        try:
            return parse_with_exiftool(path), "ExifTool"
        except FileNotFoundError:
            pass

    raise MetadataRecoveryError(
        f"Could not find ComfyUI/Krea metadata in {path.name}. "
        "For PNG, confirm it is the original file. For WEBP/JPEG, install ExifTool."
    )


def node_class(node: dict[str, Any]) -> str:
    return str(node.get("class_type") or node.get("type") or "")


def graph_nodes(prompt_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return only node-like records keyed by their Comfy graph IDs."""
    return {
        str(node_id): node
        for node_id, node in prompt_graph.items()
        if isinstance(node, dict) and ("class_type" in node or "type" in node)
    }


def referenced_node_id(value: Any) -> str | None:
    """Comfy links normally appear as ['node_id', output_index]."""
    if isinstance(value, (list, tuple)) and value:
        candidate = value[0]
        if isinstance(candidate, (str, int)):
            return str(candidate)
    return None


def resolve_conditioning_text(
    nodes: dict[str, dict[str, Any]], reference: Any, visited: set[str] | None = None
) -> str | None:
    """
    Follow a conditioning link backward to a CLIPTextEncode node.
    It intentionally knows a few common routing nodes and safely stops on unknowns.
    """
    node_id = referenced_node_id(reference)
    if node_id is None:
        return None
    if visited is None:
        visited = set()
    if node_id in visited:
        return None
    visited.add(node_id)

    node = nodes.get(node_id)
    if not node:
        return None
    klass = node_class(node).casefold()
    inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}

    if "cliptextencode" in klass:
        value = inputs.get("text")
        return value if isinstance(value, str) else None

    if "conditioningzeroout" in klass:
        # The retained source is helpful for diagnostics, but this is intentionally blank.
        return ""

    # Common single-conditioning pass-through nodes.
    for key in ("conditioning", "positive", "negative", "text", "conditioning_to", "conditioning_from"):
        if key in inputs:
            result = resolve_conditioning_text(nodes, inputs[key], visited)
            if result is not None:
                return result
    return None


def extract_requested_aspect_ratio(workflow: dict[str, Any] | None) -> str | None:
    if not isinstance(workflow, dict):
        return None
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")).casefold() == "sdxlaspectratioselector":
            values = node.get("widgets_values")
            if isinstance(values, list) and values and isinstance(values[0], str):
                return values[0]
    return None


def extract_seed_mode(workflow: dict[str, Any] | None, sampler_id: str) -> str | None:
    if not isinstance(workflow, dict):
        return None
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or str(node.get("id")) != str(sampler_id):
            continue
        values = node.get("widgets_values")
        if isinstance(values, list) and len(values) > 1 and isinstance(values[1], str):
            return values[1]
    return None


def extract_loras(nodes: dict[str, dict[str, Any]]) -> list[LoraInfo]:
    found: list[LoraInfo] = []
    for node_id, node in nodes.items():
        klass = node_class(node).casefold()
        if "lora" not in klass:
            continue
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}

        # rgthree Power Lora Loader style: lora_1, lora_2 ... each is a dict.
        for key, value in inputs.items():
            if key.casefold().startswith("lora_") and isinstance(value, dict):
                name = value.get("lora") or value.get("lora_name")
                if name:
                    found.append(
                        LoraInfo(
                            name=str(name),
                            strength=value.get("strength", value.get("strength_model")),
                            strength_clip=value.get("strength_clip", value.get("strengthTwo")),
                            enabled=bool(value.get("on", True)),
                            node_id=node_id,
                        )
                    )

        # Standard Comfy LoraLoader style.
        standard_name = inputs.get("lora_name") or inputs.get("lora")
        if isinstance(standard_name, str) and standard_name:
            found.append(
                LoraInfo(
                    name=standard_name,
                    strength=inputs.get("strength_model", inputs.get("strength")),
                    strength_clip=inputs.get("strength_clip"),
                    enabled=True,
                    node_id=node_id,
                )
            )

    # Preserve order while eliminating accidental duplicate records.
    unique: list[LoraInfo] = []
    seen: set[tuple[Any, ...]] = set()
    for item in found:
        signature = (item.node_id, item.name, str(item.strength), str(item.strength_clip), item.enabled)
        if signature not in seen:
            seen.add(signature)
            unique.append(item)
    return unique


def compute_aspect_ratio(width: int | None, height: int | None) -> str | None:
    if not width or not height:
        return None
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor} ({width / height:.4f}:1)"


def extract_recipe(
    source_path: Path,
    metadata_source: str,
    metadata: dict[str, Any],
    prompt_graph: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
) -> RecoveredRecipe:
    recipe = RecoveredRecipe(source_file=str(source_path), metadata_source=metadata_source)

    width = find_metadata_value(metadata, "ImageWidth")
    height = find_metadata_value(metadata, "ImageHeight")
    try:
        recipe.image_width = int(width) if width is not None else None
        recipe.image_height = int(height) if height is not None else None
    except (ValueError, TypeError):
        recipe.warnings.append("Image dimensions existed but could not be parsed as integers.")
    recipe.saved_aspect_ratio = compute_aspect_ratio(recipe.image_width, recipe.image_height)
    recipe.requested_aspect_ratio = extract_requested_aspect_ratio(workflow)

    if not isinstance(prompt_graph, dict):
        recipe.warnings.append("No machine-readable Comfy Prompt graph was found.")
        return recipe

    nodes = graph_nodes(prompt_graph)
    if not nodes:
        recipe.warnings.append("Prompt JSON was found, but it did not look like a ComfyUI graph.")
        return recipe

    recipe.loras = extract_loras(nodes)

    for node_id, node in nodes.items():
        klass = node_class(node).casefold()
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}

        if klass == "unetloader":
            value = inputs.get("unet_name")
            if isinstance(value, str):
                recipe.model = value
        elif "checkpointloader" in klass and not recipe.model:
            value = inputs.get("ckpt_name")
            if isinstance(value, str):
                recipe.model = value
        elif klass == "cliploader":
            value = inputs.get("clip_name")
            if isinstance(value, str):
                recipe.clip = value
            clip_type = inputs.get("type")
            if isinstance(clip_type, str):
                recipe.clip_type = clip_type
        elif klass == "vaeloader":
            value = inputs.get("vae_name")
            if isinstance(value, str):
                recipe.vae = value

        if "ksampler" in klass:
            sampler = SamplerInfo(
                node_id=node_id,
                class_type=node_class(node),
                seed=inputs.get("seed"),
                steps=inputs.get("steps"),
                cfg=inputs.get("cfg"),
                sampler_name=inputs.get("sampler_name"),
                scheduler=inputs.get("scheduler"),
                denoise=inputs.get("denoise"),
                seed_mode=extract_seed_mode(workflow, node_id),
            )
            recipe.samplers.append(sampler)

            if recipe.positive_prompt is None:
                recipe.positive_prompt = resolve_conditioning_text(nodes, inputs.get("positive"))
            if recipe.negative_prompt is None:
                recipe.negative_prompt = resolve_conditioning_text(nodes, inputs.get("negative"))

    if recipe.positive_prompt is None:
        nonempty_texts = []
        for node in nodes.values():
            if "cliptextencode" in node_class(node).casefold():
                text = node.get("inputs", {}).get("text")
                if isinstance(text, str) and text.strip():
                    nonempty_texts.append(text)
        if nonempty_texts:
            recipe.positive_prompt = nonempty_texts[0]
            recipe.warnings.append(
                "Positive prompt was inferred from a non-empty CLIPTextEncode node rather than traced from KSampler."
            )

    if recipe.negative_prompt == "":
        recipe.negative_prompt = None
        recipe.warnings.append("Negative conditioning was deliberately zeroed/blank in the recovered workflow.")

    if not recipe.samplers:
        recipe.warnings.append("No KSampler node was found. This may be an unusual workflow or a partial export.")

    return recipe


def fixed_seed_workflow(workflow: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Make a workflow copy reproducible by changing known KSampler seed widget modes
    (randomize / increment / decrement) to fixed. The recovered numeric seed stays untouched.
    """
    if not isinstance(workflow, dict):
        return None, ["No workflow JSON was available to patch."]

    patched = copy.deepcopy(workflow)
    changes: list[str] = []
    for node in patched.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type", "")).casefold()
        if "ksampler" not in node_type:
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list) or len(values) < 2:
            continue
        if isinstance(values[1], str) and values[1].casefold() in {"randomize", "increment", "decrement"}:
            old_mode = values[1]
            values[1] = "fixed"
            changes.append(f"KSampler node {node.get('id')}: seed mode {old_mode!r} → 'fixed'.")

    if not changes:
        changes.append("No recognized randomizing KSampler seed widget was found; workflow left unchanged.")
    return patched, changes


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def render_markdown(recipe: RecoveredRecipe, patch_notes: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Recovered generation settings")
    lines.append("")
    lines.append(f"**Source:** `{recipe.source_file}`  ")
    lines.append(f"**Metadata reader:** {recipe.metadata_source}  ")
    lines.append(f"**Recovered:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append("## Core files")
    lines.append("")
    lines.append(f"- **Diffusion model:** `{recipe.model or 'Not found'}`")
    lines.append(f"- **CLIP:** `{recipe.clip or 'Not found'}`" + (f" (`{recipe.clip_type}`)" if recipe.clip_type else ""))
    lines.append(f"- **VAE:** `{recipe.vae or 'Not found'}`")
    lines.append("")
    lines.append("## Resolution")
    lines.append("")
    if recipe.image_width and recipe.image_height:
        lines.append(f"- **Saved image:** {recipe.image_width} × {recipe.image_height}")
    else:
        lines.append("- **Saved image:** Not found")
    lines.append(f"- **Saved pixel ratio:** {recipe.saved_aspect_ratio or 'Not found'}")
    lines.append(f"- **Requested selector ratio:** {recipe.requested_aspect_ratio or 'Not found'}")
    lines.append("")
    lines.append("## Sampling")
    lines.append("")
    if recipe.samplers:
        for index, sampler in enumerate(recipe.samplers, start=1):
            title = "Primary sampler" if index == 1 else f"Sampler {index}"
            lines.append(f"### {title} — node `{sampler.node_id}`")
            lines.append("")
            lines.append(f"- **Seed:** `{sampler.seed if sampler.seed is not None else 'Not found'}`")
            lines.append(f"- **Original seed mode:** `{sampler.seed_mode or 'Unknown'}`")
            lines.append(f"- **Steps:** `{sampler.steps if sampler.steps is not None else 'Not found'}`")
            lines.append(f"- **CFG:** `{sampler.cfg if sampler.cfg is not None else 'Not found'}`")
            lines.append(f"- **Sampler:** `{sampler.sampler_name or 'Not found'}`")
            lines.append(f"- **Scheduler:** `{sampler.scheduler or 'Not found'}`")
            lines.append(f"- **Denoise:** `{sampler.denoise if sampler.denoise is not None else 'Not found'}`")
            lines.append("")
    else:
        lines.append("No KSampler details were recovered.")
        lines.append("")

    lines.append("## LoRAs")
    lines.append("")
    if recipe.loras:
        lines.append("| # | File | Strength | Clip strength | Enabled |")
        lines.append("|---:|---|---:|---:|---|")
        for index, lora in enumerate(recipe.loras, start=1):
            strength = "" if lora.strength is None else str(lora.strength)
            strength_clip = "" if lora.strength_clip is None else str(lora.strength_clip)
            lines.append(
                f"| {index} | `{markdown_escape(lora.name)}` | {strength} | {strength_clip} | {'Yes' if lora.enabled else 'No'} |"
            )
    else:
        lines.append("No LoRA nodes were detected.")
    lines.append("")

    lines.append("## Positive prompt")
    lines.append("")
    if recipe.positive_prompt:
        lines.append("```text")
        lines.append(recipe.positive_prompt.rstrip())
        lines.append("```")
    else:
        lines.append("Not recovered.")
    lines.append("")

    lines.append("## Negative prompt")
    lines.append("")
    if recipe.negative_prompt:
        lines.append("```text")
        lines.append(recipe.negative_prompt.rstrip())
        lines.append("```")
    else:
        lines.append("Blank / zeroed conditioning, or not recovered.")
    lines.append("")

    lines.append("## Reproduction notes")
    lines.append("")
    lines.append("- `fixed_seed_workflow.json` changes recognized KSampler seed modes to `fixed`; it does **not** change the recovered numeric seed.")
    lines.append("- Exact pixels still depend on the same model files, LoRA files, node versions, workflow behavior, and sometimes hardware/software versions.")
    lines.append("- The requested aspect-ratio node and final saved pixel ratio can differ. Preserve the actual saved dimensions when reproducing a favorite image.")
    lines.append("")
    lines.append("## Workflow patch log")
    lines.append("")
    for note in patch_notes:
        lines.append(f"- {note}")
    lines.append("")

    if recipe.warnings:
        lines.append("## Warnings / things to verify")
        lines.append("")
        for warning in recipe.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def export_recovery(
    source_path: Path,
    output_dir: Path,
    allow_exiftool: bool = True,
) -> dict[str, Any]:
    metadata, metadata_source = extract_metadata(source_path, allow_exiftool=allow_exiftool)
    raw_prompt = find_metadata_value(metadata, "Prompt")
    raw_workflow = find_metadata_value(metadata, "Workflow")
    prompt_graph = json_or_none(raw_prompt)
    workflow = json_or_none(raw_workflow)

    if prompt_graph is None and workflow is None:
        raise MetadataRecoveryError(
            "The image was readable, but no JSON Prompt or Workflow field was found. "
            "It may have been exported through a site/app that stripped generation metadata."
        )

    recipe = extract_recipe(source_path, metadata_source, metadata, prompt_graph, workflow)
    patched_workflow, patch_notes = fixed_seed_workflow(workflow)

    output_dir.mkdir(parents=True, exist_ok=True)
    readable_path = output_dir / "readable_settings.md"
    original_workflow_path = output_dir / "original_workflow.json"
    fixed_workflow_path = output_dir / "fixed_seed_workflow.json"
    api_prompt_path = output_dir / "api_prompt.json"
    summary_path = output_dir / "extraction_summary.json"

    readable_path.write_text(render_markdown(recipe, patch_notes), encoding="utf-8")
    if workflow is not None:
        write_json(original_workflow_path, workflow)
    if patched_workflow is not None:
        write_json(fixed_workflow_path, patched_workflow)
    if prompt_graph is not None:
        write_json(api_prompt_path, prompt_graph)

    summary = {
        "recipe": asdict(recipe),
        "patch_notes": patch_notes,
        "exports": {
            "readable_settings": str(readable_path),
            "original_workflow": str(original_workflow_path) if workflow is not None else None,
            "fixed_seed_workflow": str(fixed_workflow_path) if patched_workflow is not None else None,
            "api_prompt": str(api_prompt_path) if prompt_graph is not None else None,
        },
    }
    write_json(summary_path, summary)
    return summary


def collect_sources(path: Path, batch: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise MetadataRecoveryError(f"Path does not exist: {path}")
    if not batch:
        raise MetadataRecoveryError(
            "That is a folder. Re-run with --batch to process all supported files inside it."
        )
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.casefold() in (SUPPORTED_IMAGE_EXTENSIONS | EXIF_DUMP_EXTENSIONS)
    )


def output_name_for(source: Path) -> str:
    # metadata_dump.txt becomes metadata_dump_recovered; image.png becomes image_recovered.
    return f"{source.stem}_recovered"


def write_batch_csv(output_root: Path, summaries: list[dict[str, Any]]) -> Path:
    csv_path = output_root / "batch_summary.csv"
    headers = [
        "source_file",
        "metadata_source",
        "width",
        "height",
        "model",
        "seed",
        "seed_mode",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "denoise",
        "lora_count",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for summary in summaries:
            recipe = summary["recipe"]
            primary = recipe["samplers"][0] if recipe["samplers"] else {}
            writer.writerow(
                {
                    "source_file": recipe["source_file"],
                    "metadata_source": recipe["metadata_source"],
                    "width": recipe["image_width"] or "",
                    "height": recipe["image_height"] or "",
                    "model": recipe["model"] or "",
                    "seed": primary.get("seed", ""),
                    "seed_mode": primary.get("seed_mode", ""),
                    "steps": primary.get("steps", ""),
                    "cfg": primary.get("cfg", ""),
                    "sampler": primary.get("sampler_name", ""),
                    "scheduler": primary.get("scheduler", ""),
                    "denoise": primary.get("denoise", ""),
                    "lora_count": len(recipe["loras"]),
                    "warnings": " | ".join(recipe["warnings"]),
                }
            )
    return csv_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover ComfyUI/Krea generation settings from image metadata."
    )
    parser.add_argument("input", help="An image, ExifTool metadata dump .txt, or a folder with --batch.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output directory. For one input file, defaults to <input_stem>_recovered "
            "beside the input. For --batch, defaults to recovered_metadata beside the folder."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Recursively process every supported image / metadata dump under the input folder.",
    )
    parser.add_argument(
        "--no-exiftool",
        action="store_true",
        help="Do not call ExifTool; PNG direct parsing and metadata-dump text files still work.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    allow_exiftool = not args.no_exiftool

    try:
        sources = collect_sources(input_path, args.batch)
        if not sources:
            raise MetadataRecoveryError("No supported files were found.")

        if args.output:
            output_root = Path(args.output).expanduser().resolve()
        elif args.batch:
            output_root = input_path / "recovered_metadata"
        else:
            output_root = input_path.parent / output_name_for(input_path)

        summaries: list[dict[str, Any]] = []
        failures: list[tuple[Path, str]] = []

        for source in sources:
            output_dir = output_root / output_name_for(source) if args.batch else output_root
            try:
                summary = export_recovery(source, output_dir, allow_exiftool=allow_exiftool)
                summaries.append(summary)
                print(f"✓ {source.name}")
                print(f"  Exported: {output_dir}")
            except Exception as exc:  # report all failures in batch mode
                failures.append((source, str(exc)))
                print(f"✗ {source.name}: {exc}", file=sys.stderr)

        if args.batch and summaries:
            csv_path = write_batch_csv(output_root, summaries)
            print(f"\nBatch CSV: {csv_path}")

        print(f"\nRecovered {len(summaries)} of {len(sources)} item(s).")
        if failures:
            print("\nFailures:", file=sys.stderr)
            for source, error in failures:
                print(f"- {source.name}: {error}", file=sys.stderr)
            return 1
        return 0

    except MetadataRecoveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
