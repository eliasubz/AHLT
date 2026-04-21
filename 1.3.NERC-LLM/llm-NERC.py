#!/usr/bin/env python3
"""
Assignment-style entry point: ``python3 llm-NERC.py <input.xml> <result.out>``.

Uses environment variables for the frozen configuration (see ``bin/FROZEN_CONFIG.env.example``).

- ``LLM_NERC_STRATEGY``: ``fewshot`` (default) or ``finetune``.
- Few-shot: ``LLM_NERC_MODEL``, ``LLM_NERC_PROMPTS``, ``LLM_NERC_FEW_SHOT``, ``LLM_NERC_FS_TRAIN``,
  ``LLM_NERC_QUANT`` (``1`` for ``-quant``, else ``-ollama``).
- Fine-tune: same model/prompts/quant, plus ``LLM_NERC_FT_WEIGHTS`` (adapter directory **basename** under ``models/``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _repo_paths():
    here = os.path.abspath(os.path.dirname(__file__))
    bin_dir = os.path.join(here, "bin")
    sys.path.insert(0, bin_dir)
    import paths  # noqa: E402

    return here, bin_dir, paths


def _split_stem(xml_path: str) -> str:
    base = os.path.basename(xml_path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".xml":
        print(f"Expected .xml input, got: {xml_path}", file=sys.stderr)
        sys.exit(1)
    return stem


def _resolve_input_xml(paths_mod, arg: str) -> str:
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    cand = os.path.join(paths_mod.DATA, arg)
    if os.path.isfile(cand):
        return cand
    print(f"Input XML not found: {arg} (also tried {cand})", file=sys.stderr)
    sys.exit(1)


def _quant_flag() -> str:
    q = os.environ.get("LLM_NERC_QUANT", "1").strip().lower()
    if q in ("1", "true", "yes", "y", "quant", "-quant"):
        return "-quant"
    return "-ollama"


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.xml> <result.out>", file=sys.stderr)
        sys.exit(1)

    _, bin_dir, paths_mod = _repo_paths()
    in_xml = _resolve_input_xml(paths_mod, sys.argv[1])
    out_path = os.path.abspath(sys.argv[2])
    test_stem = _split_stem(in_xml)

    strategy = os.environ.get("LLM_NERC_STRATEGY", "fewshot").strip().lower()
    model = os.environ.get("LLM_NERC_MODEL", "").strip()
    if not model:
        print("Set LLM_NERC_MODEL to a short name, HF id, or local path.", file=sys.stderr)
        sys.exit(1)

    prompts = os.environ.get(
        "LLM_NERC_PROMPTS",
        os.path.join(bin_dir, "prompts01.json"),
    )
    prompts = os.path.abspath(os.path.expanduser(prompts))
    if not os.path.isfile(prompts):
        print(f"Prompt file not found: {prompts}", file=sys.stderr)
        sys.exit(1)

    slug = paths_mod.model_slug(model)
    quant = _quant_flag()
    os.makedirs(paths_mod.RESULTS, exist_ok=True)

    k = ""
    if strategy == "fewshot":
        k = os.environ.get("LLM_NERC_FEW_SHOT", "3").strip()
        train_stem = os.environ.get("LLM_NERC_FS_TRAIN", "train").strip()
        cmd = [
            sys.executable,
            os.path.join(bin_dir, "fewshot.py"),
            model,
            prompts,
            k,
            train_stem,
            test_stem,
            quant,
        ]
    elif strategy == "finetune":
        w = os.environ.get("LLM_NERC_FT_WEIGHTS", "").strip()
        if not w:
            print("Set LLM_NERC_FT_WEIGHTS to the adapter basename (e.g. FT-llama32B3-quant.weights).", file=sys.stderr)
            sys.exit(1)
        ft_extra = ["-quant"] if quant == "-quant" else []
        cmd = [
            sys.executable,
            os.path.join(bin_dir, "finetune-inference.py"),
            model,
            prompts,
            test_stem,
            w,
            *ft_extra,
        ]
    else:
        print(f"Unknown LLM_NERC_STRATEGY={strategy!r}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    # Prefer a scratch HF cache on Slurm if the user did not set HF_HOME.
    if "HF_HOME" not in env and "SLURM_TMPDIR" in env:
        env["HF_HOME"] = os.path.join(env["SLURM_TMPDIR"], "hf_cache")

    print("Running:", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=bin_dir, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)

    quant_suffix = "-quant" if quant == "-quant" else ""
    if strategy == "fewshot":
        produced = os.path.join(paths_mod.RESULTS, f"FS-{slug}-{k}-{test_stem}{quant_suffix}.out")
    else:
        produced = os.path.join(paths_mod.RESULTS, f"FT-{slug}{quant_suffix}-{test_stem}.out")

    if not os.path.isfile(produced):
        print(f"Expected output missing: {produced}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copyfile(produced, out_path)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
