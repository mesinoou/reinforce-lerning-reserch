#!/usr/bin/env python3
"""Train a fresh Galaxian target using an existing, frozen source critic."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from advantage_transfer import read_source_checkpoint, source_argument_defaults
from ocatari_source_ppo import parse_args as parse_training_args, train


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--source-checkpoint", default="")
    probe.add_argument("--resume", default="")
    probe.add_argument("--inspect-source", action="store_true")
    known, _ = probe.parse_known_args(argv)
    if "--help" in argv or "-h" in argv:
        print("Target Advantage transfer. Extra option: --inspect-source (print inherited settings and exit).")
        return parse_training_args(argv, defaults={"env": "ALE/Galaxian-v5", "transfer_alpha": 0.5})
    source_path = known.source_checkpoint
    if known.resume:
        target = torch.load(Path(known.resume), map_location="cpu", weights_only=False)
        if not target.get("configuration", {}).get("source_checkpoint_sha256"):
            probe.error("--resume must be a target transfer checkpoint, not a source model/checkpoint")
        source_path = source_path or target["configuration"].get("source_checkpoint", "")
    if not source_path:
        probe.error("--source-checkpoint is required (including alpha=0 for shared source settings)")
    source = read_source_checkpoint(Path(source_path))
    defaults = source_argument_defaults(source)
    defaults.update({
        "env": "ALE/Galaxian-v5",
        "transfer_alpha": 0.5,
        "source_checkpoint": source["path"],
    })
    argv = [arg for arg in argv if arg != "--inspect-source"]
    args = parse_training_args(argv, defaults=defaults)
    if not args.output_dir:
        args.output_dir = (
            f"target_transfer_results_{datetime.now():%Y%m%d_%H%M%S}"
            f"_alpha{args.transfer_alpha:g}_seed{args.seed}"
        )
    if known.inspect_source:
        print(json.dumps({
            "source": source["path"],
            "sha256": source["sha256"],
            "source_env": source["payload"].get("env_id"),
            "source_steps": source["payload"].get("steps", source["payload"].get("environment_steps")),
            "input_mode": source["input_mode"],
            "inherited_defaults": source_argument_defaults(source),
            "requested_target_arguments": vars(args),
            "note": "Metadata inspection only; no environment or training started.",
        }, ensure_ascii=False, indent=2))
        return None
    return args


def main():
    args = parse_args()
    if args is not None:
        train(args)


if __name__ == "__main__":
    main()
