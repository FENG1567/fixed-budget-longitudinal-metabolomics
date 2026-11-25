#!/usr/bin/env python3
"""Run the article's analyses into an output directory in independent stages."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
STAGES = ("display", "cost", "simulation", "empirical", "interleaved",
          "profiles", "long-visit", "zenodo", "multi-aux")


def script(name, *args, environment=None):
    command = [sys.executable, str(TOOLS / name), *map(str, args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    parser.add_argument("--data", type=Path, default=ROOT / "data",
                        help="Aggregate CSV and figure metadata directory for the display stage")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.inputs.resolve()
    output = (args.out or ROOT / "build" / args.stage).resolve()
    env = dict(os.environ, OMICS_REPLAY_INPUTS=str(source / "replay"),
               OMICS_REPLAY_OUTPUT=str(output), MPLBACKEND="Agg")
    if output.is_relative_to(ROOT / "data") or output.is_relative_to(source):
        parser.error("Choose an output directory outside the input and frozen data directories.")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    match args.stage:
        case "display":
            script("build_display_inputs.py", "--data", args.data.resolve(), "--out", output)
            script("figures.py", output, "--input", output / "figure_inputs.json", environment=env)
        case "cost":
            script("cost_sensitivity.py", "--out", output)
            script("price_sensitivity.py", "--out", output)
        case "simulation":
            script("stress_operational_policy.py", "--out", output)
        case "empirical":
            script("policy_reevaluation.py", output, environment=env)
            script("pomet_feature_sensitivity.py", output / "pomet_feature_sensitivity.csv", environment=env)
        case "interleaved":
            script("frozen_benchmark.py", environment=env)
        case "profiles":
            script("reanalyse_profiles.py", "--root", source, "--outdir", output)
        case "long-visit":
            script("long_visit_replay.py", source / "01_data/external/st001251/data.txt",
                   output / "long_visit_replay.csv", environment=env)
        case "zenodo":
            script("zenodo_meal_replay.py", "--data", source / "zenodo", "--output",
                   output / "zenodo_lunch_replay.csv")
        case "multi-aux":
            script("multi_aux_extra.py", "--parsed", source / "01_data/parsed", "--out", output)


if __name__ == "__main__":
    main()
