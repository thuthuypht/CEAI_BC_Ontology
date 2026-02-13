# Convenience runner
# Example:
#   python run_pipeline.py --ttl data/processed/Wisconsin_Categorized_KnowledgeGraph_enriched_final.ttl --out outputs/wdbc

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient_prefix", default="Patient_")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "scripts/03_train_pykeen_and_cluster.py",
        "--input_ttl", args.ttl,
        "--out_dir", args.out,
        "--patient_prefix", args.patient_prefix,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
