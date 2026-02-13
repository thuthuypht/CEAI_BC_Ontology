from __future__ import annotations

import argparse
from pathlib import Path

from src.io_rdf import load_graph_ttl, export_triples_tsv, RdfExportConfig
from src.utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_ttl", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    g = load_graph_ttl(args.input_ttl)
    ensure_dir(str(Path(args.out_tsv).parent))

    cfg = RdfExportConfig(keep_rdf_type=True, drop_literal_objects=True, keep_uri_objects_only=True)
    df = export_triples_tsv(g, args.out_tsv, cfg=cfg)
    print(f"✅ Exported {len(df)} triples to {args.out_tsv}")


if __name__ == "__main__":
    main()
