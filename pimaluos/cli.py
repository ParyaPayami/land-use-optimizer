"""Command-line interface: ``pimaluos fetch-context | run | report | rag-benchmark``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pimaluos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-context", help="download the public context data (network, destinations, jobs, ...)")
    f.add_argument("--out", default="data/raw/context")
    f.add_argument("--force", action="store_true")
    r = sub.add_parser("run", help="run all experiments")
    r.add_argument("--config", required=True)
    r.add_argument("--pluto", help="local MapPLUTO file (.zip/.shp/.gdb/.gpkg/.geojson/.parquet)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report", help="make figures, LaTeX tables and macros from a results directory")
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="paper/generated")
    g = sub.add_parser("rag-benchmark", help="experimental: evaluate LLM-RAG FAR extraction against MapPLUTO")
    g.add_argument("--pluto", required=True)
    g.add_argument("--zr-dir", required=True, help="directory with Zoning Resolution text/PDF")
    g.add_argument("--provider", default="openai", choices=["openai", "anthropic", "ollama"])
    g.add_argument("--model")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.cmd == "fetch-context":
        from pimaluos.context.fetch import fetch_all

        fetch_all(a.out, force=a.force)
    elif a.cmd == "run":
        from pimaluos.experiments import load_config, run_all

        run_all(load_config(a.config), Path(a.out), a.pluto)
    elif a.cmd == "report":
        from pimaluos.reporting import make_report

        make_report(Path(a.results), Path(a.out))
    elif a.cmd == "rag-benchmark":
        from pimaluos.rag_benchmark import run_rag_benchmark

        run_rag_benchmark(a.pluto, a.zr_dir, a.provider, a.model, Path(a.out))


if __name__ == "__main__":
    main()
