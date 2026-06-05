"""Evaluacion downstream con checksum-constrained inference.

Compara argmax (baseline ResNet-18) vs argmax + checksum solver sobre las
mismas actas. Reporta:

  - digit-level / field-level / acta-level accuracy antes vs despues
  - reconstruccion exacta del total agregado
  - % de actas que el solver corrige vs flagea (infeasible)
  - distribucion de n_changed por acta

Uso:
  python scripts/evaluate_with_solver.py --split val --K 10 --tolerance 0
  python scripts/evaluate_with_solver.py --split val --K 5 --tolerance 2

Requiere: data/eval_logits_<split>.parquet  (generar con evaluate.py --save-logits)
          data/evaluate_<split>.csv         (generar con evaluate.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.checksum_solver import (
    SUM_FIELDS,
    TOTAL_FIELD,
    build_candidates_from_logits,
    solve_acta,
)


def _load_field_specs() -> dict[str, int]:
    template = json.loads((ROOT / "templates.json").read_text())["presidencial"]
    return {f["name"]: f["n_digits"] for f in template["fields"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tolerance", type=int, default=0)
    ap.add_argument("--n-actas", type=int, default=None,
                    help="solo procesar primeras N actas (debug)")
    ap.add_argument("--time-limit-s", type=float, default=5.0,
                    help="timeout del solver por acta (default 5s)")
    args = ap.parse_args()

    logits_path = ROOT / f"data/eval_logits_{args.split}.parquet"
    eval_csv = ROOT / f"data/evaluate_{args.split}.csv"

    if not logits_path.exists():
        sys.exit(f"falta {logits_path}; corre: python scripts/evaluate.py "
                 f"--split {args.split} --save-logits")
    if not eval_csv.exists():
        sys.exit(f"falta {eval_csv}; corre: python scripts/evaluate.py "
                 f"--split {args.split}")

    df_logits = pd.read_parquet(logits_path)
    df_eval = pd.read_csv(eval_csv)
    field_specs = _load_field_specs()

    actas = sorted(df_eval["archivoId"].unique())
    if args.n_actas:
        actas = actas[: args.n_actas]
    n_total = len(actas)
    print(f"split={args.split}  K={args.K}  tolerance={args.tolerance}  "
          f"n_actas={n_total}")
    print(f"solver time limit: {args.time_limit_s}s por acta\n")

    rows: list[dict] = []
    n_infeasible = 0
    n_changed = 0
    t0 = time.perf_counter()

    for i, aid in enumerate(actas):
        df_l = df_logits[df_logits["archivoId"] == aid]
        df_e = df_eval[df_eval["archivoId"] == aid]
        if len(df_e) == 0:
            continue

        # Baseline argmax: valores ya estan en df_eval['pred']
        baseline = dict(zip(df_e["field"], df_e["pred"]))
        real = dict(zip(df_e["field"], df_e["real"]))

        # Solver
        cands = build_candidates_from_logits(df_l, field_specs, K=args.K)
        res = solve_acta(cands, tolerance=args.tolerance,
                         time_limit_s=args.time_limit_s)

        if res is None:
            # fallback a argmax con flag
            solver_pred = baseline
            status = "infeasible"
            changed = 0
            n_infeasible += 1
        else:
            solver_pred = res.chosen
            status = res.status
            changed = res.n_changed
            if changed > 0:
                n_changed += 1

        # Comparar baseline vs solver vs real por field
        for field in field_specs:
            row = {
                "archivoId": aid,
                "field": field,
                "real": real.get(field, 0),
                "baseline": baseline.get(field, 0),
                "solver": solver_pred.get(field, 0),
                "status": status,
            }
            row["baseline_correct"] = row["baseline"] == row["real"]
            row["solver_correct"] = row["solver"] == row["real"]
            rows.append(row)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"  {i + 1}/{n_total}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    out = pd.DataFrame(rows)
    out_path = ROOT / (f"data/eval_with_solver_{args.split}"
                       f"_K{args.K}_tol{args.tolerance}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nresultados -> {out_path}  ({len(out)} filas)")

    # -------- Reporte comparativo --------
    actas_eval = out["archivoId"].nunique()
    print(f"\nactas evaluadas: {actas_eval}")
    print(f"infeasible (solver no convergio): {n_infeasible} "
          f"({100 * n_infeasible / actas_eval:.2f}%)")
    print(f"actas con cambios del solver: {n_changed} "
          f"({100 * n_changed / actas_eval:.2f}%)")

    # Field-level
    fl_base = out["baseline_correct"].mean()
    fl_solv = out["solver_correct"].mean()
    print(f"\nfield-level accuracy:")
    print(f"  baseline (argmax):   {fl_base:.4f}")
    print(f"  solver:              {fl_solv:.4f}  "
          f"({'+' if fl_solv >= fl_base else ''}{100*(fl_solv-fl_base):.2f}pp)")

    # Acta-level
    al_base = out.groupby("archivoId")["baseline_correct"].all().mean()
    al_solv = out.groupby("archivoId")["solver_correct"].all().mean()
    print(f"\nacta-level accuracy (todos los 42 fields correctos):")
    print(f"  baseline (argmax):   {al_base:.4f}")
    print(f"  solver:              {al_solv:.4f}  "
          f"({'+' if al_solv >= al_base else ''}{100*(al_solv-al_base):.2f}pp)")

    # Reconstruccion exacta del total agregado
    sum_real = (out[out["field"] != TOTAL_FIELD]
                .groupby("archivoId")["real"].sum())
    sum_base = (out[out["field"] != TOTAL_FIELD]
                .groupby("archivoId")["baseline"].sum())
    sum_solv = (out[out["field"] != TOTAL_FIELD]
                .groupby("archivoId")["solver"].sum())
    err_base = (sum_base - sum_real)
    err_solv = (sum_solv - sum_real)
    print(f"\nreconstruccion exacta del total agregado:")
    print(f"  baseline:  {(err_base == 0).sum()}/{len(err_base)} "
          f"({100 * (err_base == 0).mean():.2f}%)  MAE={err_base.abs().mean():.2f}")
    print(f"  solver:    {(err_solv == 0).sum()}/{len(err_solv)} "
          f"({100 * (err_solv == 0).mean():.2f}%)  MAE={err_solv.abs().mean():.2f}")

    # Acta-level por status (infeasible vs converged vs no_change)
    print("\nacta-level por status del solver:")
    actas_status = out.drop_duplicates("archivoId")[["archivoId", "status"]]
    solver_correct_by_acta = (out.groupby("archivoId")["solver_correct"].all()
                              .reset_index().rename(
                                  columns={"solver_correct": "all_ok"}))
    merged = actas_status.merge(solver_correct_by_acta, on="archivoId")
    by_status = merged.groupby("status").agg(
        n=("archivoId", "size"), acc=("all_ok", "mean"))
    for st, row in by_status.iterrows():
        print(f"  {st:14s}  n={int(row['n']):4d}  acta_acc={row['acc']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"\ntiempo total solver: {elapsed:.1f}s "
          f"({elapsed / actas_eval * 1000:.0f}ms por acta)")


if __name__ == "__main__":
    main()
