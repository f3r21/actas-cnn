"""Checksum-constrained inference para actas electorales.

Toma los log_softmax por crop (salida de evaluate.py --save-logits), genera
top-K candidatos por field, y resuelve un ILP que maximiza la log-prob
conjunta sujeto a la invariante de dominio:

    sum(partido_01..38 + votos_blanco + votos_nulos + votos_impugnados)
    == total_ciudadanos    (con tolerancia opcional)

Implementacion con PuLP (CBC backend, gratis y battle-tested). Tiempo
tipico: <100ms por acta para K<=10.

Uso programatico:
    from scripts.checksum_solver import solve_acta, generate_candidates

Uso CLI:
    python scripts/checksum_solver.py --split val --K 10 --tolerance 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SolverStatus = Literal["converged", "no_change", "infeasible", "low_conf"]

SUM_FIELDS = (
    [f"partido_{i:02d}" for i in range(1, 39)]
    + ["votos_blanco", "votos_nulos", "votos_impugnados"]
)
TOTAL_FIELD = "total_ciudadanos"


@dataclass(frozen=True)
class Candidate:
    value: int
    log_prob: float


@dataclass(frozen=True)
class SolverResult:
    archivo_id: str
    chosen: dict[str, int]
    status: SolverStatus
    n_changed: int
    objective: float


def _digits_to_value(digits: list[int]) -> int:
    """Concatena digitos right-justified como entero. Reusa la convencion
    de scripts/evaluate.reconstruct_value."""
    return int("".join(str(d) for d in digits))


def generate_candidates(
    crops: list[tuple[int, np.ndarray]],
    n_cells: int,
    K: int = 10,
) -> list[Candidate]:
    """Genera top-K candidatos para un field.

    crops: lista de (pos, log_probs[10]). Las posiciones ausentes son
        leading zeros implicitos (convencion right-justified ONPE).
    n_cells: numero total de cells del field (3 para partidos/votos,
        4 para total_ciudadanos).
    K: cantidad maxima de candidatos a retornar.

    Returns: lista ordenada por log_prob desc, deduplicada por value
        (si dos combos de digitos producen el mismo entero, mantiene el
        de mayor log_prob).
    """
    if not crops:
        # Ningun crop -> field tiene valor 0 con certeza
        return [Candidate(value=0, log_prob=0.0)]

    positions = [pos for pos, _ in crops]
    log_probs_per_pos = [lp for _, lp in crops]
    m = len(positions)

    best_by_value: dict[int, float] = {}
    for combo in itertools.product(range(10), repeat=m):
        digits = [0] * n_cells
        for i, pos in enumerate(positions):
            digits[pos] = combo[i]
        value = _digits_to_value(digits)
        log_prob = float(sum(log_probs_per_pos[i][combo[i]] for i in range(m)))
        if value not in best_by_value or log_prob > best_by_value[value]:
            best_by_value[value] = log_prob

    candidates = [Candidate(value=v, log_prob=lp) for v, lp in best_by_value.items()]
    candidates.sort(key=lambda c: -c.log_prob)
    return candidates[:K]


def solve_acta(
    candidates_per_field: dict[str, list[Candidate]],
    tolerance: int = 0,
    time_limit_s: float = 5.0,
) -> SolverResult | None:
    """Resuelve el ILP para 1 acta.

    candidates_per_field: dict {field_name: [Candidate, ...]} con TODOS
        los 42 fields del template.
    tolerance: |sum_pred - total_pred| permitido. 0 = estricto.
    time_limit_s: timeout del solver CBC.

    Returns: SolverResult con la asignacion optima, o None si infactible.
    El campo status puede ser:
      - 'converged': el solver eligio asignacion que cumple constraint
        Y cambio al menos 1 field respecto al argmax (top-1).
      - 'no_change': el solver eligio el top-1 por field (es decir,
        argmax ya satisfacia el constraint).
      - 'infeasible': no hay asignacion factible con la tolerancia dada.
    """
    # Verificar que tenemos todos los fields necesarios
    required = set(SUM_FIELDS + [TOTAL_FIELD])
    missing = required - set(candidates_per_field.keys())
    if missing:
        raise ValueError(f"faltan fields en input: {missing}")

    prob = LpProblem("checksum", LpMaximize)
    x: dict[tuple[str, int], LpVariable] = {}
    for field, cands in candidates_per_field.items():
        for k in range(len(cands)):
            x[(field, k)] = LpVariable(f"x_{field}_{k}", cat=LpBinary)

    # Cada field elige exactamente 1 candidato
    for field, cands in candidates_per_field.items():
        prob += lpSum(x[(field, k)] for k in range(len(cands))) == 1, f"one_{field}"

    # Restriccion de suma con tolerancia
    sum_expr = lpSum(
        candidates_per_field[f][k].value * x[(f, k)]
        for f in SUM_FIELDS
        for k in range(len(candidates_per_field[f]))
    )
    total_expr = lpSum(
        candidates_per_field[TOTAL_FIELD][k].value * x[(TOTAL_FIELD, k)]
        for k in range(len(candidates_per_field[TOTAL_FIELD]))
    )

    if tolerance == 0:
        prob += sum_expr == total_expr, "checksum"
    else:
        prob += sum_expr - total_expr <= tolerance, "checksum_upper"
        prob += sum_expr - total_expr >= -tolerance, "checksum_lower"

    # Objetivo: maximizar log-prob conjunta
    prob += lpSum(
        candidates_per_field[f][k].log_prob * x[(f, k)]
        for f in candidates_per_field
        for k in range(len(candidates_per_field[f]))
    )

    solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    status_code = prob.solve(solver)
    status_name = LpStatus[status_code]

    if status_name != "Optimal":
        return None

    chosen: dict[str, int] = {}
    n_changed = 0
    objective = 0.0
    for field, cands in candidates_per_field.items():
        for k, cand in enumerate(cands):
            if x[(field, k)].value() is not None and x[(field, k)].value() > 0.5:
                chosen[field] = cand.value
                objective += cand.log_prob
                if k != 0:
                    n_changed += 1
                break

    return SolverResult(
        archivo_id="",
        chosen=chosen,
        status="converged" if n_changed > 0 else "no_change",
        n_changed=n_changed,
        objective=objective,
    )


def build_candidates_from_logits(
    df_acta: pd.DataFrame,
    field_specs: dict[str, int],
    K: int = 10,
) -> dict[str, list[Candidate]]:
    """Toma los logits de UNA acta (filas con columnas archivoId, field,
    pos, lp_0..lp_9) y construye el dict de candidatos por field, para
    los 42 fields del template (los ausentes -> Candidate(0, 0.0))."""
    lp_cols = [f"lp_{c}" for c in range(10)]
    candidates_per_field: dict[str, list[Candidate]] = {}
    for field, n_cells in field_specs.items():
        rows = df_acta[df_acta["field"] == field]
        crops: list[tuple[int, np.ndarray]] = []
        for _, row in rows.iterrows():
            pos = int(row["pos"])
            lp = row[lp_cols].to_numpy(dtype=np.float64)
            crops.append((pos, lp))
        candidates_per_field[field] = generate_candidates(crops, n_cells, K=K)
    return candidates_per_field


def _load_field_specs() -> dict[str, int]:
    template = json.loads((ROOT / "templates.json").read_text())["presidencial"]
    return {f["name"]: f["n_digits"] for f in template["fields"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--K", type=int, default=10,
                    help="cantidad de candidatos por field (default 10)")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="|sum - total| permitido en votos (default 0)")
    ap.add_argument("--n-actas", type=int, default=None,
                    help="solo resolver las primeras N actas (smoke test)")
    args = ap.parse_args()

    logits_path = ROOT / f"data/eval_logits_{args.split}.parquet"
    if not logits_path.exists():
        sys.exit(f"falta {logits_path}; corre primero: "
                 f"python scripts/evaluate.py --split {args.split} --save-logits")

    df = pd.read_parquet(logits_path)
    field_specs = _load_field_specs()
    actas = sorted(df["archivoId"].unique())
    if args.n_actas:
        actas = actas[: args.n_actas]
    print(f"resolviendo {len(actas)} actas (K={args.K}, tol={args.tolerance})...")

    results: list[dict] = []
    for i, aid in enumerate(actas):
        df_acta = df[df["archivoId"] == aid]
        cands = build_candidates_from_logits(df_acta, field_specs, K=args.K)
        res = solve_acta(cands, tolerance=args.tolerance)
        if res is None:
            results.append({"archivoId": aid, "status": "infeasible", "n_changed": 0})
        else:
            row = {"archivoId": aid, "status": res.status, "n_changed": res.n_changed}
            for field, val in res.chosen.items():
                row[f"solver_{field}"] = val
            results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(actas)}")

    out = pd.DataFrame(results)
    out_path = ROOT / f"data/solver_results_{args.split}_K{args.K}_tol{args.tolerance}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nresultados -> {out_path}")
    print(f"status: {out['status'].value_counts().to_dict()}")
    print(f"actas con n_changed > 0: "
          f"{(out['n_changed'] > 0).sum()}/{len(out)} "
          f"({100 * (out['n_changed'] > 0).mean():.1f}%)")


if __name__ == "__main__":
    main()
