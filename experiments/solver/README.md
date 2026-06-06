# experiments/solver/

Experimento de **post-procesamiento**: corrige las predicciones de campo del
modelo usando una restriccion (probablemente el total reportado del acta), para
mejorar la reconstruccion agregada.

## Estado: codigo a recuperar

Hay artefactos de resultados en `data/` (gitignored) del 2026-05-27:

- `data/eval_with_solver_val_K*_tol*.csv` — barrido de top-K x tolerancia,
  columnas `baseline` vs `solver` por campo.
- `data/solver_results_val_K10_tol0.parquet`, `data/solver_comparison.csv`.
- `data/eval_logits_val.parquet` — logits por celda (insumo del solver).

**El script/notebook que los genera NO esta en el repo** (ningun `.py`/`.ipynb`
menciona `solver`). Antes de citar este experimento en el informe hay que
recuperar o reimplementar el generador a partir de los logits + la restriccion
del total. Mientras tanto, las metricas oficiales son las del modelo sin solver
(`scripts/evaluate.py`).
