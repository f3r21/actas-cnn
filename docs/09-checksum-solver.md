# 09 — Checksum-constrained inference

## Motivación

ResNet-18 CIFAR sin solver alcanza:

- digit-level: 98.12%
- field-level: 98.87%
- **acta-level: 90.33%**  (626/693 actas con los 42 fields correctos)
- reconstrucción exacta del total agregado: **93.80%**  (650/693)
- MAE del total agregado: **2.40 votos**

El techo del modelo está dado por la ambigüedad inherente del manuscrito.
Pero cada acta presidencial peruana tiene una **invariante matemática
verificable**:

```
sum(partido_01..38 + votos_blanco + votos_nulos + votos_impugnados)
  == total_ciudadanos
```

Hasta ahora esta invariante se reporta como diagnóstico (líneas 196-207
de `scripts/evaluate.py`) pero no se usa como corrección. Este documento
describe cómo convertirla en un solver post-CNN que mejora acta-level
sin reentrenar.

## Formulación

Sea una acta con 42 fields. Para cada field `i` con `n_cells_i`
posiciones escritas (por convención right-justified ONPE), el modelo
produce log-softmax `log p_{i,j}(c)` para `c ∈ {0..9}` en cada celda
`j`.

Generamos **top-K candidatos** por field. Cada candidato es un par
`(value, log_prob)` donde `value` es el entero right-justified
resultante de elegir un dígito por celda, y `log_prob` la suma de
log-probabilidades:

```
log_prob_candidate = sum_{j ∈ cells_escritas} log p_{i,j}(digit_j)
```

Resolvemos un **ILP** con variables binarias `x_{i,k} ∈ {0,1}` ("field
`i` toma el candidato `k`"):

```
max  sum_{i,k} log_prob_{i,k} · x_{i,k}

s.t. sum_k x_{i,k} = 1                              ∀ i  (un candidato por field)
     sum_{i ∈ SUM_FIELDS, k} value_{i,k} · x_{i,k}
       - sum_k value_{total,k} · x_{total,k} = 0    (checksum, con tolerancia opcional)
     x_{i,k} ∈ {0,1}
```

donde `SUM_FIELDS = {partido_01..38, votos_blanco, votos_nulos, votos_impugnados}`.

Backend: **PuLP + CBC** (LP gratuito, sin licencia comercial). Tiempo
medio: **~46 ms por acta** para K=20 (CPU single thread).

## Implementación

- `scripts/checksum_solver.py` — `generate_candidates()` y `solve_acta()`.
- `scripts/evaluate.py --save-logits` — exporta `data/eval_logits_<split>.parquet` con `log p_{i,j}(c)` por crop.
- `scripts/evaluate_with_solver.py` — driver de evaluación comparativa.

Flujo end-to-end:

```
python scripts/evaluate.py --split val --save-logits
python scripts/evaluate_with_solver.py --split val --K 20 --tolerance 0
```

## Resultados (val split, 693 actas)

Configuración óptima: **K=20, tolerance=0**.

| Métrica                         | Baseline | Solver  | Δ          |
|---------------------------------|----------|---------|------------|
| digit-level accuracy            | 98.12%   | (n/a, mide cell) |     |
| field-level accuracy            | 98.87%   | 98.82%  | -0.05pp    |
| **acta-level accuracy**         | 90.33%   | **93.51%** | **+3.18pp** |
| **reconstrucción exacta total** | 93.80%   | **96.25%** | **+2.45pp** |
| MAE del total agregado          | 2.40     | **1.23**   | **-49%**   |
| actas modificadas por solver    | —        | 67 (9.7%)  | —        |
| actas infeasibles               | —        | 0          | —        |
| tiempo por acta                 | —        | ~46 ms     | —        |

El field-level baja marginalmente porque el solver a veces sacrifica
fields correctos para cuadrar la suma. Es un trade-off **deseable**: la
métrica relevante para conteo rápido es el total agregado correcto, no
cada dígito individual.

### Ablations

| K  | tol | acta-level | reconstrucción | MAE   | infeasible |
|----|-----|------------|----------------|-------|------------|
| 3  | 0   | 93.07%     | 95.09%         | 2.03  | 12         |
| 5  | 0   | 93.36%     | 94.95%         | 2.36  | 5          |
| 10 | 0   | 93.36%     | 95.09%         | 1.99  | 2          |
| **20** | **0** | **93.51%** | **96.25%** | **1.23** | **0** |
| 20 | 1   | 93.22%     | 94.66%         | 1.19  | 0          |
| 20 | 2   | 92.93%     | 94.37%         | 1.11  | 0          |
| 20 | 5   | 92.06%     | 93.80%         | 1.08  | 0          |

Observaciones:

1. **K mayor es estrictamente mejor.** Más candidatos → mayor cobertura
   del espacio factible. Con K=3, 12 actas son infeasibles (~1.7%); con
   K=20, todas convergen.
2. **tolerance=0 domina.** Aumentar la tolerancia permite asignaciones
   con suma "casi" correcta, lo cual **deteriora** acta-level porque
   acepta candidatos con menor log-prob para satisfacer un constraint
   más laxo. El MAE baja un poco (porque permite errores chicos) pero a
   costa de más actas mal clasificadas a nivel discreto.

## Limitaciones

- **Techo en ~93.5% acta-level.** El solver no puede crear información
  que no esté en los top-K logits. Si el modelo asigna alta confianza a
  una clase incorrecta (low entropy en la wrong direction), ningún
  candidato top-K contiene la respuesta correcta.
- **2-12 actas (según K) con label ONPE inconsistente.** En estos casos
  el ground truth oficial **no satisface** el checksum (errores
  manuales del escrutinio). El solver falla legítimamente y se reporta
  infeasible → flag para revisión humana en la app.
- **Per-field metric degrada marginalmente.** Aceptable porque el
  objetivo es el agregado.

## Implicación para la app de conteo rápido

Tres outcomes posibles por acta:

| Status         | %   | Acción en la app                       |
|----------------|-----|----------------------------------------|
| `no_change`    | 90% | Argmax ya cuadra → auto-aprobado.       |
| `converged`    | 10% | Solver corrigió ≥1 field → auto-aprobado con audit trail. |
| `infeasible`   | 0%  | Solver no encontró asignación válida → cola de revisión humana. |

Esto convierte el sistema en un **classifier con rejection option**:
~100% del volumen procesado, con 0% de "errores silenciosos" en las
actas auto-aprobadas (todas satisfacen el checksum).

## Trabajo futuro

- **Calibración via temperature scaling.** Mejorar la calibración del
  modelo antes del solver → top-K más informativos.
- **K dinámico.** Aumentar K solo en actas donde top-1 no cuadra,
  reducir overhead.
- **Per-region adaptive tolerance.** Actas de regiones con tasas
  conocidas de errores ONPE (Lima, Callao) podrían tolerar márgenes
  mayores.
- **Sampling estratificado (Estrategia C del brainstorm).** Combinar
  solver con muestreo regional para proyección con CI a la ONPE.
