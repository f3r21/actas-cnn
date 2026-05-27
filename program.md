# program.md — Loop autonomo de optimizacion de hiperparametros

Este archivo es la "skill" del agente Claude que itera ablations sobre el
ResNet-18 CIFAR del proyecto. Inspirado en `karpathy/autoresearch`, adaptado
a CNN + clasificacion de digitos manuscritos. El agente lee este archivo al
inicio de cada turno y sigue el protocolo.

## Objetivo unico

Maximizar la metrica compuesta

    composite = 0.5 * digit_acc + 0.5 * acta_acc

sobre el split `val` (693 actas, 29,106 campos). NUNCA evaluar sobre
`test` durante el search.

Baseline conocido (resnet18_best.pt actual): digit=0.9812, acta=0.9033,
composite=0.94225.

## Espacio de busqueda

Solo se varia lo que `train.py` ya expone via CLI. Hiperparametros como
`lr`, `batch_size` y `seed` viven en `config.py` (`TRAIN`) y quedan
fijos: 5e-4, 128, 42.

| Hiperparametro    | Valores                              | Notas                       |
|-------------------|--------------------------------------|-----------------------------|
| `arch`            | `resnet18`                           | unico modelo en model.py    |
| `label_smoothing` | {0.0, 0.05, 0.1}                     | epsilon en CrossEntropy     |
| `randaugment`     | {false, true}                        | RandAugment(num_ops=2, m=9) |
| `mixup`           | {0.0, 0.1, 0.2}                      | alpha del Beta              |
| `cosine_lr`       | {false, true}                        | CosineAnnealingLR           |
| `epochs`          | 20 fijo                              | budget del run              |

Espacio combinatorio total: 3 x 2 x 3 x 2 = **36 configs**. Con budget de
~20 runs no se cubre todo: el agente debe elegir.

## Protocolo por iteracion

1. **Leer estado**. Cargar `results.tsv` entero. Identificar:
   - mejor `composite` actual (`best_prev`),
   - configs ya exploradas (no repetir),
   - direcciones prometedoras (cambios que mejoraron).

2. **Proponer una sola config**. Emitir un JSON valido en
   `configs/run_XXX.json` (numero correlativo) con 1-2 lineas de
   justificacion: que hipotesis prueba. Ejemplo: "label smoothing alivia
   overconfidence -> sube acta_acc sin bajar digit_acc".

3. **Smoke en M2** (opcional, recomendable la primera vez en una rama
   nueva del espacio). 2 epochs locales para verificar que la config no
   rompe nada:

       python scripts/run_experiment.py --config configs/run_XXX.json --smoke

4. **Run completo en Kaggle**. Emitir el bloque de comandos que el humano
   copia al notebook. Esperar a que el humano regrese con la salida
   completa de `run_experiment.py` (que ya incluye el row agregado a
   `results.tsv`).

5. **Decision keep/discard**. Comparar `composite_new` vs `best_prev`:
   - `composite_new > best_prev + 0.001` -> `kept=Y`, nuevo best.
   - en caso contrario -> `kept=N`. La config queda registrada pero no
     se adopta como punto de partida.

6. **Reportar**. Una linea: que se cambio, que paso con la metrica, que
   sigue.

## Criterios de parada

- 20 runs alcanzados, **o**
- 5 runs consecutivos con `kept=N` (search converge), **o**
- el humano dice "stop".

## Estrategia sugerida (no obligatoria)

Orden razonable para los primeros ~10 runs, una variable a la vez sobre
el baseline:

1. **Regularizacion individual** (4 runs):
   - label_smoothing=0.1 solo.
   - randaugment=true solo.
   - mixup=0.2 solo.
   - cosine_lr=true solo.
2. **Combinaciones de los 2 mejores del paso 1** (~3 runs).
3. **Variantes finas** (label_smoothing=0.05, mixup=0.1) sobre la mejor
   combinacion (~3 runs).
4. **Confirmacion** del best con un re-run (~1 run).

## Reglas duras (no negociables)

- **Nunca tocar `data/`** ni `scripts/build_crops.py` ni
  `scripts/split_dataset.py` ni los manifests CSV.
- **Nunca evaluar `test`** durante el search. Solo `--split val`.
- **No modificar `train.py` ni `model.py` ni `config.py`** durante el
  loop. Solo flags CLI soportados. Si una idea requiere editar codigo
  (varia lr, batch_size, weight_decay, agregar ResNet-34), anotarla
  aparte como follow-up.
- Cada run usa **`suffix` unico** (e.g. `run_007`) para no sobrescribir
  `checkpoints/resnet18_best.pt` (baseline protegido).
- Si un run **crashea**, queda registrado en `results.tsv` con
  `kept=ERR` y `notes` describiendo la falla. No silenciar errores.
- Resultados con diferencias `<= 0.001` en composite **no se consideran
  mejoras** (ruido de seed).

## Como ejecutar el run completo en Kaggle

El humano corre, en el notebook de Kaggle o equivalente:

```python
!python scripts/run_experiment.py --config configs/run_XXX.json
```

El script:
1. Llama a `train.py` con los flags del JSON.
2. Llama a `scripts/evaluate.py --split val --checkpoint <ckpt>`.
3. Parsea stdout (regex sobre las tres lineas "X-level accuracy: Y.YYYY").
4. Calcula `composite` y append una fila a `results.tsv`.

El humano hace commit de `results.tsv` + el nuevo `configs/run_XXX.json`
al repo entre sesiones.

## Que NO hacer

- **No** integrar W&B en este loop (overkill para 20 runs).
- **No** correr Optuna ni Ray Tune por encima.
- **No** tocar preprocesamiento de crops (intentos previos degradaron;
  ver `CLAUDE.md` "Experimento Sem 2 dia 2").
- **No** entrenar ResNet-34 sin antes agregarlo a `model.py` (fuera del
  scope de este loop; anotarlo como follow-up).
- **No** evaluar con `evaluate_with_solver.py` durante el search; ese
  truco es para el numero final del informe, no para optimizar.

## Cierre

Cuando se alcance criterio de parada:

1. Imprimir las 5 mejores filas de `results.tsv` por `composite`.
2. Resumir en prosa (~5 lineas) que hiperparametros importaron y cuales
   no, apto para copiar al capitulo de ablations del informe.
3. Recomendar la config final para entrenar el modelo definitivo
   (run de mayor `composite`).
