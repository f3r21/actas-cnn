# experiments/

Codigo de pruebas y experimentos **fuera del hot path** del entregable. Sirve
para investigar y validar, no para producir las metricas oficiales. El pipeline
oficial vive en `src/actas_cnn/` y los notebooks en `notebooks/`.

- **`fiducial/`** — Localizador alternativo de digitos por marcadores fiduciales
  (detecta 15 marcadores y alinea por afin antes de recortar). Fue el
  **experimento negativo** de Semana 2: tecnicamente correcto pero -0.72pp
  acta-level vs el zonal por plantilla. Implementa la misma idea que la interfaz
  `DigitLocalizer` de `actas_cnn.preprocess`. Scripts: `detect_fiducials.py`,
  `regenerate_anchors.py`, `test_alignment.py`.

- **`audits/`** — Auditorias exploratorias: calidad de render, generalizacion del
  template, ranking de errores, validacion del detector fiducial. La auditoria
  oficial de claims (la que genera `AUDIT_REPORT.md`) es `scripts/audit.py`, no
  estas.

- **`solver/`** — Post-procesamiento que corrige predicciones de campo con una
  restriccion. Ver su README: el codigo generador no esta en el repo (a recuperar).

Estos scripts importan el paquete via `sys.path` a `src/` (ver el bootstrap al
inicio de cada archivo); el detector fiducial se importa como modulo hermano
dentro de `fiducial/`.
