"""actas_cnn — pipeline de reconocimiento de cifras manuscritas en actas ONPE.

Paquete "laboratorio": fuente de verdad del pipeline (render, preprocesamiento,
modelo, entrenamiento, evaluacion). El entregable es el notebook autonomo de
Colab; este paquete soporta el desarrollo, los experimentos y la reproducibilidad
local via los wrappers de `scripts/`.

Submodulos:
  - render        PDF -> PNG (PyMuPDF, tamano fijo)
  - preprocess    *** superficie de iteracion: deteccion de digitos + crop ***
  - data          CropsDataset, transforms, build_manifest
  - model         resnet18_cifar (modelo del proyecto) + baselines
  - training      loop de entrenamiento
  - evaluate      metricas digit/field/acta-level + reconstruccion del total
  - metrics       matriz de confusion, P/R/F1 por clase, tabla de ablations
  - config/env/storage   transversales

Los imports pesados (torch, etc.) viven dentro de cada submodulo, no aca, para
que `import actas_cnn` sea barato.
"""

__version__ = "0.2.0"
