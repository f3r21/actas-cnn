# AUDIT — Generalizacion del template Presidencial

Auditoria sobre 1401 actas (val + test).
Modelo: DeepCNN 5 epochs val_acc 95.5%.

## Distribucion de accuracy por acta

- **mediana**: 1.0000
- **media**:   0.9702
- **min**:     0.2353
- **P10**:     0.9487
- **P25**:     1.0000
- **P75**:     1.0000

![histograma](data/visualizaciones/audit_template_histogram.png)

## Lectura cuantitativa

- 1188 actas con acc >= 0.99
- 70 actas con acc 0.95-0.99
- 85 actas con acc 0.80-0.95
- 21 actas con acc 0.50-0.80
- 37 actas con acc < 0.50 (template probablemente roto)

## Los 20 peores actas

![worst overlays](data/visualizaciones/audit_template_worst_20.png)

- `69e47a45bbc459e6486a81f3`  acc=0.235  n_crops=34
- `69e291bdd7b6147f63ecbc8f`  acc=0.250  n_crops=32
- `69e22147d7b6147f63ec8db0`  acc=0.258  n_crops=31
- `69e0aafcd7b6147f63eb1c0d`  acc=0.258  n_crops=31
- `69dc54cfd7b6147f63e5afd2`  acc=0.273  n_crops=33
- `69dc7271d7b6147f63e5f6cd`  acc=0.280  n_crops=25
- `69e045ded7b6147f63ea4b3d`  acc=0.294  n_crops=17
- `69e2c744d7b6147f63ecf91d`  acc=0.310  n_crops=29
- `69e029abd7b6147f63ea0ee0`  acc=0.312  n_crops=32
- `69df62efd7b6147f63e8e47f`  acc=0.320  n_crops=25
- `69e0ad37d7b6147f63eb2161`  acc=0.333  n_crops=21
- `69e41cdfbd301579eab812d5`  acc=0.350  n_crops=20
- `69e00779d7b6147f63e9cbb3`  acc=0.355  n_crops=31
- `69e04df3d7b6147f63ea5e9e`  acc=0.360  n_crops=25
- `69dfb3bad7b6147f63e971b4`  acc=0.400  n_crops=30
- `69e05094d7b6147f63ea649b`  acc=0.417  n_crops=12
- `69dc5048d7b6147f63e5a44a`  acc=0.417  n_crops=24
- `69e46e5e6edef3a0ec581f33`  acc=0.429  n_crops=35
- `69e039a0d7b6147f63ea3086`  acc=0.452  n_crops=31
- `69e00a31d7b6147f63e9d2f3`  acc=0.485  n_crops=33

## Conclusion

**Veredicto**: FAIL — template NO generaliza

Hay 37 actas con accuracy <50%. El template necesita revision sistematica.