# QUALITY REPORT -- Capa lakehouse (silver + gold)

## Resumen

- **PASS**: 8
- **WARNING**: 0
- **FAIL**: 0

## Detalle por expectativa

### [EXP 1] Conteo silver vs fuente -- **PASS**

- **Claim:** silver_votos_oficial = votos presidenciales de actas de escrutinio (sin fan-out ni perdida).
- **Metodo:** count(silver_votos_oficial) vs count(votos idEleccion=10 de archivos tipo=1).
- **Evidencia:** svo=3,461,671 ref=3,461,671

### [EXP 2] Integridad de FK en gold -- **PASS**

- **Claim:** Toda FK de fact_qa_modelo y fact_resultados_oficiales existe en su dimension.
- **Metodo:** Anti-joins fact -> dim_acta / dim_organizacion_politica / dim_ubicacion.
- **Evidencia:** huerfanos: fqa_acta=0, fqa_npos=0, fro_acta=0, fro_npos=0, fro_ubic=0

### [EXP 3] Dominio de nposicion -- **PASS**

- **Claim:** nposicion en {1..38, 80, 81, 82}.
- **Metodo:** count de filas con nposicion fuera del dominio.
- **Evidencia:** fuera_de_dominio=0

### [EXP 4] Sin nulls en nvotos_oficial -- **PASS**

- **Claim:** nvotos_oficial nunca es null (los votos existen aunque el total del acta sea NaN).
- **Metodo:** count de nulls en silver_votos_oficial y fact_resultados_oficiales.
- **Evidencia:** nulls_svo=0, nulls_fro=0

### [EXP 5] match_flag de gold ~= field_acc de evaluate -- **PASS**

- **Claim:** avg(match_flag) de fact_qa_modelo coincide con field_acc de evaluate (bronze).
- **Metodo:** avg(match_flag) gold vs avg(correct) de pred_evaluate_val.
- **Evidencia:** gold=0.9887 evaluate=0.9887 diff=0.0000 (tol=0.01)

### [EXP 6] MAE del total agregado gold == evaluate -- **PASS**

- **Claim:** El MAE del total agregado calculado sobre gold coincide con evaluate (bronze).
- **Metodo:** MAE = avg(|sum_pred - sum_real|) por acta (sin total_ciudadanos) en gold vs bronze.
- **Evidencia:** mae_gold=2.4040 mae_evaluate=2.4040 diff=0.0000 (tol=0.01)

### [EXP 7] Sin leak entre splits -- **PASS**

- **Claim:** Las actas evaluadas (fact_qa_modelo) estan todas en val y en ningun train/test.
- **Metodo:** Set diff de archivoId de fact_qa_modelo vs val/train/test_ids.txt.
- **Evidencia:** evaluadas=693 fuera_de_val=0 en_train=0 en_test=0

### [EXP 8] Grano unico -- **PASS**

- **Claim:** (idActa,nposicion) unico en fact_resultados_oficiales; (archivoId,field) unico en silver_predicciones (29,106 filas).
- **Metodo:** Conteo de grupos duplicados por grano + conteo de filas.
- **Evidencia:** dup_fro=0 dup_sp=0 filas_sp=29,106
