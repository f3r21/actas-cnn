# 08 - Walkthrough del codigo (notebooks 01 y 02)

> Explicacion a fondo del codigo del entregable: bloque por bloque, decisiones de
> implementacion y que decir en la presentacion. **Fuente de verdad:** el paquete
> `src/actas_cnn/`; los notebooks Colab inline-an esta misma logica via
> `tools/_inline_code.py` (generados con `python tools/build_notebooks.py`). La
> logica es identica; solo cambia el empaquetado.
>
> El orden sigue el **pipeline**: PDF -> crops etiquetados (notebook 01) -> modelo
> + metricas (notebook 02).

## Mapa rapido del pipeline

```
PDF ONPE
  -> render.rasterize_first_page      PDF -> imagen gris 2339x3309 (en memoria)
  -> preprocess.localize_digits       42 campos por plantilla -> celdas de digito
  -> preprocess (ink-aware) + labels  etiqueta cada celda desde los parquets ONPE
  -> data.build_manifest              CSV (path,label) -> CropsDataset
  -> model.resnet18_cifar             ResNet-18 estilo CIFAR (entrada 1x32x32)
  -> training.train_model             receta ls_ra_mu_cos (LS + RandAug + mixup + cosine)
  -> evaluate.evaluate_split          reconstruye votos y compara vs parquets ONPE
```

> **Sobre los numeros de este doc:** los valores de metricas que aparecen como
> ejemplo (p.ej. digit 98.12% / field 98.87% / acta 90.33%, o el delta eval-side
> 98.87 -> 99.45%) son las cifras *baseline / right-justified* documentadas, usadas
> para ilustrar que mide cada metrica. Las **oficiales vigentes** (modelo
> `ls_ra_mu_cos` ink-aware, re-entrenado y **publicado en HF** el 2026-06-18) estan
> en `README.md` y `docs/04`: **val** digit 98.83 / field 99.34 / acta 90.62; **test**
> 98.31 / 99.00 / 88.84.


---

## Indice

1. 1. Deteccion y recorte de digitos (notebook 01)
2. 2. Etiquetas desde el ground truth de ONPE + manifest (notebook 01)
3. 3. Orquestacion del notebook 01 en Colab (render paralelo + bundle)
4. 4. El modelo: ResNet-18 estilo CIFAR (notebook 02)
5. 5. Dataset, transforms y entrenamiento: la receta ls_ra_mu_cos (notebook 02)
6. 6. Evaluacion y metricas: de digitos a votos vs ONPE (notebook 02)

---

## 1. Deteccion y recorte de digitos (notebook 01)

**Que hace** — Este modulo es el primer eslabon del pipeline: convierte un PDF de acta electoral (una imagen escaneada de papel con cifras manuscritas) en un conjunto de recortes individuales de digitos, cada uno guardado en una carpeta nombrada con su etiqueta (`crops/<label>/...png`). Es lo que produce el dataset de imagenes 1-canal que despues alimenta a la CNN. No hay modelo de deteccion aprendido aqui: la localizacion de los digitos es preprocesamiento clasico por plantilla (zonal), lo cual mantiene el proyecto dentro del temario del curso (la CNN hace la clasificacion; el recorte es geometria + reglas).

Vive en `src/actas_cnn/preprocess/crops.py` (fuente de verdad) y `src/actas_cnn/render.py` (rasterizado). El notebook entregable `01_preprocesamiento_colab.ipynb` lleva la *misma* logica inline, ensamblada desde el bloque `PREPROCESS` + `LABELS_BUILD` de `tools/_inline_code.py`. La logica es identica; las unicas diferencias son de empaquetado (en el notebook `rasterize_first_page` se llama `rasterize_acta` con `TARGET_SIZE` como constante de modulo, y `field_value_for` se inyecta como string compartido por los notebooks 01 y 02). Lo anoto donde corresponde.

**Explicacion del codigo** — El flujo va de PDF a crops en cuatro pasos. Lo recorro en orden.

*1) Rasterizado del PDF — `render.rasterize_first_page` (`render.py:25-43`).* Abre el PDF con PyMuPDF (`fitz`), toma `doc[0]` (primera pagina; las actas manuscritas son de 1 pagina, las STAE digitales de 2 ya estan filtradas) y renderiza a un **tamano fijo** 2339x3309 (`TARGET_W, TARGET_H` en `render.py:22`), no a un DPI fijo:

```python
if page.rect.width > page.rect.height:
    page.set_rotation(90)                    # landscape -> portrait
pix = page.get_pixmap(matrix=fitz.Matrix(tw / page.rect.width,
                                         th / page.rect.height))
return Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
```

La `fitz.Matrix` con factores `tw/ancho` y `th/alto` escala cada pagina a exactamente el mismo lienzo, y `.convert("L")` la pasa a escala de grises (8-bit). Devuelve una `PIL.Image` en memoria — no escribe PNG. En el notebook esto es `rasterize_acta(pdf_path)` (`_inline_code.py:28-40`), pixel-identico.

*2) Localizar los 42 campos — `crop_fields` + `localize_digits` (`crops.py:29-73`).* La plantilla (`templates.json`, clave `presidencial`) define 42 campos: 38 `partido_NN`, `votos_blanco/nulos/impugnados` y `total_ciudadanos`. Cada campo trae `box: [x0,y0,x1,y1]` en **fraccion [0,1]** y `n_digits` (3 para casi todos, 4 para el total). `crop_fields` multiplica esas fracciones por el ancho/alto reales y recorta:

```python
x0, y0, x1, y1 = field["box"]
box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
crops[field["name"]] = img.crop(box)
```

`localize_digits` (`crops.py:62-73`) es el **punto unico de "donde estan los digitos"**: recorta cada campo y lo parte en celdas, devolviendo `{nombre_campo: [celda_0, celda_1, ...]}`. La docstring lo dice explicito: para cambiar el metodo de deteccion, se reemplaza esta funcion o se pasa otro callable con la misma firma a `build_crops_for_acta` (el localizador fiducial alternativo, experimento negativo, vive en `experiments/fiducial/`).

*3) Partir cada campo en celdas — `split_digits` (`crops.py:47-59`).* Division **equiespaciada**: `step = w / n_digits` y recorta `n_digits` franjas verticales iguales de izquierda a derecha. Un campo de 3 digitos da 3 celdas; el total da 4.

*4) Filtrar celdas vacias — `es_celda_escrita` (`crops.py:76-94`).* Aqui entra la convencion ONPE: las cifras se escriben **right-justified** y los ceros a la izquierda se dejan en blanco. La funcion decide, dado el valor real y la posicion de celda, si esa celda *deberia* tener tinta:

```python
if value == 0:
    return False                              # nadie escribe "000"
num_digits = len(str(int(value)))
first_written = n_cells - num_digits
return cell_position >= first_written
```

Asi, `value=5` en 3 celdas marca solo la celda 2 como escrita; `value=20` marca celdas 1 y 2 (el "0" final SI se escribe). Las celdas vacias no se guardan como crops.

*5) El hallazgo clave — remapeo ink-aware (`crops.py:109-250`).* El problema: ~3% de las actas viola la convencion right-justified (el escribiente llena desde la primera celda, o centrado). En esas actas el etiquetado posicional **envenena el dataset**: pone label de digito sobre celdas vacias y label del vecino sobre digitos reales. La doc del proyecto reporta que estas actas concentraban el 82% de los errores de campo. La solucion es detectar *donde cae la tinta* y remapear labels solo en esas actas. Las funciones helper:

- `ventana_central(cell_img, mx=0.25, my=0.15)` (`crops.py:120-126`): recorta el centro de la celda (descarta 25% horizontal, 15% vertical de margen). La tinta del digito propio cae al centro; el sangrado del trazo del vecino y los bordes punteados de la celda quedan en los margenes. Devuelve un array uint8.
- `umbral_adaptativo(arrays, delta=55)` (`crops.py:129-138`): umbral de oscuridad **relativo al fondo de ESTA acta**. Toma la mediana de *todos* los pixeles de las celdas (que es fondo casi puro, porque la tinta es minoria) y devuelve `clip(fondo - delta, 40, 200)`. Un umbral fijo fallaria en escaneos grisaceos (fondo ~170 haria que todo cuente como tinta).
- `patron_de_tinta(fracs, piso=0.07, rel=0.55)` (`crops.py:141-180`): recibe la fraccion de pixeles oscuros por celda de un campo y clasifica el patron. El corte es **relativo al maximo del campo** (`corte = max(piso, rel*fmax)`), porque la fraccion absoluta de un "1" delgado empata con el sangrado del vecino, pero dentro del campo el digito escrito siempre domina. Luego busca *runs* contiguos de celdas con tinta:
  - 1 run que termina en la ultima celda -> `RIGHT` (cumple la convencion)
  - 1 run que arranca en la primera sin llegar al final -> `LEFT` (corrido a la izquierda)
  - 1 run que no toca ningun extremo -> `MEDIO` (centrado)
  - mas de un run -> `OTRO` (salteado); tinta demasiado tenue -> `AMBIGUO`

`remapeo_ink_aware` (`crops.py:183-250`) orquesta la decision **por acta**:

1. Calcula `fracs` por campo (solo campos con `value > 0`) y su patron.
2. Cuenta patrones. Si `informativos < min_informativos` (4) o la proporcion de violadores `(LEFT+MEDIO)/informativos < umbral_viola` (0.5), devuelve `{}` — la acta cumple la convencion, nada cambia. **El caso normal devuelve diccionario vacio.**
3. Junta candidatos: campos `LEFT/MEDIO` cuyo run de tinta tiene *exactamente* el largo del valor (`run[1]-run[0] == len(digitos)`). Si el run no calza con el numero de digitos, no se toca.
4. **El guard contra digitos a caballo** (`crops.py:235-247`): distingue una violacion real de un simple *offset* del escaneo. Si el acta esta corrida unos pixeles, un digito puede quedar "a caballo" entre dos ventanas, y la celda que la convencion *espera* escrita tambien tiene tinta — pero medida **a celda completa** (`mx=0.05, my=0.05`), porque pegada al borde la ventana central no la ve. En la violacion real esa celda esta vacia. Se decide por la *mediana* sobre los campos remapeables (el straddle es sistematico, el sangrado puntual no). Si `median(evidencias) >= piso`, devuelve `{}`: es offset, el etiquetado posicional ya es correcto.
5. Solo si pasa todo, devuelve `{campo: {pos: label}}` con los labels reasignados a las celdas realmente escritas.

*6) Orquestador — `build_crops_for_acta` (`crops.py:303-364`).* Carga el ground truth del acta, descarta actas sin `totalVotosEmitidos` (NaN: estado "Para envio al JEE"), localiza las celdas, llama `remapeo_ink_aware`, y por cada campo elige el etiquetado:

```python
if name in remap:
    etiqueta_en = remap[name]                 # acta violadora: labels ink-aware
else:
    labels = int_to_digits(value, n_cells)
    etiqueta_en = {pos: labels[pos] for pos in range(n_cells)
                   if not filtrar_vacias or es_celda_escrita(value, n_cells, pos)}
```

Luego guarda cada celda presente en `crops_root/<label>/<archivoId>_<campo>_c<pos>.png` y cuenta `(n_guardados, n_filtrados)`. El nombre del archivo codifica `archivoId`, campo y posicion — exactamente lo que `parse_crop_path` re-extrae en evaluacion (notebook 02) para reconstruir el valor del campo concatenando los digitos predichos.

**Decisiones de implementacion (el porque)** —
- **Tamano fijo en vez de DPI fijo** (`render.py` docstring): los PDFs originales tienen page sizes ligeramente distintos; a DPI fijo salen ~74 dimensiones distintas en disco. Tamano fijo da PNGs uniformes, hace estable cualquier detector basado en pixeles absolutos y elimina la rama de "imagen rara".
- **Render en memoria (sin PNG intermedio)**: el encode+write+decode de un PNG de 7.7 Mpx es ~3/4 del tiempo por acta (0.75s de 0.97s medidos en M2) y el archivo se borraria igual. Verificado byte a byte que da pixeles identicos.
- **Cajas en fraccion [0,1]**: tolera diferencias de DPI/escaneo sin recalibrar la plantilla por acta.
- **Segmentacion equiespaciada en vez de projection profile**: probaron projection profile vertical (Sem 2 dia 2) para acomodar escritura irregular, pero la transformacion afin del template ya alinea bien en >98% de las actas y el equiespaciado dio mejor accuracy downstream (acta-level 90.33% vs 89.61%). KISS gano.
- **Filtrar vacias por label (`es_celda_escrita`) y no por imagen (`tiene_tinta`)**: la separacion empty-vs-escrito es determinista desde el ground truth, no depende de un umbral de imagen. Resuelve el imbalance brutal (76% de las celdas serian vacias y dominarian el training). `tiene_tinta` queda solo como sanity-check.
- **Ink-aware conservador**: el remapeo *solo* toca campos donde el remapeo es confiable (run del largo exacto + guard de straddle); todo lo demas conserva el etiquetado posicional. El principio explicito en la docstring: "el fix solo toca lo que puede arreglar con confianza, nunca empeora el statu quo". Resultado eval-side (mismo checkpoint, sin re-entrenar): field 98.87 -> 99.45%, MAE 2.40 -> 1.58, 0 regresiones en las otras 673 actas.

**Sutilezas / gotchas** —
- **Auto-rotacion landscape -> portrait**: si no se normalizara la orientacion, las cajas en fraccion caerian en el lugar equivocado. Esta tanto en `rasterize_first_page` como en `pdf_to_images` (deben coincidir).
- **El "0" final SI se escribe**: `value=20` -> celdas [vacia, "2", "0"]. Confunde a primera vista pensar que los ceros nunca se escriben; solo los *leading* zeros se dejan en blanco.
- **El guard mide a celda completa, no en la ventana central**: parece inconsistente con el resto (que usa ventana central), pero es deliberado: un digito a caballo queda pegado al borde, donde la ventana central no lo ve, asi que para detectarlo hay que mirar la celda entera (`mx=0.05`). Es el truco que evita "arreglar" actas que en realidad solo estan corridas unos pixeles (1 acta de val escribe asi y evalua perfecto sin tocarla).
- **Corte relativo al maximo del campo, no absoluto**: clave para que un "1" delgado no se confunda con el sangrado del vecino. La normalizacion por campo es lo que hace robusto a `patron_de_tinta`.
- **`reconstruct_value` (notebook 02) es agnostico a la convencion**: concatena los digitos de las celdas *presentes* en orden de posicion. Funciona igual para right-justified e ink-aware, porque ya remapeamos los labels al guardar — la posicion es orden de lectura.
- **Estado actual del modelo**: el delta ink-aware de abajo (98.87 -> 99.45%) se midio del lado de evaluacion con el checkpoint viejo *right-justified*. Desde el 2026-06-18 el modelo oficial **ya es ink-aware** (`ls_ra_mu_cos`, re-entrenado en Colab y publicado en HF; metricas vigentes en `README.md`/`docs/04`).

**Para la presentacion** —
- "El recorte de digitos es **preprocesamiento clasico por plantilla**, no un modelo de deteccion — la CNN se reserva para la clasificacion. Cada acta se rasteriza a un lienzo fijo de 2339x3309, se recorta en 42 campos por cajas en fraccion [0,1], y cada campo se parte en 3-4 celdas equiespaciadas."
- "El truco para el dataset es **etiquetar sin OCR**: usamos el ground truth de ONPE y la convencion right-justified (`es_celda_escrita`) para saber que celda tiene cada digito. Esto elimina el 76% de celdas vacias que arruinarian el balance de clases."
- "El **hallazgo clave** fue que ~3% de las actas viola esa convencion (escriben corrido o centrado) y, aunque son pocas, concentraban el 82% de los errores de campo. El etiquetado posicional las envenenaba: ponia el label sobre la celda equivocada."
- "La solucion es **ink-aware**: detectamos donde cae la tinta (umbral adaptativo al fondo de cada escaneo, runs de celdas con tinta) y remapeamos los labels solo en las actas violadoras. Es conservador por diseno — con un guard que distingue una violacion real de un simple offset del escaneo — asi que nunca empeora las actas normales: 0 regresiones."
- "Impacto medido sin re-entrenar, solo corrigiendo el etiquetado: field-level 98.87 -> 99.45%, MAE del total 2.40 -> 1.58 votos, -52% de campos mal. El bug no estaba en la geometria del preprocesamiento sino en los **labels**."

---

## 2. Etiquetas desde el ground truth de ONPE + manifest (notebook 01)

**Que hace** — Este modulo es el puente entre los pixeles y la verdad. El recortador (`localize_digits`) sabe *donde* estan los digitos en la imagen, pero no *que* digito es cada celda. Este modulo resuelve eso: cruza cada acta con los parquets oficiales de ONPE para sacar el valor entero de cada campo (cuantos votos sacó cada partido, blancos, nulos, total), descompone ese entero en digitos por celda siguiendo la convencion ONPE de escritura *right-justified*, y guarda cada celda como un PNG dentro de la carpeta de su clase (`crops/<label>/*.png`). Sin esta etapa no hay supervision: la CNN aprende a leer digitos porque cada crop ya viene con la respuesta correcta en el nombre de su carpeta.

La salida en disco es un dataset estilo ImageFolder (una subcarpeta por digito 0–9), y `build_manifest` (`data.py:16`) lo aplana a un CSV `path,label` que es lo unico que el `Dataset` de PyTorch necesita leer. Esa indireccion via manifest hace que sea indiferente si los crops viven en local, en Hugging Face o en un mirror: solo cambia la raiz, no el codigo de entrenamiento. En el entregable, todo esto corre dentro del **notebook 01** (`01_preprocesamiento_colab.ipynb`), que inline-a exactamente esta logica desde los bloques `FIELD_VALUE_FOR` y `LABELS_BUILD` de `tools/_inline_code.py`.

**Explicacion del codigo**

*El cruce determinista `archivoId -> idActa`.* Los parquets de ONPE no estan indexados por el nombre del PDF. Cada acta-archivo tiene un `archivoId` (el identificador del documento que da nombre al PDF), pero los votos y la cabecera estan indexados por `idActa` (el identificador de la mesa/acta logica). El puente esta en `actas_archivos.parquet`. En el driver del notebook (`build_notebooks.py:256`) se construye el diccionario una sola vez:

```python
aid_to_idacta = dict(zip(archivos["archivoId"], archivos["idActa"]))
```

y luego se llama `build_crops_for_acta(img, aid, int(aid_to_idacta[aid]), ...)` (`build_notebooks.py:269`). Es un join determinista uno-a-uno: cada PDF resuelve a exactamente un `idActa`, y todos los lookups posteriores (votos, cabecera) son por ese `idActa`. El paquete deja este join *afuera* de `build_crops_for_acta`: la funcion recibe `id_acta` ya resuelto como argumento (`crops.py:308`), no la tabla `archivos`. Esto mantiene la funcion pura respecto al esquema de parquets y le da una sola responsabilidad.

*Sacar el valor real de cada campo: `field_value_for`* (`crops.py:277`). Esta es la funcion clave del etiquetado. Recibe el nombre del campo de la plantilla, el sub-DataFrame de votos de *esa* acta (`votos_acta`) y el total ya extraido. Despacha por nombre:

```python
if name.startswith("partido_"):
    pos = int(name.split("_")[1])
    row = votos_acta[votos_acta["nposicion"] == pos]
    return int(row.iloc[0]["nvotos"]) if len(row) else 0
```

Los campos se llaman `partido_1`, `partido_2`, …, `partido_38`: el sufijo numerico *es* la `nposicion` en la tabla `actas_votos`. Asi que `partido_7` busca la fila con `nposicion == 7` y devuelve su `nvotos`. Para los campos especiales hay un mapping a posiciones reservadas (`crops.py:288`):

```python
mapping = {"votos_blanco": 80, "votos_nulos": 81, "votos_impugnados": 82}
```

ONPE codifica blanco/nulos/impugnados en `nposicion` 80/81/82 (definidos como constantes `NPOSICION_BLANCO/NULOS/IMPUGNADOS` en `crops.py:256-258`), fuera del rango de partidos. Y `total_ciudadanos` no sale de `actas_votos` sino de `actas_cabecera.totalVotosEmitidos`, que se pasa como argumento `total_emitidos` (`crops.py:296`). El **fallback a 0** (`if len(row) else 0`) es deliberado: si un partido no tiene fila en el parquet, significa que sacó cero votos, no que falte el dato. Cualquier nombre que no calce lanza `ValueError` (`crops.py:298`) — falla rapido si la plantilla y el codigo se desincronizan.

> Nota de naming: la funcion que saca el valor real de un campo aparece con **dos nombres para el mismo cuerpo** — `field_value_for` en `crops.py` y en los bloques inline de los notebooks (etiquetado, notebook 01), y `real_value_for` en `src/actas_cnn/evaluate.py` (evaluacion, notebook 02). Mismo despacho por `nposicion` (partidos 1..38, blanco/nulos/impugnados 80/81/82, total desde cabecera). Ver la seccion 6.

*Descomponer el entero en digitos por celda: `int_to_digits`* (`crops.py:261`) y la convencion *right-justified*:

```python
s = str(int(value)).zfill(n_cells)   # 18 con 3 celdas -> "018"
return [int(c) for c in s]            # -> [0, 1, 8]
```

`zfill` rellena con ceros a la izquierda hasta `n_cells`. La sutileza es que ONPE escribe los numeros *pegados a la derecha* y deja las celdas sobrantes de la izquierda *en blanco* (nadie escribe "018", escribe " 18"). Por eso hay una segunda funcion, `es_celda_escrita` (`crops.py:76`), que decide cuales de esos digitos realmente fueron escritos a mano:

```python
if value == 0:
    return False                       # nadie escribe "000"
num_digits = len(str(int(value)))
first_written = n_cells - num_digits
return cell_position >= first_written
```

Para `value=18` en 3 celdas: `first_written = 3 - 2 = 1`, asi que la celda 0 esta vacia y las celdas 1 y 2 (el "1" y el "8") estan escritas. Para `value=0`: todas vacias. Notar que el "0" de `value=20` *si* cuenta como escrito (es el ultimo digito, posicion 2 ≥ 1), porque el escribiente si lo escribe — el comentario del codigo (`crops.py:84`) lo aclara explicitamente.

*El orquestador: `build_crops_for_acta`* (`crops.py:303`). Une todo y escribe a disco. Primero filtra actas sin ground truth utilizable:

```python
cab_row = cabecera[cabecera["idActa"] == id_acta]
if len(cab_row) == 0:
    return 0, 0
total_raw = cab_row.iloc[0]["totalVotosEmitidos"]
if pd.isna(total_raw):
    return 0, 0                        # actas "Para envio al JEE" / "Pendiente"
```

Las actas en estado pendiente tienen `totalVotosEmitidos` NaN; sin total no hay etiqueta para `total_ciudadanos`, asi que la acta entera se descarta del training set. Luego localiza los digitos (`fields_cells = localizer(image, template)`) y calcula el plan de remapeo ink-aware (cubierto en otra seccion; aca lo importante es como interactua con el etiquetado). El loop central (`crops.py:345`) es el corazon del etiquetado:

```python
for field in template["fields"]:
    name = field["name"]; n_cells = field["n_digits"]
    value = field_value_for(name, votos_acta, total_emitidos)
    digit_imgs = fields_cells[name]
    if name in remap:
        etiqueta_en = remap[name]      # acta corrida: posiciones reales de tinta
    else:
        labels = int_to_digits(value, n_cells)
        etiqueta_en = {pos: labels[pos] for pos in range(n_cells)
                       if not filtrar_vacias or es_celda_escrita(value, n_cells, pos)}
```

`etiqueta_en` es un dict `{posicion_de_celda: digito}` que contiene *solo* las celdas que se van a guardar. Si `filtrar_vacias=True` (el default), las celdas blancas segun `es_celda_escrita` ni siquiera entran al dict. Despues recorre las celdas y guarda cada una en la carpeta de su label:

```python
for pos, dimg in enumerate(digit_imgs):
    if pos not in etiqueta_en:
        n_filtered += 1; continue
    dest_dir = crops_root / str(etiqueta_en[pos])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dimg.save(dest_dir / f"{archivo_id}_{name}_c{pos}.png")
    n_saved += 1
```

El nombre del archivo `{archivo_id}_{name}_c{pos}.png` no es decorativo: codifica `archivoId`, nombre de campo y posicion de celda, y eso es exactamente lo que `parse_crop_path` (bloque `EVAL`, `_inline_code.py:355`) vuelve a parsear en la evaluacion para reconstruir el valor del campo. La carpeta destino (`crops_root / str(label)`) es lo que convierte el directorio en un dataset ImageFolder por clase.

*El manifest: `build_manifest`* (`data.py:16`). Aplana el arbol de carpetas a un CSV:

```python
for label_dir in sorted(crops_dir.iterdir()):
    if not label_dir.is_dir(): continue
    label = label_dir.name
    for img in sorted(label_dir.glob("*.png")):
        rows.append((str(img.relative_to(crops_dir)), label))
```

El `label` sale del *nombre de la carpeta* (`label_dir.name`), no del nombre del archivo — la verdad esta en la estructura de directorios. Las rutas se guardan **relativas** a `crops_dir` (`img.relative_to(crops_dir)`), que es lo que permite mover los crops entre local/HF sin reescribir el CSV. El `sorted(...)` en ambos niveles hace la salida determinista. Escribe el header `["path","label"]` y devuelve el conteo de filas. `CropsDataset.__getitem__` (`data.py:69`) luego hace `Image.open(self.root / row["path"])` y `torch.tensor(int(row["label"]))` — cierra el ciclo.

**Decisiones de implementacion (el porque)**

- **Etiquetado deterministico desde el label, no por deteccion de tinta image-based.** Existe `tiene_tinta` (`crops.py:97`) que detecta tinta por umbral de pixeles, pero el filtro principal vacio-vs-escrito se hace con `es_celda_escrita`, que es *deterministico desde el ground truth*. El comentario lo dice (`crops.py:98`): es mas robusto saber que `value=18` ocupa exactamente las 2 ultimas celdas que adivinarlo de los pixeles. Esto evita falsos positivos (manchas, sangrado del vecino) y falsos negativos (digitos tenues).

- **Filtrar celdas vacias resuelve un desbalance brutal.** El docstring de `build_crops_for_acta` (`crops.py:320`) lo cuantifica: ~76% de las celdas son vacias por la convencion right-justified. Si entraran al training set, la clase mayoritaria seria "vacio" y el modelo aprenderia a no escribir nada. Filtrarlas con `es_celda_escrita` deja solo celdas con digito real. Notar que *no se crea una clase "vacio"*: el problema se define como 10 clases (0–9) y las vacias simplemente no existen para el modelo. La reconstruccion en evaluacion respeta esto: concatena solo las celdas presentes.

- **El sufijo del campo *es* la posicion ONPE.** Llamar a los campos `partido_7` en vez de inventar otro mapping hace que `field_value_for` sea trivial: `int(name.split("_")[1])` ya da la `nposicion`. Cero tabla de traduccion para los 38 partidos.

- **Manifest CSV como capa de indireccion.** En vez de que el Dataset escanee el disco cada vez (lento, no reproducible), se materializa un CSV ordenado una vez. Esto desacopla "donde estan los bytes" de "que aprende el modelo" y hace el split train/val/test reproducible.

- **El join afuera de la funcion orquestadora.** `build_crops_for_acta` recibe `id_acta` ya resuelto en vez de la tabla `archivos`. El driver hace el join una vez y cachea `aid_to_idacta`; ademas restringe `votos`/`cabecera` a los `idActa` del split (`build_notebooks.py:258-260`) para que los lookups `votos["idActa"] == ida` sean rapidos sobre tablas pre-filtradas en vez de los millones de filas del parquet completo.

**Sutilezas / gotchas**

- **NaN en el total descarta la acta entera, no solo el campo total.** El guard `pd.isna(total_raw)` (`crops.py:335`) hace `return 0, 0` para *toda* la acta. Esto es intencional: sin total no se puede etiquetar `total_ciudadanos`, y mantener actas parcialmente etiquetadas ensuciaria la metrica acta-level. El driver del notebook elige por separado un acta-demo *con* total no-NaN para la preview (`build_notebooks.py:306-308`) justo por esto.

- **El "0" que si se escribe.** `es_celda_escrita` retorna `False` para `value==0` (acta sin votos) pero `True` para el ultimo digito de `value==20`. Confunde a primera vista: un "0" puede ser "celda escrita" o no, segun *donde* cae. La regla real es posicional: una celda es escrita si su posicion ≥ `n_cells - num_digits`, sin importar que digito sea.

- **`int_to_digits` puede lanzar si el valor no cabe.** `crops.py:272` valida `len(s) > n_cells` y lanza `ValueError`. Si una plantilla declara 3 celdas para un campo pero ONPE reporta 4 digitos, falla ruidosamente en vez de truncar silenciosamente — exactamente lo que uno quiere para detectar una plantilla mal dimensionada.

- **El parseo del nombre asume que el campo no tiene `_c<num>` adentro.** `parse_crop_path` hace `"_".join(parts[1:-1])` para el nombre del campo y `parts[-1][1:]` para la posicion (quita la "c"). Funciona porque `archivoId` no lleva guiones bajos problematicos y la posicion siempre es el ultimo token `c<pos>`. Es un acoplamiento implicito entre como se *escribe* el nombre en `build_crops_for_acta` y como se *lee* en `parse_crop_path`: cambiar el formato del nombre en un lado rompe el otro.

- **Paridad notebook/paquete.** La logica de `field_value_for`, `int_to_digits`, `es_celda_escrita`, `build_crops_for_acta` y `build_manifest` es identica entre `crops.py`/`data.py` y los bloques `FIELD_VALUE_FOR`/`LABELS_BUILD` de `_inline_code.py`. Dos diferencias *de superficie, no de logica*: (1) en el paquete `build_crops_for_acta` recibe `localizer=None` como parametro inyectable (para enchufar otro detector); el notebook lo cablea directo a `localize_digits` por simplicidad didactica. (2) El paquete tiene type hints y docstrings mas largos. El resultado en disco es bit-a-bit el mismo.

**Para la presentacion**

- "El modelo no etiqueta nada a mano: el ground truth de ONPE (los parquets oficiales) es la fuente de verdad. Cruzamos cada PDF con su acta via un join determinista `archivoId -> idActa`, y de ahi sacamos cuantos votos sacó cada uno de los 38 partidos mas blanco/nulos/impugnados (posiciones 80/81/82) y el total."

- "El truco central es la convencion ONPE: los numeros se escriben pegados a la derecha y las celdas de la izquierda quedan en blanco. Eso nos deja *derivar* la etiqueta de cada celda desde el numero — `18` en 3 celdas es `[vacio, 1, 8]` — sin tener que mirar la tinta. Etiquetado deterministico, no heuristico."

- "Filtramos las celdas vacias *antes* de entrenar: el 76% de las celdas son blancas. Si las metieramos, la CNN aprenderia a no escribir nada. Por eso el problema son 10 clases (0–9), no 11 con 'vacio'."

- "Guardamos cada digito en `crops/<label>/archivoId_campo_cPosicion.png`. El nombre codifica de donde viene cada celda, y eso es lo que en evaluacion nos deja *reconstruir* el numero completo concatenando los digitos predichos — cerramos el ciclo de imagen a entero a voto."

- "El manifest CSV (`path,label`) es una capa de indireccion deliberada: el entrenamiento solo lee ese CSV, asi que es indiferente si los crops viven en local o en Hugging Face. Cambia la raiz, no el codigo."

---

## 3. Orquestacion del notebook 01 en Colab (render paralelo + bundle)

**Que hace** — Este modulo no es codigo "del modelo": es el *generador* del notebook entregable de preprocesamiento. `build_notebooks.py` ensambla, con `nbformat`, las celdas de `01_preprocesamiento_colab.ipynb` combinando texto markdown, los bloques de logica inline-ados desde `_inline_code.py` (`INSTALL`, `PREPROCESS`, `LABELS_BUILD`) y unas pocas celdas de *orquestacion Colab* que viven solo aqui: la config con manejo del `HF_TOKEN`, la seleccion del universo de actas y descarga de PDFs, el loop paralelo de render+recorte reanudable, y el empaquetado+subida del bundle a Hugging Face. El notebook resultante es autonomo (no clona el repo): lleva su propia copia aplanada de la logica del paquete `actas_cnn`, de modo que un alumno solo abre el `.ipynb` en Colab y hace Run all.

El rol en el pipeline es ser el **primer notebook de los dos entregables**: toma 5000 PDFs de actas publicados en HF, los renderiza, detecta y recorta los digitos, etiqueta cada celda contra el ground truth de ONPE, arma los manifests, y publica `crops_bundle.tar.gz` en HF. El segundo notebook (`02_modelo_colab.ipynb`) baja ese bundle en segundos y entrena. Lo unico que acopla 01 y 02 son los *datos* (el bundle), no el codigo: por eso se iteran por separado.

**Explicacion del codigo**

*La funcion ensambladora.* `build_preprocesamiento()` (`build_notebooks.py:151`) devuelve una lista `cells` que es, literalmente, el notebook en orden. Cada elemento es `md(...)` (celda markdown, helper en `:33`) o `code(...)` (celda de codigo, helper en `:37`). Al final llama `_nb(cells, gpu=False)` (`:346`) que arma el `NotebookNode`, asigna IDs deterministas `c00, c01, ...` por indice (`:487`, para que regenerar no produzca churn de git) y — clave — **no** mete `accelerator: GPU` en el metadata porque `gpu=False` (`:498`).

*Config + manejo del HF_TOKEN (la celda mas delicada).* La celda de config se genera con `config_cell(modo_block, con_gpu=False)` (`:61`). El argumento `con_gpu=False` selecciona el `device_block` que es solo un comentario de aviso (`:74-76`): este notebook nunca instancia un device de PyTorch porque no entrena. El `modo_block` que se le inyecta (`:173-198`) contiene la logica del token:

```python
N_ACTAS = 5000
SUBIR_A_HF = True
REHACER_DESDE_CERO = False
if SUBIR_A_HF:
    import os
    if not os.environ.get("HF_TOKEN"):
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if not os.environ.get("HF_TOKEN"):
        from huggingface_hub import get_token
        if get_token() is None:
            raise RuntimeError("No hay HF_TOKEN y SUBIR_A_HF=True: la subida final fallaria...")
```

El flujo es: (1) si `HF_TOKEN` no esta como variable de entorno, intenta leerlo del **panel de secretos de Colab** via `google.colab.userdata.get("HF_TOKEN")` y lo exporta como env var; (2) el `try/except Exception` cubre los casos "no estoy en Colab" o "el secreto no existe / sin acceso"; (3) si despues de todo eso sigue sin token, hace fallback a `get_token()` (lee `.env` o token cacheado, util en local); (4) si tampoco hay, **frena AHORA** con `RuntimeError` en vez de dejar correr ~40 minutos y morir en la subida final. El comentario inline (`:182-184`) documenta por que se usa `userdata.get()` y no `get_token()` para el caso Colab: `get_token()` cachea por sesion, y si la primera consulta ocurrio antes de darle acceso al secreto, devuelve `None` para siempre.

*Por que CPU, no GPU.* Esta decision se ve en tres lugares. El markdown de cabecera (`:163-170`) instruye runtime CPU normal y explica la causa: Colab desconecta runtimes con GPU ociosa a mitad de la celda larga de render ("Runtime disconnected"). El `device_block` de `con_gpu=False` (`:74-76`) lo repite como comentario en la celda. Y `_nb(..., gpu=False)` omite `accelerator: GPU` del metadata (`:495-499`), asi que el notebook ni siquiera pide GPU al abrirse. El notebook 01 solo usa PyMuPDF/PIL/pandas/multiprocessing — la GPU jamas se toca.

*Seleccion del universo + descarga de PDFs* (celda en `:207-230`). Esta es la celda "## 2. Labels, universo y descarga de PDFs". Primero baja los parquets de labels en una pasada:

```python
snapshot_download(HF_DATASET_REPO, repo_type="dataset", allow_patterns="labels/*", local_dir=str(DATA))
archivos = pd.read_parquet(DATA / "labels/actas_archivos.parquet")
...
con_label = set(archivos["archivoId"])
ids = sorted(f[:-4] for f in list_repo_files(HF_DATASET_REPO, repo_type="dataset")
             if f.endswith(".pdf") and f[:-4] in con_label)
random.Random(42).shuffle(ids)
ids = ids[:N_ACTAS]
n = len(ids); ntr = int(n * 0.70); nva = int(n * 0.15)
splits = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
```

El detalle fino: el universo **no** sale del parquet `archivos` (que tiene ~84k presidenciales), sino de `list_repo_files(...)` filtrado a `.pdf` — esto da *exactamente* lo que realmente se subio a HF (5000 manuscritas). Seleccionar del parquet pediria PDFs inexistentes y daria 404 (comentario en `:215-217`). Se cruza con `con_label` para asegurar que cada PDF tenga su fila de labels. Luego `random.Random(42).shuffle(ids)` baraja con **semilla fija 42** (reproducibilidad: misma particion en cada corrida), se recorta a `N_ACTAS`, y se parte 70/15/15 por slicing posicional. El split es **por archivoId** (cada acta entera cae en un solo split), no por crop, lo que evita leakage entre train/val/test. Finalmente descarga **solo** esos PDFs en una pasada con `snapshot_download(..., allow_patterns=[f"{a}.pdf" for a in ids], ...)` (`:228-229`). El `logging.getLogger("huggingface_hub").setLevel(logging.ERROR)` (`:208`) silencia los 5000 avisos que congelarian el front-end de Colab.

*El loop de render+recorte: paralelo, reanudable, tolerante a fallos* (celda en `:245-295`). Primero `REHACER_DESDE_CERO` borra crops y progreso si editaste la deteccion (`:250-254`). Luego restringe los DataFrames de `votos`/`cabecera` a las actas elegidas (`:256-260`) para acelerar los joins por `idActa` dentro del recorte. La unidad de trabajo es `procesa_acta` (`:262-273`):

```python
def procesa_acta(aid, croot):
    try:
        pdf = pdf_dir / f"{aid}.pdf"
        if not pdf.exists():
            return aid, 0, "pdf no descargado"
        img = rasterize_acta(pdf)
        ns, _ = build_crops_for_acta(img, aid, int(aid_to_idacta[aid]),
                                     TEMPLATE, votos, cabecera, croot)
        return aid, ns, None
    except Exception as e:
        return aid, 0, repr(e)
```

Toma una acta end-to-end: rasteriza **en memoria** (sin PNG), recorta y guarda sus crops. El `try/except Exception` ancho es deliberado: un PDF corrupto devuelve `(aid, 0, repr(e))` y se registra como error, pero **no tumba la corrida entera** de 5000 actas. El driver (`:275-295`):

```python
NPROC = os.cpu_count() or 2
for split, sids in splits.items():
    croot = DATA / f"crops_{split}"
    done_file = DATA / f"procesadas_{split}.txt"
    hechas = set(done_file.read_text().split()) if done_file.exists() else set()
    pend = [a for a in sids if a not in hechas]
    if pend:
        with get_context("fork").Pool(NPROC) as pool, open(done_file, "a") as marca:
            tareas = pool.imap_unordered(partial(procesa_acta, croot=croot), pend)
            for aid, ns, err in tqdm(tareas, total=len(pend), desc=split):
                saved += ns
                if err:
                    errores.append((aid, err))
                else:
                    marca.write(aid + "\n"); marca.flush()
    n_rows = build_manifest(croot, DATA / f"manifest_{split}.csv")
```

Tres mecanismos clave:
- **Paralelismo**: `get_context("fork").Pool(NPROC)` con `imap_unordered`. El `fork` es importante (ver gotchas) y `imap_unordered` permite consumir resultados a medida que llegan, no en orden, lo que maximiza throughput. `partial(procesa_acta, croot=croot)` fija el directorio de salida por split (los workers solo reciben `aid`).
- **Reanudabilidad**: `procesadas_<split>.txt` es un log append-only de IDs ya hechos. Al re-correr, `hechas` se lee de ahi y `pend` filtra lo que falta. Cada acta exitosa se escribe con `marca.write(aid + "\n"); marca.flush()` *inmediatamente* — el `flush()` asegura que el progreso sobrevive aunque la celda se interrumpa a mitad. Solo se marca como hecha si `err is None`, asi una acta fallida se reintenta en la proxima corrida.
- **Manifests por split**: `build_manifest` recorre `crops/<label>/*.png` y escribe el CSV `(path, label)`.

Tras el loop se imprimen cuantos crops nuevos, cuantas actas ya estaban, filas del manifest y los primeros 5 errores.

*El render en memoria.* `procesa_acta` llama `rasterize_acta` (inline-ada desde `PREPROCESS`, `_inline_code.py:28-40`): abre el PDF con PyMuPDF, auto-rota landscape→portrait, escala a tamano fijo `2339x3309` con `fitz.Matrix`, y devuelve un `PIL.Image` en gris construido directo desde `pix.samples` con `Image.frombytes(...)`. **Nunca escribe un PNG.** El comentario (`:30-32`) cuantifica: el encode+write+decode de un PNG de 7.7Mpx cuesta ~3/4 del tiempo por acta (0.75s de 0.97s medidos en M2) y el archivo se borraria igual; los pixeles son identicos al PNG (verificado byte a byte). El markdown de la celda (`:231-244`) repite esto para el alumno.

*Empaquetado + subida del bundle* (celda en `:328-342`):

```python
bundle = WORK / "crops_bundle.tar.gz"
with tarfile.open(bundle, "w:gz") as t:
    for split in ("train", "val", "test"):
        t.add(DATA / f"crops_{split}", arcname=f"data/crops_{split}")
        t.add(DATA / f"manifest_{split}.csv", arcname=f"data/manifest_{split}.csv")
...
if SUBIR_A_HF:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=str(bundle), path_in_repo="crops_bundle.tar.gz",
                    repo_id=HF_DATASET_REPO, repo_type="dataset")
```

Comprime los 3 splits (crops + manifests) en un solo `tar.gz`, usando `arcname` para que al descomprimir queden bajo `data/crops_<split>` — exactamente la ruta que espera el notebook 02. La subida usa `HfApi().upload_file(...)`, que toma el `HF_TOKEN` ya puesto en el entorno por la celda de config. Si `SUBIR_A_HF=False`, el bundle queda local.

**Decisiones de implementacion (el porque)**

- **Generar el notebook desde codigo (no editarlo a mano).** El `.ipynb` es un derivado; la fuente de verdad es `build_notebooks.py` + `_inline_code.py`. Esto evita el churn de git de Colab (re-guardar reordena celdas y regenera IDs) — por eso los IDs son deterministas (`c{i:02d}`) y la badge "Open in Colab" se inserta nosotros mismos (`:41-49`), para que el save de Colab no la re-agregue.
- **Universo via `list_repo_files`, no via parquet.** El parquet describe el universo completo de ONPE (~84k), pero solo se subieron 5000 PDFs. Filtrar por lo *fisicamente presente* en HF garantiza que cada `snapshot_download` encuentre su archivo (cero 404).
- **`snapshot_download` en una pasada vs 5000 descargas.** 5000 llamadas a `hf_hub_download` generarian 5000 barras de progreso que congelan el front-end de Colab. Una sola pasada con `allow_patterns` = una barra. Con token, ademas, esquiva el rate-limit de descargas anonimas.
- **Render en memoria.** El PNG intermedio era ~3/4 del costo por acta y se borraba de inmediato. Eliminarlo no cambia un pixel y casi cuadruplica el throughput.
- **`fork` y no `spawn`.** Con `fork`, los workers heredan los DataFrames `votos`/`cabecera` y `TEMPLATE` ya cargados en memoria del proceso padre — sin re-serializar ni re-leer parquets por worker. `spawn` reimportaria todo. (Trade-off: ver gotchas.)
- **CPU-only por diseno.** Pedir GPU para un notebook que no la usa es la causa probable del "Runtime disconnected" historico: Colab mata runtimes con GPU ociosa, y el render secuencial de 5000 actas dura lo suficiente para gatillarlo. Quitar la GPU del metadata es la fix.
- **Validacion del token al inicio, no al final.** Un fallo de upload tras 40 min de render es la peor UX posible. La config falla rapido si falta el token y `SUBIR_A_HF=True`.

**Sutilezas / gotchas**

- **`get_token()` cachea y miente.** El comentario en `:182-184` advierte: `huggingface_hub.get_token()` cachea por sesion. Si lo consultas antes de habilitar el secreto en Colab, devuelve `None` para siempre esa sesion. Por eso el codigo lee el secreto directo con `userdata.get("HF_TOKEN")` y solo cae a `get_token()` como fallback de entorno local.
- **`fork` no esta en macOS/Windows por default.** El loop usa `get_context("fork")` explicitamente. En Colab (Linux) `fork` esta disponible; correr este *notebook* tal cual en local Mac fallaria (el default es `spawn`). Como el destino es Colab, es correcto, pero es una atadura a la plataforma.
- **Reanudabilidad solo dentro de la misma VM.** `procesadas_<split>.txt` vive en `/content` (efimero). Si Colab recicla la VM, `/content` se pierde y la corrida empieza de cero — el markdown lo dice explicito (`:241-242`). No es un checkpoint en HF; es continuidad intra-sesion.
- **`REHACER_DESDE_CERO` es obligatorio al cambiar la deteccion.** Sin el flag, las actas ya en `procesadas_*.txt` se saltan y publicarias un bundle mezclado (crops del metodo viejo + nuevo). El comentario en config (`:175-176`) y el markdown (`:158-162`) insisten en esto.
- **El `try/except` ancho en `procesa_acta` es intencional, no pereza.** Captura `Exception` para que un PDF malo (404, corrupto, sin label) no aborte 4999 actas buenas. Los errores se acumulan y se reportan al final; los primeros 5 se imprimen.
- **Naming: `rasterize_acta` (inline) vs `rasterize_first_page` (paquete).** La logica es identica byte-a-byte (`src/actas_cnn/render.py:25` vs `_inline_code.py:28`), solo cambia el nombre. La orquestacion del loop/bundle/seleccion de universo **solo existe en el notebook builder** — el paquete provee las primitivas por-acta (`render.py`, `preprocess/crops.py`), no el driver Colab.
- **El etiquetado ink-aware ocurre *dentro* de `build_crops_for_acta`**, no en esta celda. La orquestacion solo invoca; el remapeo de actas "corridas" (`remapeo_ink_aware`) vive en `LABELS_BUILD`. Desde la perspectiva del loop, es transparente.

**Para la presentacion**

- "El notebook 01 es **generado por codigo**: editamos bloques Python en el repo y un script los ensambla con `nbformat`. Eso garantiza que el entregable Colab y el paquete `actas_cnn` nunca se desincronicen — misma logica, validada contra el baseline."
- "El 01 corre en **CPU a proposito**. Solo usa PyMuPDF/PIL; la GPU jamas se toca. Pedir GPU para esto hacia que Colab desconectara el runtime por GPU ociosa a mitad del render de 5000 actas — un bug operativo que costaba horas."
- "El loop de render es **paralelo (un proceso por core via `fork`), reanudable (log append-only `procesadas_<split>.txt` con `flush` por acta) y tolerante a fallos (un PDF corrupto no tumba la corrida)**. Y rasteriza **en memoria, sin PNG intermedio**, lo que ahorra ~3/4 del tiempo por acta sin cambiar un pixel."
- "El universo es **reproducible**: se selecciona de los PDFs *realmente publicados* en HF (no del parquet completo, que pediria archivos inexistentes), se baraja con **semilla 42** y se parte 70/15/15 **por acta** para que no haya leakage entre train/val/test."
- "El acople entre los dos notebooks es **solo el bundle de datos** (`crops_bundle.tar.gz` en HF), no el codigo. El 01 publica; el 02 lo baja en segundos y entrena. Eso deja iterar el preprocesamiento sin tocar el modelo."

---

## 4. El modelo: ResNet-18 estilo CIFAR (notebook 02)

**Que hace** — Este módulo define la red neuronal que es el corazón del proyecto: el clasificador que, dada una imagen de **una sola celda de un dígito manuscrito** (32×32 píxeles, en escala de grises), predice cuál de los **10 dígitos** (0–9) es. No clasifica actas enteras ni campos completos; clasifica *un dígito a la vez*. La reconstrucción del voto entero (concatenar `[1, 8]` → `18`) y la suma por partido ocurren después, en el módulo de evaluación. El modelo vive en `src/actas_cnn/model.py` y se construye con `build_model(arch="resnet18", ...)` (`model.py:82`), que delega en `resnet18_cifar` (`model.py:64`). Junto a él conviven dos baselines metodológicos de Semana 1 — `LeNetCNN` (`model.py:12`) y `DeepCNN` (`model.py:45`) — que ya no son el modelo oficial pero se conservan para reproducibilidad y para tener una línea de comparación.

El truco central del módulo es que **no se escribe una ResNet desde cero**: se toma la `resnet18` ya implementada y probada de `torchvision`, y se la *parchea* en dos puntos para adaptarla a imágenes chiquitas. Eso da la mejor relación esfuerzo/calidad: aprovechamos una arquitectura canónica (He et al., 2015, ~11.17M de parámetros) bien testeada, cambiando solo lo imprescindible.

**Explicacion del codigo** — La función clave son seis líneas (`model.py:64-79`):

```python
def resnet18_cifar(in_channels=1, num_classes=10):
    m = _torchvision_resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m
```

Recorrido bloque por bloque:

- **`m = _torchvision_resnet18(num_classes=num_classes)`** (`model.py:75`). Instancia la ResNet-18 estándar de torchvision. El import en `model.py:9` la renombra a `_torchvision_resnet18` (el guion bajo señala "uso interno, no es nuestra API"). Con `num_classes=10`, torchvision ya construye internamente la última capa como `Linear(512, 10)` — esa es la capa `fc` que mapea las 512 features finales a las 10 clases de dígitos, así que **no la tocamos**. La ResNet original esperaba 1000 clases (ImageNet); pasarle `num_classes=10` la ajusta sin esfuerzo.

- **El "stem" o `conv1`** (`model.py:76-77`). Aquí está el primer parche. La ResNet de torchvision viene diseñada para ImageNet (imágenes de 224×224), donde el stem es un `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`: un kernel grande de 7×7 que avanza de a 2 píxeles (`stride=2`). En una imagen de 224×224 eso reduce la resolución a la mitad de entrada — está bien, sobra resolución. Pero nuestras imágenes son de **32×32**: aplicar `stride=2` ahí *tira a la basura la mitad de la información espacial en el primer paso*. Lo reemplazamos por `Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)`: kernel chico de 3×3, `stride=1` (no reduce) y `padding=1` (que con kernel 3 mantiene exactamente el mismo alto/ancho de salida que de entrada). Resultado: la imagen sale del stem todavía a 32×32. Notar `in_channels=1`: la ResNet original espera 3 canales (RGB); como trabajamos en gris, ponemos 1 canal de entrada. Y `bias=False` porque la capa siguiente es una BatchNorm, que tiene su propio término aditivo: el bias de la conv sería redundante.

- **`m.maxpool = nn.Identity()`** (`model.py:78`). Segundo parche. La ResNet de ImageNet, después del stem, mete un `MaxPool2d(3, stride=2)` que vuelve a partir la resolución a la mitad. Con 224×224 eso es razonable (224→112→56). Con 32×32 sería catastrófico: la imagen es tan chica que se nos desintegra antes de que las etapas residuales puedan "ver" el dígito. `nn.Identity()` es una capa que *no hace nada* (devuelve su entrada tal cual): así "salteamos" el maxpool sin tener que reescribir el `forward` de la ResNet, que sigue llamando a `self.maxpool(x)` pero ahora es un no-op.

- **Lo que NO tocamos — las 4 etapas residuales.** Después del stem, la ResNet-18 tiene 4 etapas (`layer1..layer4` en torchvision), cada una con 2 bloques residuales (BasicBlock), con canales **64 → 128 → 256 → 512**. Cada etapa salvo la primera reduce la resolución a la mitad (stride 2 en su primer bloque) mientras duplica los canales. Partiendo de 32×32 a 64 canales: layer1 deja 32×32×64, layer2 → 16×16×128, layer3 → 8×8×256, layer4 → 4×4×512. **Esto es exactamente lo que queremos**: la reducción de resolución la hacen las etapas residuales gradualmente (4 pasos suaves), no el stem de un saque.

- **El final — GAP y `Linear(512,10)`.** Tras `layer4` la ResNet aplica un `AdaptiveAvgPool2d((1,1))` (Global Average Pooling): convierte el mapa de 4×4×512 en un vector de 512 promediando cada canal sobre su grilla espacial. Luego `fc = Linear(512, 10)` produce los 10 logits. Ambos vienen "gratis" de torchvision; no los parcheamos.

**`build_model`** (`model.py:82-89`) es solo un *dispatcher*: un `if`/`elif` que mapea la cadena `arch` a la clase/función correspondiente (`"lenet"` → `LeNetCNN`, `"deep"` → `DeepCNN`, `"resnet18"` → `resnet18_cifar`) y lanza `ValueError` si la arquitectura es desconocida (`model.py:89`). Es el único punto de entrada que usa el resto del código (`train.py`, notebooks), lo cual permite cambiar de arquitectura pasando un flag sin tocar nada más.

**Sobre qué son los bloques residuales y por qué importan** (lo que NO está en el código pero hay que entender, porque es la esencia de "ResNet"): un bloque residual calcula `salida = F(x) + x`, donde `F(x)` son un par de convoluciones. Ese `+ x` es la **skip connection** (o conexión de atajo): en vez de pedirle a las capas que aprendan la transformación completa, les pedimos que aprendan solo el **residuo** (la *corrección* respecto a la entrada). Importa por dos razones: (1) si la transformación ideal es "casi la identidad", aprender un residuo ≈0 es trivial, mientras que aprender la identidad con convoluciones es sorprendentemente difícil; (2) durante el backprop, el gradiente fluye por el atajo "+x" sin atenuarse, lo que **evita el problema del gradiente que se desvanece** y permite entrenar redes profundas (18, 50, 100+ capas) de forma estable. Esa es la idea de He et al. (2015) que ganó ImageNet y por la que ResNet sigue siendo la arquitectura de referencia.

**El bloque inline (notebook 02).** En `tools/_inline_code.py:237-248`, el bloque `MODEL` contiene `resnet18_cifar` con **la misma lógica idéntica**: mismo parche de `conv1` (3×3, stride 1, padding 1), mismo `maxpool = nn.Identity()`, mismo `in_channels`. La única diferencia es de *empaquetado*, no de comportamiento: el notebook **solo inline-a `resnet18_cifar`** (no copia `LeNetCNN`, `DeepCNN` ni el dispatcher `build_model`), porque el entregable corre exclusivamente la arquitectura oficial y no necesita los baselines ni el selector. El docstring inline ya documenta el conteo "11.17M params" directamente. Además, en el bloque `TRAIN` (`_inline_code.py:328`) el modelo se instancia llamando `resnet18_cifar(1, 10)` directo, sin pasar por `build_model`. Lógica equivalente, superficie reducida.

**Decisiones de implementacion (el porque)** —

- **Parchear torchvision en vez de escribir la ResNet a mano.** Reescribir BasicBlocks, downsample, BatchNorm e inicialización es propenso a errores sutiles que degradan la accuracy sin avisar. Tomar la implementación canónica y cambiar 2 atributos (`conv1`, `maxpool`) es DRY, auditable y reproduce *exactamente* la arquitectura de referencia. El docstring (`model.py:65-73`) deja explícito qué se parchea y qué se mantiene.

- **"Estilo CIFAR" no es un capricho.** Es una adaptación estándar y bien documentada: las ResNets para datasets de imágenes chicas (CIFAR-10 es 32×32, igual que nuestros crops) usan precisamente stem 3×3 stride 1 y sin maxpool inicial. No estamos inventando; estamos aplicando la receta conocida para el tamaño de imagen que tenemos.

- **Preservar resolución temprano.** El razonamiento de fondo: en imágenes grandes sobra resolución y reducir rápido ahorra cómputo; en 32×32 cada píxel cuenta. Un dígito manuscrito en 32×32 ya es poca información — si el stem y el maxpool nos dejan en 8×8 antes de la primera etapa residual, la red nunca llega a "ver" bien los trazos. Por eso entramos a las etapas residuales todavía a 32×32.

- **ResNet-18 sobre la CNN custom.** El baseline `DeepCNN` (la CNN custom de Semana 1: Conv+BN+LeakyReLU+Dropout) ya alcanzaba 97.77% digit-level. ResNet-18 sube a 98.12% (+0.35pp) *solo cambiando la arquitectura*, sin cambiar datos ni receta. La mejora se atribuye limpiamente a las skip connections y la profundidad. Mantener `DeepCNN` y `LeNetCNN` accesibles vía `build_model` permite defender esa comparación de forma reproducible.

**Sutilezas / gotchas** —

- **El bug de MPS (pytorch#96056) y por qué el GAP `(1,1)` lo evita.** En la Mac M2 del usuario, PyTorch corre sobre MPS (Metal). Hay un bug conocido: `AdaptiveAvgPool2d` con dimensiones de salida que *no dividen exactamente* la entrada produce resultados erróneos o crashea en MPS. La ResNet usa `AdaptiveAvgPool2d((1,1))` — salida 1×1 — y **1 siempre divide cualquier tamaño de entrada**, así que es el caso seguro y nunca dispara el bug. Esto NO es accidente: mirá el contraste con los baselines. En `LeNetCNN`, el comentario en `model.py:23-24` documenta que tuvieron que elegir un pool `(3,3)` *a propósito* porque con la geometría de LeNet (features 6×6) el 3 divide al 6; ahí el bug fue una restricción de diseño real. La ResNet se libra del problema "gratis" por usar GAP `(1,1)`. Es un buen punto para la presentación: la arquitectura oficial fue elegida también por ser robusta al hardware del proyecto.

- **`m.maxpool = nn.Identity()` confunde a primera vista.** Uno espera ver una red reconstruida; en cambio se *muta* un atributo de un objeto ya creado. Funciona porque el `forward` de la ResNet de torchvision invoca `self.maxpool(x)` por nombre: al reemplazar ese atributo por una identidad, el flujo sigue igual pero ese paso no hace nada. Es PyTorch idiomático (las capas son atributos reasignables), pero hay que saberlo para no buscar un `forward` custom que no existe.

- **`bias=False` en el stem.** Fácil de pasar por alto: no es un olvido. La conv va seguida de BatchNorm, que resta la media y aprende su propio shift (`beta`); un bias en la conv sería matemáticamente redundante y desperdicia parámetros. Es la convención de toda ResNet.

- **La entrada es 32×32 aunque los crops originales no lo sean.** El modelo asume 32×32, pero los recortes de celdas tienen tamaños variables. La uniformización la hace el pipeline de transforms (`data.py:48-49`, `transforms.Resize((32, 32))`), no el modelo. El modelo confía en que el `Dataset` ya entregó tensores `1×32×32` normalizados a media 0.5.

- **`num_classes` solo afecta la capa `fc`.** Pasar `num_classes=10` no cambia las etapas residuales; solo dimensiona el `Linear` final a 10 salidas. Si mañana hubiera que clasificar otra cosa, ese es el único punto que cambia.

**Para la presentacion** —

- "Nuestro modelo es una **ResNet-18 (He et al., 2015)** adaptada al estilo CIFAR. No la escribimos desde cero: partimos de la implementación canónica de torchvision y le cambiamos **solo dos cosas** para imágenes chicas de 32×32." (Es la frase de apertura más sólida: rigor + eficiencia.)

- "**Las skip connections son la idea clave de ResNet**: cada bloque aprende un *residuo* `F(x)+x` en vez de la transformación completa. Eso deja pasar el gradiente sin desvanecerse y permite entrenar redes profundas de forma estable — por eso ResNet ganó ImageNet y sigue siendo el estándar."

- "Los **dos parches** son: (1) el stem pasa de un kernel 7×7 con stride 2 a un **3×3 con stride 1**, y (2) **eliminamos el MaxPool inicial** (`nn.Identity()`). Ambos cambios **preservan la resolución** al inicio: en una imagen de 32×32 no nos podemos dar el lujo de tirar la mitad de los píxeles en el primer paso, como sí hace la versión de ImageNet."

- "Las **4 etapas residuales se quedan intactas** (canales 64→128→256→512), igual que el **Global Average Pooling y el `Linear(512,10)`** final. La reducción de resolución la hacen las etapas, gradualmente — no el stem de golpe. Total: **~11.17M de parámetros**."

- "El **GAP `(1,1)`** no es solo arquitectura: nos **evita un bug real de PyTorch en MPS** (issue #96056) que rompe el AdaptiveAvgPool con dimensiones no divisibles en la Mac M2. El 1 siempre divide, así que es el caso seguro — un punto donde la decisión de arquitectura y la restricción de hardware coinciden."

- (Si preguntan por baselines) "Conservamos una **CNN custom (97.77%)** y una **LeNet** como líneas de referencia; ResNet-18 sube a **98.12% a nivel dígito** solo por cambiar la arquitectura, manteniendo todo lo demás igual."

---

## 5. Dataset, transforms y entrenamiento: la receta ls_ra_mu_cos (notebook 02)

**Que hace** — Este modulo es el corazon del notebook 02 (`02_modelo_colab.ipynb`): toma los crops de digitos ya recortados (un PNG por celda, organizados en `crops_<split>/<label>/*.png`) y los convierte en un dataset PyTorch, los pasa por un pipeline de transforms, y entrena la ResNet-18. La fuente de verdad es `src/actas_cnn/data.py` (clase `CropsDataset`, funcion `default_transforms`) y `src/actas_cnn/training.py` (`train_model`/`main`, `run_epoch`, `_mixup_batch`); los bloques `DATASET` y `TRAIN` de `tools/_inline_code.py` (`_inline_code.py:251` y `:280`) inlinean exactamente la misma logica para que el notebook sea autonomo (sin clonar el repo).

El bloque entrena con una receta configurable por flags. La receta "base" (defaults) reproduce el checkpoint oficial `resnet18_best.pt`; la receta ganadora de las ablations, **`ls_ra_mu_cos`**, activa los cuatro ingredientes a la vez: **l**abel **s**moothing + **Ra**ndAugment + **mu**xup + **cos**ine LR. Esa receta domina todas las metricas (acta-level +1.88pp segun el `CLAUDE.md`).

**Explicacion del codigo**

*Lectura del manifest y carga 1x32x32 (`CropsDataset`, `data.py:60`).* El dataset no guarda imagenes: guarda un DataFrame con dos columnas `path,label` leido del manifest CSV.

```python
class CropsDataset(Dataset):
    def __init__(self, manifest_csv, root=".", transform=None):
        self.df = pd.read_csv(manifest_csv)        # data.py:62
        self.root = Path(root)
        self.transform = transform or default_transforms()
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["path"])  # data.py:71 — abre el PNG bajo demanda
        x = self.transform(img)                    # -> tensor 1x32x32
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y
```

`__getitem__` (`data.py:69`) abre el PNG **bajo demanda** (lazy: solo se carga al pedir el indice), aplica el transform y devuelve `(x, y)`. El label sale del nombre de la carpeta, no de la imagen: el manifest se construye en `build_manifest` (`data.py:16`) recorriendo `crops/<label>/*.png` y escribiendo `(ruta_relativa, label)`; la ruta es **relativa a `crops_dir`** (`data.py:30`), asi la misma manifest funciona venga de local, de HF o de un mirror — solo cambia `root`.

*Transforms (`default_transforms`, `data.py:38`).* Construye una lista de operaciones segun `train` y `randaugment`:

```python
ops = [transforms.Grayscale(num_output_channels=1),
       transforms.Resize((image_size, image_size))]   # -> 1x32x32
if train:
    ops.append(transforms.RandomAffine(degrees=8, translate=(0.1,0.1), scale=(0.9,1.1)))
    if randaugment:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
ops += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
```

- `Grayscale(1)` + `Resize(32,32)`: garantiza **1 canal, 32x32** (la entrada que espera la ResNet-18 CIFAR), sea cual sea el tamano/modo del crop original.
- En `train=True` siempre se agrega `RandomAffine` (augmentation suave: rotacion ±8°, traslacion 10%, escala 0.9-1.1) — un digito manuscrito sigue siendo el mismo digito si rota un poco o se desplaza.
- `RandAugment(num_ops=2, magnitude=9)` (los defaults del paper) se agrega **solo si `randaugment=True`**; es augmentation mas agresiva.
- En `train=False` (eval) **no hay augmentation**: solo resize + normalize.
- `ToTensor` lleva los pixeles a `[0,1]`; `Normalize((0.5,),(0.5,))` los recentra a `[-1,1]` (`(x-0.5)/0.5`).

*Holdout interno 80/20 (`training.py:89`, `_inline_code.py:320`).* Aqui hay una sutileza importante de design: el split train/val/test del proyecto (3500/750/750 actas, sin leak) se hace en otro lado; **dentro del entrenamiento se reserva ademas un holdout interno** del propio set de train para seleccionar el mejor epoch:

```python
n_val = max(1, int(0.2 * len(full)))
train_set, val_set = random_split(full, [len(full) - n_val, n_val])
```

`random_split` baraja y corta 80/20. Ese `val_acc` del holdout es lo que decide cual checkpoint guardar — **no** es el val-split oficial de 750 actas (ese se mide despues con `scripts/evaluate.py`). El loader de train usa `shuffle=True`, el de val `shuffle=False` (`training.py:93-96`).

*Setup del optimizador y la receta (`training.py:98-102`, `_inline_code.py:328-331`).*

```python
model = build_model(args.arch, ...).to(device)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # componente 1
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.lr)                # lr=5e-4
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine_lr else None  # componente 4
```

Hiperparametros fijos en `config.py` (`TrainConfig`, `config.py:18`): `batch_size=128`, `epochs=20`, `lr=5e-4`, `seed=42`, `image_size=32`, `num_classes=10`, `in_channels=1`. La semilla se fija con `torch.manual_seed(TRAIN.seed)` (`training.py:83`) para reproducibilidad.

*El loop de epoch (`run_epoch`, `training.py:32`).* Una sola funcion sirve train y eval: si `optimizer is None` esta en modo eval (`is_train=False`), si no, entrena.

```python
is_train = optimizer is not None
model.train(is_train)
with torch.set_grad_enabled(is_train):     # no calcula gradientes en eval
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train and mixup_alpha > 0:
            x_m, y_a, y_b, lam = _mixup_batch(x, y, mixup_alpha)   # componente 3
            out = model(x_m)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            correct += (out.argmax(1) == y_a).sum().item()
        else:
            out = model(x); loss = criterion(out, y)
            ...
            correct += (out.argmax(1) == y).sum().item()
```

*Mixup — la matematica (`_mixup_batch`, `training.py:21`; `_mixup`, `_inline_code.py:285`).*

```python
def _mixup_batch(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)   # permutacion del batch
    x_mixed = lam * x + (1 - lam) * x[idx]              # combinacion convexa de imagenes
    return x_mixed, y, y[idx], lam
```

La idea (Zhang et al., 2017): se mezcla cada imagen con **otra imagen al azar del mismo batch** (`x[idx]` es el batch permutado). El peso `lam ∈ [0,1]` sale de una distribucion Beta(α,α); con α=0.2 la Beta es bimodal (favorece `lam` cerca de 0 o 1, mezclas suaves). La imagen mezclada es `lam·x + (1-lam)·x[idx]`. Como la entrada es mezcla de dos clases (`y_a=y`, `y_b=y[idx]`), la loss tambien se mezcla **linealmente con el mismo `lam`**:

```python
loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)   # training.py:42
```

Por que se ve `train_acc` baja con mixup: el modelo recibe imagenes que son superposicion de, por ejemplo, un "3" y un "7", pero la accuracy de train se mide contra **solo `y_a`** (el target dominante) — ver el comentario explicito en `training.py:43-45`: *"Para tracking de accuracy en train con mixup, usamos y_a... Es una aproximacion; el numero real comparable es val_acc."* Cuando `lam` es cercano a 0.5, ni `y_a` ni `y_b` describen bien la imagen, asi que `argmax(out)==y_a` falla mucho aunque el modelo este aprendiendo bien. **La metrica honesta es `val_acc` del holdout** (donde no hay mixup). Esto es esperado, no un bug.

*Cosine LR (`training.py:120-121`).* Tras cada epoch, `scheduler.step()` baja el learning rate siguiendo media onda de coseno desde `lr=5e-4` hasta ~0 en `T_max=epochs`. Arranca agresivo (exploracion) y termina fino (convergencia suave).

*Guardar solo el mejor por `val_acc` del holdout (`training.py:122-126`, `_inline_code.py:338-342`).*

```python
if va_acc > best_acc:
    best_acc = va_acc
    torch.save({"model": model.state_dict(), "acc": best_acc, "arch": args.arch}, ckpt)
```

Solo se persiste el checkpoint cuando el `val_acc` del holdout **supera** el mejor visto. En el notebook (`_inline_code.py:340-345`) se guarda ademas `best_state = copy.deepcopy(model.state_dict())` y al final se hace `model.load_state_dict(best_state)`, de modo que la funcion devuelve el modelo del **mejor epoch**, no el del ultimo. Esto es early-stopping implicito (model selection): si el epoch 20 sobreajusta, te quedas con el epoch 14 si fue mejor.

**Decisiones de implementacion (el porque)**

- **Manifest CSV con rutas relativas** en vez de un dataset binario: portabilidad total entre local/HF/Colab; el `root` desacopla *donde estan los bytes* de *que crop es*. Y permite inspeccionar/auditar el dataset con `pandas` sin codigo especial.
- **Carga lazy de PNGs** (`Image.open` en `__getitem__`): 106k+22k+22k crops no caben comodos en RAM como tensores; abrir bajo demanda mantiene la huella baja y deja que los `num_workers` paralelicen la I/O.
- **Receta por flags con defaults = base**: el comentario en `_inline_code.py:313-316` es la clave de la decision: los defaults reproducen el checkpoint oficial `resnet18_best.pt` y son rapidos (~5-8 min en T4). RandAugment "corre en CPU por imagen y es el cuello de botella en Colab", asi que `ls_ra_mu_cos` (los 4 ingredientes juntos) se reserva **solo para la ablacion**, no como default. Permitir cada componente por separado (`--label-smoothing`, `--randaugment`, `--mixup`, `--cosine-lr`) fue lo que habilito la comparativa de ablations limpia.
- **Adam lr=5e-4**: Adam por robustez sin tunear (converge bien sin barrer LR a mano); 5e-4 es conservador y estable para 20 epochs.
- **Holdout interno random_split** en vez de usar el val oficial para seleccionar epoch: separa estrictamente *seleccion de modelo* (holdout barato dentro de train) de *reporte de metricas* (val/test oficiales con `evaluate.py`). Evita filtrar el set de evaluacion en la eleccion del checkpoint.
- **`run_epoch` unica para train y eval** (DRY): el unico discriminador es si llega `optimizer`. `torch.set_grad_enabled(is_train)` apaga el grafo de gradientes en eval (memoria + velocidad).

**Sutilezas / gotchas**

- **El `val_acc` del entrenamiento NO es la metrica oficial.** Es un holdout interno del 20% de train via `random_split`, usado solo para elegir checkpoint. Las metricas que se reportan (digit 98.12%, field 98.87%, acta-level 90.33%) salen de `scripts/evaluate.py --split val` sobre las 750 actas de val oficiales, que `run_epoch` nunca ve.
- **`train_acc` baja con mixup es normal, no es bug** (ver explicacion arriba): se compara contra `y_a` mientras la imagen es mezcla.
- **Doble paso del optimizador en la version del paquete (sutil pero correcto):** en `training.py`, la rama no-mixup hace `zero_grad/backward/step` en las lineas 50-53, y hay un bloque separado `if is_train and mixup_alpha > 0` (lineas 55-58) que hace el step para la rama mixup. La estructura confunde a primera vista, pero **no hay doble step**: cada rama da exactamente un step. El bloque `TRAIN` del notebook (`_inline_code.py:305-306`) lo reescribe mas limpio (`if is_train: optimizer.zero_grad(); loss.backward(); optimizer.step()` una sola vez al final, comun a ambas ramas) — **misma logica, codigo mas claro**.
- **`random_split` sin generador con semilla fija explicito**: `torch.manual_seed(seed)` se llama antes, asi que el split es reproducible globalmente, pero no se pasa un `generator=` dedicado a `random_split`. En la practica reproduce, pero es algo a saber si se reordena el codigo.
- **El checkpoint oficial es sensible a la geometria del render** (nota del `CLAUDE.md`): `resnet18_best.pt` fue entrenado con crops de mayo (right-justified); con crops de geometria nueva su digit-level baja ~2pp por mismatch de render, no por las labels.
- **`num_workers`**: el paquete usa `num_workers=2` fijo (`training.py:93`); el notebook usa `min(4, os.cpu_count())` con `persistent_workers` (`_inline_code.py:323-327`) — diferencia de rendimiento, no de logica.

**Para la presentacion**

- "El dataset es solo un CSV `path,label`: las imagenes se cargan bajo demanda y se normalizan a 1 canal, 32x32, rango [-1,1]. El label viene de la carpeta, no de la imagen."
- "La receta ganadora `ls_ra_mu_cos` son cuatro regularizadores que atacan el sobreajuste por vias distintas: **label smoothing** (no exige confianza 100%, evita sobreconfianza), **RandAugment** + **RandomAffine** (mas variedad de digitos), **mixup** (interpola pares de imagenes y sus labels con el mismo `lam` de una Beta), y **cosine LR** (baja el learning rate en onda de coseno para converger fino)."
- "Mixup en una frase: entrada = `lam·imgA + (1-lam)·imgB`, loss = `lam·CE(·,yA) + (1-lam)·CE(·,yB)`. Por eso la `train_acc` se ve baja —comparamos contra una sola clase mientras la imagen es mezcla— pero la metrica real es el `val_acc`."
- "Guardamos solo el mejor checkpoint segun un holdout interno del 20% de train (model selection / early stopping). Las metricas oficiales se miden aparte, sobre las 750 actas de val que el entrenamiento nunca toca: separamos seleccion de modelo de reporte de resultados."
- "Defaults = receta base que reproduce el modelo oficial; `ls_ra_mu_cos` se activa solo para la ablacion porque RandAugment corre en CPU y es el cuello de botella en Colab. Eso nos permitio comparar ingrediente por ingrediente de forma limpia."

---

## 6. Evaluación y métricas: de dígitos a votos vs ONPE (notebook 02)

**Qué hace** — Este módulo es la última etapa del pipeline: toma la CNN ya entrenada y mide qué tan bien lee actas *de verdad*, no dígitos sueltos. El modelo solo sabe clasificar imágenes de 32×32 en una de 10 clases (0-9); pero un acta electoral no es eso: es un documento con 42 campos numéricos (38 partidos + votos en blanco + nulos + impugnados + total de ciudadanos que votaron), cada uno compuesto por varias celdas-dígito. `evaluate.py` cierra esa brecha. Predice cada celda, **reagrupa** las celdas por (acta, campo, posición), **reconstruye** el entero de cada campo concatenando dígitos, y lo **compara contra el ground truth oficial de ONPE** que vive en los parquets. De ahí salen las cuatro métricas que reporta el proyecto y que importan para el alcance real (contar votos), más la matriz de confusión 10×10 y precision/recall/F1 por clase.

La lógica es **idéntica** en el paquete (`src/actas_cnn/evaluate.py`, vía `scripts/evaluate.py`) y en el notebook Colab 02 (bloques `EVAL` y `METRICS` de `tools/_inline_code.py`). El paquete es la fuente de verdad; el notebook inline-a las mismas funciones para ser autónomo. Hay diferencias menores de empaquetado que anoto abajo (`real_value_for` vs `field_value_for` es el mismo cuerpo con otro nombre; el paquete genera PNGs de visualización y el notebook usa `plt.show()`).

**Explicación del código** — El flujo, en orden, dentro de `main()` (`evaluate.py:90`) o `evaluate_split()` en el notebook (`_inline_code.py:368`):

1. **Cargar ground truth y template** (`evaluate.py:108-115`). Se leen tres parquets de ONPE: `actas_archivos` (mapea `archivoId`→`idActa`), `actas_votos` (un row por partido/categoría con su `nvotos`) y `actas_cabecera` (el `totalVotosEmitidos` oficial). Se arma el diccionario `aid_to_idacta` para cruzar el ID del archivo físico con el ID lógico del acta. Del `templates.json` se extrae `field_specs = {nombre: n_digits}` — los 42 campos y cuántas celdas tiene cada uno (verifiqué: 41 campos de 3 dígitos + `total_ciudadanos` de 4).

2. **Cargar el modelo** (`evaluate.py:118-125`). `_find_checkpoint()` busca en orden `resnet18_best.pt`, `deep_best.pt`, `lenet_best.pt`. Se lee `ckpt["arch"]` para reconstruir la arquitectura correcta con `build_model(arch, 1, 10)` (1 canal de entrada, 10 clases), se cargan los pesos y `model.eval()`.

3. **Predecir todas las celdas de golpe** (`evaluate.py:127-137`). Se crea un `CropsDataset` sobre el manifest del split con `default_transforms(32, train=False)` (sin augmentation). Un `DataLoader` de `batch_size=512`, `shuffle=False` (clave: el orden debe coincidir con `ds.df` para poder pegar predicciones por índice), recorre todo bajo `torch.no_grad()`:
   ```python
   for x, _ in loader:
       all_preds.append(model(x.to(device)).argmax(1).cpu().numpy())
   df["pred"] = np.concatenate(all_preds)
   ```
   `argmax(1)` toma la clase de máxima probabilidad por celda. Ahora `df` tiene una fila por celda con `path`, `label` (verdad de la celda) y `pred`.

4. **Parsear el path para recuperar (acta, campo, posición)** — esta es la pieza que devuelve la estructura del documento. `parse_crop_path` (`evaluate.py:52-60`):
   ```python
   stem = Path(rel).stem        # '<label>/<aid>_<field>_c<pos>.png'
   parts = stem.split("_")
   aid = parts[0]               # archivoId
   pos = int(parts[-1][1:])     # 'c2' -> 2
   field = "_".join(parts[1:-1])# todo lo del medio
   ```
   El `field` se reconstruye con `"_".join(parts[1:-1])` y no con `parts[1]` **a propósito**: nombres como `partido_07` o `votos_blanco` contienen un guion bajo, así que tomar solo `parts[1]` los partiría. Se aplica a todo el DataFrame con `df["path"].apply(parse_crop_path).apply(pd.Series)` (`evaluate.py:140-142`), expandiendo las tres columnas `archivoId, field, pos`.

5. **Reconstruir el entero de cada campo** — el corazón conceptual. Se agrupa por acta (`for aid, df_acta in df.groupby("archivoId")`, `evaluate.py:146`), se filtran las actas sin ground truth o sin `totalVotosEmitidos` (`continue` en `:147-153`), y por cada uno de los 42 campos:
   ```python
   crops_field = df_acta[df_acta["field"] == fname]
   preds_by_pos = dict(zip(crops_field["pos"], crops_field["pred"]))
   pred_value = reconstruct_value(preds_by_pos)
   real_value = real_value_for(fname, votos_acta, total_real)
   ```
   `reconstruct_value` (`evaluate.py:63-73`) **concatena los dígitos de las celdas presentes en orden de posición**:
   ```python
   if not preds_by_pos: return 0
   return int("".join(str(preds_by_pos[p]) for p in sorted(preds_by_pos)))
   ```
   Si no hay ninguna celda → 0 (campo vacío). Si hay `{1:1, 2:8}` → `"18"` → `18`. El `int(...)` colapsa ceros a la izquierda accidentales. `real_value_for` (`evaluate.py:76-87`) saca el valor oficial: para `partido_NN` filtra `votos_acta` por `nposicion == NN`; para blanco/nulos/impugnados usa el mapeo a posiciones especiales `{80, 81, 82}`; para `total_ciudadanos` devuelve directamente `total_emitidos`. Si no encuentra el row, devuelve 0 (el partido sacó 0 votos y nadie escribió nada). Cada campo produce un dict con `pred`, `real`, `correct = pred==real`, `error = pred-real` (`evaluate.py:160-166`).

6. **Las cuatro métricas** (`evaluate.py:178-221`):
   - **digit-level** (`:179`): `df["pred"].eq(df["label"]).mean()` — fracción de celdas individuales correctas. La métrica más permisiva (98.12%).
   - **field-level** (`:183`): `res["correct"].mean()` — fracción de *campos enteros* correctos (98.87%). Más estricto: las 3 celdas de un partido tienen que estar todas bien (o coincidir el entero).
   - **acta-level** (`:187-189`): `res.groupby("archivoId")["correct"].all()` — un acta cuenta solo si **los 42 campos están correctos a la vez**, luego `.mean()` sobre actas (90.33%). Es la métrica reportada como la dura.
   - **reconstrucción del total agregado** (`:200-212`): se suman todos los campos *excepto* `total_ciudadanos` por acta (`sum_pred`, `sum_real`), y se compara contra esa suma real. Reporta MAE (`abs_err.mean()`, 2.40 votos), mediana, y % de actas con error 0 / ≤1 / ≤5 / ≤20. Esto es lo que de verdad le importa a una autoridad electoral: ¿cuánto se desvía el total de votos que reconstruimos?

7. **Análisis profundo** — La matriz de confusión 10×10 (`evaluate.py:229-231`) se llena celda a celda: `confusion[real, predicho] += 1`. De ahí salen recall (`confusion[i,i] / fila i`), precision (`confusion[i,i] / columna i`) y F1 por clase (`:232-236`), con `max(..., 1)` y `max(..., 1e-9)` como guardas anti-división-por-cero. Se grafica con `imshow(cmap="Blues")` anotando cada celda, un histograma de `|error|` con bins exponenciales y eje `symlog` para ver la cola larga, y un ranking de las 20 peores actas (`:291-311`) ordenadas por `|error|` del total, listando qué campos fallaron — es el material que reveló el hallazgo ink-aware (los errores se concentraban en pocas actas con escritura corrida).

En el notebook, `evaluate_split` (`_inline_code.py:368`) hace exactamente 1-5 y devuelve `(df, res)`; `report_metrics` (`:407`) hace el punto 6 en forma compacta y retorna un dict; `confusion_and_prf` (`:428`) y `ablations_table` (`:453`) hacen el 7. `ablations_table` lee varios CSVs `evaluate_val_*.csv` (uno por variante de entrenamiento) y reconstruye field/acta-level + MAE de cada uno para la tabla comparativa.

**Decisiones de implementación (el porqué)** —

- **Predecir todo y luego agrupar, no acta por acta.** Se corre el modelo sobre el dataset entero en batches de 512 (un solo pase GPU, eficiente) y la estructura del documento se recupera *después*, parseando paths. Alternativa descartada: cargar acta por acta y predecir sus ~130 celdas — mata el throughput de la GPU con batches diminutos.
- **La estructura vive en el nombre del archivo, no en una base de datos.** `<aid>_<field>_c<pos>.png` codifica acta/campo/posición en el path. Es un acoplamiento deliberado entre el preprocesamiento (que nombra los crops) y la evaluación (que los parsea). Mantiene la evaluación *stateless* y reproducible: el manifest CSV + los PNGs bastan.
- **Reconstruir por concatenación de celdas presentes, no por valor posicional fijo.** La versión vieja asumía posiciones fijas (celda 3 = unidades) y rellenaba con ceros a la izquierda. La actual (`reconstruct_value`) concatena en orden de lectura las celdas *que existen*. Esto es lo que hace el sistema robusto al fix **ink-aware**: en las ~3% de actas con escritura corrida, los labels se remapean a las celdas donde realmente cae la tinta, y la posición deja de ser "valor posicional" para ser solo "orden". Para actas right-justified las dos interpretaciones dan el mismo entero (equivalencia verificada), así que el cambio no regresiona nada y arregla la cola.
- **`total_ciudadanos` se evalúa aparte y se excluye de la suma agregada** (`evaluate.py:202, 215`). Es un campo *independiente* (lo que dice el acta que votó), no la suma de los demás. Sumarlo dentro contaría doble. Compararlo por separado permite detectar inconsistencias del propio escribano.

**Sutilezas / gotchas** —

- **`shuffle=False` es obligatorio** (`evaluate.py:131`). Las predicciones se pegan a `df` por orden posicional (`np.concatenate`), no por una clave; barajar el loader desalinearía `pred` con `path`/`label` silenciosamente y todas las métricas saldrían mal sin error visible.
- **`field = "_".join(parts[1:-1])`**, no `parts[1]`. Olvidar esto rompe `partido_NN` y `votos_blanco/nulos/impugnados`.
- **Campos ausentes = 0, intencional.** Si un partido sacó 0 votos, ONPE no tiene row y nadie escribió celdas; tanto `real_value_for` como `reconstruct_value` devuelven 0 y coinciden. La convención "nadie escribe 000" es consistente entre etiquetado, entrenamiento y evaluación.
- **Filtrado de actas sin ground truth** (`evaluate.py:147-153`): se saltan actas sin entrada en parquets o con `totalVotosEmitidos` NaN. El notebook lo hace explícito con un aviso `n_eval/n_total` (`_inline_code.py:402-404`); el paquete lo hace en silencio. Por eso "actas evaluadas" puede ser < total del split.
- **Por qué acta-level es la métrica más estricta:** es una conjunción (`.all()`) sobre 42 campos. Con field-level de 98.87%, si los errores fueran independientes, acta-level esperado ≈ 0.9887^42 ≈ 62%; observamos 90.33% porque los errores se *concentran* en pocas actas (escritura corrida, tinta mala), no se reparten parejo. Un solo dígito mal en cualquiera de los 42 campos tumba el acta entera — refleja el caso de uso real: para certificar un acta tienes que leer *todo* bien.
- **Guardas numéricas** `max(..., 1)` y `max(..., 1e-9)` (`evaluate.py:232-235`): evitan dividir por cero cuando una clase no aparece nunca como real (recall) o nunca se predice (precision).

**Para la presentación** —

- "El modelo solo clasifica dígitos 0-9; la evaluación es lo que convierte esas predicciones sueltas en *votos reconstruidos* y los confronta contra el conteo oficial de ONPE. Sin esta etapa no podríamos decir nada sobre el problema real."
- "Reportamos cuatro métricas en orden de exigencia creciente: digit-level (98.12%), field-level (98.87%), **acta-level (90.33%)** — los 42 campos correctos a la vez — y la reconstrucción del total con MAE de 2.40 votos. Acta-level es la honesta porque para certificar un acta hay que leerla *entera* sin un solo error."
- "La estructura del documento (qué acta, qué campo, qué posición) se codifica en el nombre del crop y se recupera parseando el path: la evaluación es stateless, solo necesita los PNGs y el manifest."
- "Reconstruimos cada cifra concatenando las celdas con tinta en orden de lectura, no por posición fija. Eso fue lo que permitió el fix ink-aware: en las ~3% de actas con escritura corrida pasamos de field 98.87% a 99.45% y bajamos el MAE de 2.40 a 1.58 sin re-entrenar."
- "La matriz de confusión y el ranking de las 20 peores actas no son decoración: fueron el instrumento de diagnóstico que mostró que los errores se concentraban en un puñado de actas mal etiquetadas, no en debilidad del modelo."

Notas de paridad paquete↔notebook: en el notebook la función de ground truth se llama `field_value_for` (`_inline_code.py:69-81`) — cuerpo idéntico a `real_value_for` del paquete. `report_metrics` (notebook) y el bloque de prints de `main()` (paquete) computan las mismas 4 métricas. El paquete además persiste CSVs (`data/evaluate_<split>.csv`, `evaluate_worst20_<split>.csv`) y PNGs de visualización; el notebook usa `plt.show()` inline. El bloque `METRICS` separa `confusion_and_prf`/`plot_confusion`/`ablations_table` como funciones reutilizables; el paquete las tiene inline dentro de `main()`.
