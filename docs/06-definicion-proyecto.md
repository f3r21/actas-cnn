# Definicion del Proyecto II — Deep Learning

Curso: Topicos en Inteligencia Artificial (CCOMP9-1).
Unidad: Redes Neuronales Convolucionales — Proyecto Deep Learning.

## Titulo

Reconocimiento y agregacion de votos manuscritos en actas electorales
peruanas con redes neuronales convolucionales.

## Resumen

Sistema automatico para leer los votos manuscritos de las actas
electorales presidenciales del Peru y agregar los totales por partido.
La Oficina Nacional de Procesos Electorales (ONPE) publica un
repositorio de aproximadamente 871,000 documentos de las Elecciones
Generales 2026 que cubre cinco elecciones simultaneas y varios tipos
de documento; el proyecto trabaja unicamente con el ~10% que
corresponde a actas de escrutinio Presidencial (84,449 actas), pues es
ahi donde estan las cifras manuscritas de conteo. El pipeline recorta
cada acta en 42 campos con una plantilla calibrada, separa los digitos
individuales y los clasifica con una red neuronal convolucional en
PyTorch. El modelo es una ResNet-18 estilo CIFAR (He et al., 2015),
adaptada a entrada de 32 × 32 px en escala de grises, con skip
connections que facilitan el entrenamiento profundo y batch
normalization para estabilizar la convergencia. La salida es el conteo
de votos por organizacion politica reconstruido a partir de los
digitos predichos.

## Datos

Los datos **ya existen**, no es necesario construirlos. Provienen del
repositorio publico de la Oficina Nacional de Procesos Electorales
(ONPE) en `gs://onpe-eg2026-pdfs-v2/` (1.58 TB, 871,001 PDFs de cinco
elecciones simultaneas y cuatro tipos de documento). De ese total, las
**actas de escrutinio Presidencial** (`idEleccion=10`, `tipo=1`) son
84,449 (~10% del bucket) y conforman el universo objetivo del
proyecto; los demas tipos (instalacion, sufragio, resoluciones) no
contienen las cifras manuscritas de conteo. Se excluyen las actas
STAE de Lima y Callao (impresas digitalmente, 2 paginas; las
manuscritas tienen 1 pagina) y se trabaja con una muestra de 5,000
actas manuscritas con semilla fija. Las etiquetas se obtienen por
cruce determinístico con los manifiestos oficiales de ONPE
(`actas_archivos`, `actas_cabecera`, `actas_votos`); no requiere
anotacion manual. Particion 70/15/15 por archivoId, ~152,000 crops de
32 × 32 px en escala de grises.
