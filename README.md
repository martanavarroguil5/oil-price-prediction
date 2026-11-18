# Estrato · Observatorio del Brent

Aplicación para prever el precio del Brent a **5, 63 y 252 sesiones de cotización**. Incluye la serie diaria, los pesos entrenados y la validación de RNN simple, LSTM y GRU frente a mantener el último precio conocido. [Enunciado de la práctica](docs/assignment/ENUNCIADO.pdf)

## Abrir la aplicación

Con Python 3.12 o superior y Poetry, desde la raíz del proyecto:

```sh
poetry install
make app
```

Abre [el observatorio](http://127.0.0.1:8501). La primera previsión utiliza los datos incluidos y funciona sin conexión. En **Datos y modelo** se puede actualizar la serie y generar otra previsión con los pesos existentes. El reentrenamiento es una acción independiente. La pestaña **Validación** permite elegir red y plazo para comparar el precio observado, el pronóstico y el ingenuo en las mismas fechas.

## Qué modelo se utiliza

La aplicación utiliza la **RNN simple W1** en los tres plazos, con pesos propios para cada uno. Se eligió con precios objetivo anteriores a 2022, dando a cada horizonte el mismo peso en la mejora porcentual del MAE frente al ingenuo. La RNN W1 consiguió **+3,82 %** de media en ese periodo. [Regla de selección](docs/experimento-ronda2.md)

| Horizonte | MAE de la app | MAE del ingenuo | Mejora de la app |
|---|---:|---:|---:|
| 5 sesiones | 3,951 | 3,987 | +0,89 % |
| 63 sesiones | 11,469 | 11,944 | +3,98 % |
| 252 sesiones | 17,627 | 16,502 | −6,82 % |

El MAE se expresa en USD por barril. La tabla corresponde a emisiones desde enero de 2022 con precio objetivo conocido hasta el 3 de noviembre de 2026. A un año, la ventaja observada en desarrollo se invirtió. Las previsiones diarias a ese plazo comparten buena parte del periodo que pretenden anticipar, lo que limita la evidencia sobre su estabilidad. [Comparativa completa, intervalos y errores por año](reports/comparativa.md)

En la comparación posterior de la app con las tres redes W0, la **GRU W0** ganó a 63 sesiones (**11,099 USD/barril**) y obtuvo la mejor mejora media del grupo en test (**+1,33 %**). Este periodo ya se había consultado. La elección de la familia publicada se mantiene ligada al criterio previo. [Registro de experimentos](docs/registro-experimentos.md) · [Comparación anual de W0](reports/experiments/w0/resultado.md)

## Datos y validación

Los datos diarios proceden de la EIA vía FRED y de Yahoo Finance. El calendario conserva las sesiones con precio Brent observado desde 2000 y alinea las demás series con la información disponible en cada fecha. La cesta OPEP quedó fuera por falta de acceso a la serie. El cuaderno de exploración documenta los huecos, el WTI negativo y el tratamiento de extremos. [Exploración y limpieza](notebooks/01_eda_y_limpieza.ipynb)

Las tres redes reciben las mismas 60 sesiones y 20 variables. Se compararon con 80 ensayos por red y horizonte, cuatro bloques de validación cronológica y cinco semillas. En cada corte solo se entrena con objetivos ya realizados. La salida incluye la mediana y los intervalos de predicción. [Cuaderno de comparación](notebooks/02_comparativa_modelos.ipynb)

## Dónde está cada cosa

| Carpeta | Contenido |
|---|---|
| `app/` y `src/oilcast/` | Interfaz HTML y Three.js, datos, redes, entrenamiento, evaluación e inferencia. |
| `data/` y `models/` | Series originales, conjunto preparado y pesos de la familia publicada. |
| `configs/` | Configuración principal, variante W0 y ejecución de prueba. |
| `notebooks/` y `docs/` | Los dos cuadernos, el enunciado, la decisión y el registro experimental. |
| `reports/` | Informe, figuras, resultados de la aplicación y tablas de los experimentos principales. |
| `scripts/` y `tests/` | Script de descarga facilitado para la práctica y pruebas automáticas. |

## Reproducir la comparativa

`make forecast` calcula los tres pronósticos. `make report` reconstruye el informe y las figuras con los resultados incluidos. Para repetir el entrenamiento y la evaluación con los hiperparámetros publicados:

```sh
make evaluate
make timing
make summarize
make report
```

`make tune` repite la búsqueda de hiperparámetros. `make test` ejecuta las pruebas y `make help` enumera los comandos disponibles. La práctica pide además una presentación de hasta cinco diapositivas de contenido. Su elaboración sigue pendiente.
