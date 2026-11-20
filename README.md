# Estrato · Observatorio del Brent

Aplicación para prever el precio del Brent a **5, 63 y 252 sesiones de cotización**. Incluye la serie diaria, los pesos entrenados y la validación de RNN simple, LSTM y GRU frente a mantener el último precio conocido. [Enunciado de la práctica](docs/assignment/ENUNCIADO.pdf)

## Abrir la aplicación

Con Python 3.12 o superior y Poetry, desde la raíz del proyecto:

```sh
poetry install
make app
```

Abre [el observatorio](http://127.0.0.1:8501). La primera previsión utiliza los datos incluidos y funciona sin conexión. En **Datos y modelo** se puede actualizar la serie y generar otra previsión con los pesos existentes. El reentrenamiento es una acción independiente.

**Decisiones** traduce las previsiones a un presupuesto en USD: volumen equivalente,
precio de referencia, coste adicional y porcentaje de compra al inicio. Compara comprar
todo al precio de referencia, esperar al final del plazo o repartir entre ambos momentos.
Incluye un escenario manual y permite guardar el resumen. Son cálculos bajo supuestos;
no se recomienda una estrategia ni se afirma que esas políticas tengan una ventaja histórica.

**Prueba histórica** permite recorrer todos los orígenes guardados de las tres redes y
revelar después el resultado. Antes de revelar, la respuesta de la aplicación no incluye
el precio posterior ni observaciones futuras. Se advierte que es una evaluación retrospectiva.

**Validación** compara precio, pronóstico e ingenuo en las mismas fechas, e incluye una
tabla en rentabilidad simple: `precio futuro / precio de origen - 1`. Sus errores se
expresan en puntos porcentuales y el ingenuo predice un cambio del 0 %. Las redes ya
aprenden rentabilidades logarítmicas; esta tabla añade otra unidad de evaluación.
`poetry run python -m oilcast returns` exporta esa comparación a `reports/rentabilidad.csv`.
[Guía de demostración](docs/guia-demostracion.md)
[Mejoras implementadas y siguientes pasos](docs/mejoras-implementadas.md)

**Velas y violines.** En Decisiones se muestran 90 sesiones de velas OHLC y volumen del
futuro continuo Brent `BZ=F`, con sus fechas propias; no son velas del Brent spot. En
Validación se pueden consultar los violines de errores a cada plazo, en dólares o en
puntos porcentuales de rentabilidad, junto a cobertura, anchura y puntuación de Winkler
de los intervalos. `make diagnostics` (o `poetry run python -m oilcast diagnostics`)
regenera los PNG y sus metadatos con los datos y predicciones locales guardados.
En Validación se incluye también la dispersión observado–predicho, con la diagonal ideal,
referencia ingenua y R² de las predicciones originales. Puede verse en precio o rentabilidad.
Las figuras de dispersión de los tres plazos se descargan por red.
Las figuras incluyen el periodo utilizado; tras actualizar datos o resultados deben
regenerarse para reflejar los cambios. No se descarga ni se reentrena al generarlas.

**Calibración experimental de intervalos.** `make improve-intervals` compara siete
procedimientos con una regla fijada en desarrollo, sin reentrenar redes ni modificar sus
medianas. En Validación, **Calibración, a prueba** muestra el método publicado frente al
elegido, las métricas históricas y ambos intervalos para la previsión actual. El simulador
conserva los intervalos publicados: el experimento no los sustituye automáticamente.
Para reproducirlo en Windows: `.venv\Scripts\python.exe -m oilcast -c configs/intervals.yaml improve-intervals`.
[Protocolo](docs/experimento-intervalos.md) · [Resultados](reports/experiments/intervals/resultado.md)

## Qué modelo se utiliza

La aplicación utiliza la **LSTM R4**, con pesos propios para cada plazo. A 5 sesiones recibe las 20 variables diarias. A 63 y 252 sesiones solo recibe la relación oro/crudo: cuántos barriles paga una onza de oro frente a su media de 3 años (el "golden constant" de Steve Hanke), con un atajo lineal. Se eligió con una regla fijada antes de la búsqueda y con precios objetivo anteriores a 2022: **+5,3 %** de media en 2008-2021 y **+5,6 %** en 2008-2013, años que no se usaron para elegir hiperparámetros. [Regla y protocolo](docs/experimento-ronda4.md)

| Horizonte | MAE de la app | MAE del ingenuo | Mejora del MAE | Mejora del RMSE |
|---|---:|---:|---:|---:|
| 5 sesiones | 3,987 | 3,987 | 0,0 % | 0,0 % |
| 63 sesiones | 12,221 | 11,944 | −2,3 % | +6,6 % |
| 252 sesiones | 14,854 | 16,502 | **+10,0 %** | **+20,7 %** |

El MAE se expresa en USD por barril. La tabla corresponde a emisiones desde enero de 2022 con precio objetivo conocido hasta el 3 de noviembre de 2026. La red anterior (RNN W1) perdía un 6,8 % a un año. **A un año, la mejora depende de un episodio**: casi toda procede de las previsiones emitidas en 2025, que acertaron la subida de 2026 por la guerra con Irán. Sin ellas, la LSTM R4 pierde un 11,6 % frente al ingenuo. A 5 sesiones la previsión es el último precio, y a 63 mejora el error cuadrático pero no el absoluto. [Resultados de la ronda 4](reports/experiments/ronda4/resultado.md) · [Comparativa completa, intervalos y errores por año](reports/comparativa.md)

Las rondas 3 y 4 probaron otras ideas de la literatura con el mismo protocolo: una red para los tres plazos, márgenes de refino, selección por la mediana y otros predictores. Solo la relación oro/crudo mejoró al ingenuo a un año en los dos periodos de desarrollo. [Ronda 3](reports/experiments/ronda3/resultado.md) · [Registro de experimentos](docs/registro-experimentos.md)

## Datos y validación

Los datos diarios proceden de la EIA vía FRED y de Yahoo Finance. El calendario conserva las sesiones con precio Brent observado desde 2000 y alinea las demás series con su última observación fechada hasta ese día. No se reconstruyen todos los instantes de publicación ni versiones históricas: es una limitación de la evaluación operativa. La cesta OPEP quedó fuera por falta de acceso a la serie. El cuaderno de exploración documenta los huecos, el WTI negativo y el tratamiento de extremos. [Exploración y limpieza](notebooks/01_eda_y_limpieza.ipynb) · [Auditoría y decisión de cierre](docs/auditoria-final.md)

Las tres redes reciben las mismas 60 sesiones y las mismas variables en cada plazo: 20 a 5 sesiones y la relación oro/crudo a 63 y 252. Se compararon con 80 ensayos por red y horizonte, cuatro bloques de validación cronológica y cinco semillas. En cada corte solo se entrena con objetivos ya realizados. La salida incluye la mediana y los intervalos de predicción. [Cuaderno de comparación](notebooks/02_comparativa_modelos.ipynb)

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

## Comparación con modelos sencillos

`make baselines` compara las redes guardadas con regresión Ridge y reversión a la media,
con una elección por horizonte basada en objetivos anteriores a 2022. Publica las métricas,
la incertidumbre por bloques y los resultados por año en un directorio independiente.
En Windows se puede ejecutar `.venv\Scripts\python.exe -m oilcast -c configs/baselines.yaml baselines`.
Es una comparación retrospectiva; el periodo desde 2022 ya se había consultado.
[Protocolo](docs/experimento-baselines.md) · [Resultados](reports/experiments/baselines/resultado.md)
