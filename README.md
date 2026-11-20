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

## Versión publicada en GitHub Pages

La web también se publica como página estática en
<https://martanavarroguil5.github.io/oil-price-prediction/>. `make static` la genera en `site/` con los datos, los pesos y los resultados del repositorio: la previsión de los tres plazos, el simulador de compra, la prueba histórica, la validación y las figuras. Es una foto fija; la previsión no cambia hasta que se publica una versión nueva.

Diferencias con la aplicación local:
- No descarga datos nuevos ni reentrena: esos botones solo aparecen con `make app`.
- En la prueba histórica, los resultados ya están en los ficheros publicados, aunque la página no los muestra hasta pulsar **Ver qué ocurrió**.

El flujo [`pages.yml`](.github/workflows/pages.yml) genera y publica la web al subir cambios a `main`, o a mano desde la pestaña **Actions**. La primera vez hay que elegir **GitHub Actions** como origen en **Settings › Pages** del repositorio. Para probarla en local:

```sh
make static
python3 -m http.server 8530 --directory site
```

## Qué modelo se utiliza

La aplicación utiliza la **RNN simple** en los tres plazos, con pesos propios para cada uno (propuesta B). A 5 y 63 sesiones recibe las 20 variables diarias (red W1, elegida antes de ver el test). A 252 sesiones solo recibe la relación oro/crudo: cuántos barriles paga una onza de oro frente a su media de 3 años (el "golden constant" de Steve Hanke), con un atajo lineal.

| Horizonte | MAE de la app | MAE del ingenuo | Mejora del MAE | Mejora del RMSE |
|---|---:|---:|---:|---:|
| 5 sesiones | 3,962 | 3,987 | +0,6 % | +0,5 % |
| 63 sesiones | 11,499 | 11,944 | +3,7 % | +1,7 % |
| 252 sesiones | 15,003 | 16,502 | **+9,1 %** | **+20,8 %** |

El MAE se expresa en USD por barril. La tabla corresponde a emisiones desde enero de 2022 con precio objetivo conocido hasta el 3 de noviembre de 2026. Las cifras están verificadas: no hay datos del futuro en el entrenamiento y salen igual recalculadas desde cero. Su solidez es distinta en cada plazo:
- **1 semana:** gana todos los años, pero solo por 2,5 céntimos por barril.
- **3 meses:** gana con las cinco semillas y su intervalo bootstrap no incluye el cero, aunque antes de 2022 perdía un 1,6 %.
- **1 año:** depende de pocos episodios. Gana en las previsiones de 2022 y 2025, que acertaron la subida de 2026 por la guerra con Irán. Sin las de 2025 pierde un 16 %, y una variable inventada iguala el resultado una de cada cinco veces. Antes de 2022 ganó un 13 %.

[Verificación de la propuesta B](reports/experiments/propuesta_b/resultado.md) · [Comparativa completa, intervalos y errores por año](reports/comparativa.md)

### Por qué la RNN simple

- **Precisión.** En el test es la mejor de las tres redes a 1 semana (+0,6 %, frente a 0,0 % de la LSTM y la GRU) y en la media de los tres plazos (+4,5 %, frente a +4,4 % y +3,8 %).
- **Estabilidad.** La desviación del MAE entre semillas es del 2,8 %, frente al 5,4 % de la LSTM y el 5,6 % de la GRU.
- **Coste.** Es la más ligera: 32.000 parámetros de media, frente a 504.000 de la LSTM y 669.000 de la GRU. Las tres se entrenan en segundos en GPU.

**Esta combinación se eligió después de ver el test 2022-2026.** La regla fijada de antemano en la ronda 4 había elegido una LSTM, mejor antes de 2022 (+5,3 % frente a +3,7 %) y peor en el test (+2,6 % frente a +4,5 %). Para no volver a elegir con el test, la ronda 5 fijó su regla antes de entrenar. [Propuestas por plazo](reports/experiments/propuestas.md) · [Ronda 4](docs/experimento-ronda4.md)

### Alternativas que ganan a B en algún plazo

Mejora del MAE frente al ingenuo; entre paréntesis, la de B. Otras redes de rondas anteriores con resultados parecidos, como la GRU W0 a 63 sesiones (+7,1 %), están en [propuestas por plazo](reports/experiments/propuestas.md).

| Plazo | Alternativa | Test 2022-2026 | 2008-2021 | Por qué no se usa |
|---|---|---|---|---|
| 5 sesiones | GRU de la primera versión de la app | +1,6 % (B +0,6 %) | — | Es otra arquitectura y pierde en los otros plazos: −8,0 % a 3 meses y −15,1 % a 1 año. |
| 5 sesiones | RNN con el movimiento de los futuros tras la valoración del spot (ronda 5) | +0,9 % (B +0,6 %) | −0,2 % (B −0,2 %) | Antes de 2022 empataba con B. En el test la diferencia es de 1,3 céntimos por barril. |
| 63 sesiones | Media de RNN, LSTM y GRU (ronda 5) | +7,1 % (B +3,7 %) | −1,5 % (B −1,6 %) | Mezcla tres arquitecturas y antes de 2022 empataba con B. |
| 63 sesiones | GRU con las mismas variables | +5,7 % (B +3,7 %) | −1,2 % (B −1,6 %) | Es otra arquitectura. A 1 semana no mejora al ingenuo y a 1 año gana menos (+5,8 % frente a +9,1 %). |
| 252 sesiones | LSTM con la relación oro/crudo | +10,0 % (B +9,1 %) | +12,9 % (B +13,0 %) | Es otra arquitectura. A 1 semana no se aparta del ingenuo y a 3 meses gana menos (+3,2 % frente a +3,7 %). |
| 252 sesiones | RNN con 20 semillas en lugar de 5 (ronda 5) | +11,1 % (B +9,1 %) | +14,7 % (B +13,0 %) | En 2008-2013 quedaba 0,3 puntos por debajo de B, y la regla fijada antes de entrenar exigía no empeorar en ningún periodo. Su intervalo en el test es igual de ancho. |

Ninguna gana a B a la vez en los tres plazos, en el test y en todos los periodos anteriores a 2022. Cambiar el modelo por una de ellas sería ajustarlo al periodo 2022-2026, que ya se usó para elegir B. [Ronda 5](reports/experiments/ronda5/resultado.md)

Las rondas 3 a 5 probaron otras ideas de la literatura con el mismo protocolo: una red para los tres plazos, márgenes de refino, selección por la mediana, otros predictores, el movimiento de los futuros, más semillas y la media de las tres redes. Solo la relación oro/crudo mejoró al ingenuo a un año en los dos periodos de desarrollo. [Ronda 3](reports/experiments/ronda3/resultado.md) · [Ronda 4](reports/experiments/ronda4/resultado.md) · [Ronda 5](docs/experimento-ronda5.md) · [Registro de experimentos](docs/registro-experimentos.md)

## Datos y validación

Los datos diarios proceden de la EIA vía FRED y de Yahoo Finance. El calendario conserva las sesiones con precio Brent observado desde 2000 y alinea las demás series con su última observación fechada hasta ese día. No se reconstruyen todos los instantes de publicación ni versiones históricas: es una limitación de la evaluación operativa. La cesta OPEP quedó fuera por falta de acceso a la serie. El cuaderno de exploración documenta los huecos, el WTI negativo y el tratamiento de extremos. [Exploración y limpieza](notebooks/01_eda_y_limpieza.ipynb) · [Auditoría y decisión de cierre](docs/auditoria-final.md)

Las tres redes reciben las mismas 60 sesiones y las mismas variables en cada plazo: 20 a 5 y 63 sesiones y la relación oro/crudo a 252. Se compararon con 80 ensayos por red y horizonte, cuatro bloques de validación cronológica y cinco semillas. En cada corte solo se entrena con objetivos ya realizados. La salida incluye la mediana y los intervalos de predicción. [Cuaderno de comparación](notebooks/02_comparativa_modelos.ipynb)

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

`make tune` repite la búsqueda de hiperparámetros. `make test` ejecuta las pruebas y `make help` enumera los comandos disponibles. La práctica pide además una presentación de hasta cinco diapositivas de contenido: problema y oportunidad, solución, comparativa de las tres redes, demostración con los tres plazos y propuesta de valor.

## Comparación con modelos sencillos

`make baselines` compara las redes guardadas con regresión Ridge y reversión a la media,
con una elección por horizonte basada en objetivos anteriores a 2022. Publica las métricas,
la incertidumbre por bloques y los resultados por año en un directorio independiente.
En Windows se puede ejecutar `.venv\Scripts\python.exe -m oilcast -c configs/baselines.yaml baselines`.
Es una comparación retrospectiva; el periodo desde 2022 ya se había consultado.
[Protocolo](docs/experimento-baselines.md) · [Resultados](reports/experiments/baselines/resultado.md)
