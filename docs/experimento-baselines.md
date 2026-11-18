# Comparación por horizonte con modelos sencillos

Protocolo escrito el 18/11/2026, antes de ejecutar esta comparación.

## Pregunta y alcance

¿Aportan las redes valor sobre referencias sencillas? ¿Cambia la elección al decidir cada
horizonte por separado? Se reutilizan el dataset incluido, sus 20 variables y las predicciones
guardadas de RNN, LSTM y GRU W1. No se amplía la búsqueda de redes ni se actualizan los datos.
Este experimento no modifica los pesos ni la familia publicada en la aplicación.

## Candidatos fijados

- Ingenuo: mantener el precio del origen.
- Reversión a la media de 63 o 252 sesiones: pronosticar la rentabilidad logarítmica como
  `beta * -dev_ma`, con beta ajustada por mínimos cuadrados solo en entrenamiento y acotada
  a [0, 1]. No lleva intercepto ni impone que exista reversión si la beta estimada es negativa.
- Ridge: regresión de la rentabilidad logarítmica futura sobre el último valor y la media de
  las 60 sesiones de cada variable. Se usan los mismos datos disponibles para las redes,
  resumidos en 40 entradas. Recorte y estandarización ajustados solo en entrenamiento.
  Objetivo centrado y sin devolver la media histórica, como las redes W1. Sin intercepto.
  Penalizaciones fijadas: 0,01; 0,1; 1; 10; 100, en la función
  `mean((y - X beta)^2) + alpha * sum(beta^2)`.
- RNN, LSTM y GRU W1: mediana del ensemble guardado, con la recalibración causal existente.
  Los modelos sencillos no llevan esa recalibración adicional: se comparan procedimientos
  completos, no se atribuye cualquier diferencia únicamente a la arquitectura.

## Entrenamiento, selección y comparación

Los modelos sencillos se ajustan en los mismos cortes de 2014, 2016, 2018, 2020 y después
anualmente desde 2022. En cada corte solo se usan objetivos realizados estrictamente antes
de él. Mismos orígenes, precios y fechas objetivo para todos los candidatos de cada plazo.

Se decide un candidato por horizonte con el MAE en dólares sobre las predicciones cuyos
orígenes y objetivos son anteriores a 2022. Se agrupan los cuatro bloques de desarrollo;
las penalizaciones Ridge son candidatos dentro de esta selección, sin consultar el test.
Si un candidato queda a no más de 0,01 de la mejor razón MAE/modelo ingenuo, se prefiere
por este orden: ingenuo, reversión 63, reversión 252, Ridge con mayor penalización, RNN,
LSTM, GRU. Este margen favorece la sencillez ante diferencias pequeñas; no es una prueba
de significación estadística.

Primero se guarda `decision.json` con esa selección y luego se calcula el contraste desde
2022. Se publican todos los candidatos, los resultados por año y el ganador de desarrollo,
también cuando empeore después. No se cambia la regla tras observar ese contraste.

## Incertidumbre y límites

El periodo desde 2022 ya se consultó en experimentos anteriores: esta comparación es
retrospectiva, no un nuevo test ciego. Los hiperparámetros de las redes también utilizaron
los bloques de desarrollo; su puntuación en desarrollo no es una validación anidada nueva.
Una confirmación independiente requiere nuevas previsiones registradas y resultados futuros.

Se informa MAE, RMSE, mejora frente al ingenuo y Diebold-Mariano sobre **error absoluto**,
con HAC de h-1 retardos, alineado con el criterio MAE. Para el contraste histórico se calcula
un intervalo percentil del 95 % para la mejora del MAE mediante 1.000 remuestreos circulares
por bloques de h previsiones consecutivas, con semilla 42. Es orientativo, no corrige la
exploración anterior ni las comparaciones múltiples. A 252 sesiones hay muy pocos bloques.
También se informa la mejora en submuestras de orígenes espaciados h sesiones, para todos
los desplazamientos posibles; su rango es una sensibilidad al muestreo, no un intervalo de
confianza. Las sesiones sin una ventana completa pueden introducir huecos en los orígenes.

La comparación estudia el valor central. No evalúa nuevos intervalos predictivos, rentabilidad
de una estrategia, costes de transacción ni la disponibilidad intradía de las series originales.

## Reproducción

```sh
poetry run python -m oilcast -c configs/baselines.yaml baselines
```

En Windows con el entorno local ya instalado:

```powershell
.venv\Scripts\python.exe -m oilcast -c configs/baselines.yaml baselines
```

Salidas: `reports/experiments/baselines/resultado.md`, tablas CSV, predicciones y una decisión
con hashes del dataset, configuración, protocolo y predicciones de las redes para identificar
los archivos utilizados.
