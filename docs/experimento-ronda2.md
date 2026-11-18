# Experimento, ronda 2: atacar las causas de que el modelo pierda frente al ingenuo

> Documento escrito **antes** de ejecutar la ronda 2 (16/11/2026). La regla de decisión no se
> cambia después de ver los resultados.

## Punto de partida

La primera ronda adoptó V1 (búsqueda de hiperparámetros ampliada).
Analizando sus predicciones **anteriores a 2022** (reentrenos de 2014, 2016, 2018 y 2020, con
objetivos ya conocidos antes de 2022) se encontraron dos defectos:

1. **Deriva alcista.** Las redes predicen de media subidas del +4 % al +7 % a 3 meses y del
   +10 % al +19 % a 12 meses, cuando lo observado fue −1 % y −4,6 %. Aprendieron como tendencia
   permanente la subida de 2001-2013 (de 25 a 110 $).
2. **Exceso de confianza.** Su información es real pero débil (correlación 0,2-0,5 con lo que
   pasó) y predicen movimientos 2-3 veces mayores de lo que esa señal justifica.

Un tercer problema, a 12 meses, es de **tamaño de muestra**: con objetivos que se solapan casi por
completo, 2001-2021 solo aporta unos 20 años independientes para aprender.

## Variantes

Todas parten de V1 (mismas 20 variables, 80 trials, 16-256 neuronas, 1-3 capas) y comparan las
tres redes en igualdad de condiciones.

| Variante | Qué cambia | Causa que ataca |
|---|---|---|
| W0 `w0_recal` | Sin reentrenar: recalibración del centro sobre las predicciones de V1 | Exceso de confianza |
| W1 `w1_nodrift` | Se entrena **sin deriva** (objetivo centrado: la red no puede aprender una tendencia media) + recalibración | Deriva y exceso de confianza |
| W2 `w2_long` | W1 + **historia larga**: Brent desde 1987 (las variables que empiezan más tarde llevan canal de disponibilidad) | Tamaño de muestra a 12 meses |

**Recalibración del centro.** En cada fecha de previsión, el valor central se multiplica por un
factor λ entre 0 y 1 que se estima solo con los errores de predicciones anteriores **ya
conocidos en esa fecha** (igual que el intervalo conformal). Con λ = 0 la previsión coincide con
el ingenuo; mientras no hay al menos 100 errores conocidos, λ = 0: la red no apuesta hasta haber
demostrado que acierta. La misma λ se aplica en la aplicación.

## Regla de decisión (fijada de antemano)

- Se decide **solo con predicciones anteriores a 2022**: los reentrenos de 2014, 2016, 2018 y 2020,
  con orígenes y objetivos realizados antes del 1/1/2022.
- Métrica: mejora del MAE frente al ingenuo en esas predicciones, por horizonte.
- Se elige la combinación (variante, red) con mayor **mejora media en los tres horizontes**.
  Candidatos: V1 (tal cual), W0 y W1, con RNN simple, LSTM y GRU.
- **Corrección del 16/11/2026, antes de conocer ningún resultado de la ronda 2:** W2 queda fuera
  de la decisión porque el enunciado fija el periodo de estudio «desde el 1 de enero de 2000» y
  W2 usa datos desde 1987. Se ejecuta y se publica solo como análisis complementario (¿ayudaría
  más historia a 12 meses?).
- Esa combinación pasa a la aplicación solo si su mejora media antes de 2022 es positiva (si no,
  se mantiene la configuración actual).
- El test 2022-2026 se calcula una sola vez para todas las variantes y se publica tal cual, gane o
  pierda al ingenuo. **No se harán más rondas ajustadas mirando el test.**

## Advertencia sobre el test

El test 2022-2026 ya se ha consultado en la ronda 1 y en el análisis de los defectos, así que no
es una evaluación completamente ciega. Lo que hace defendible esta ronda es que los defectos se
diagnostican y la decisión se toma con datos anteriores a 2022.

## Resultados (17/11/2026)

Ejecutado en la GPU el 16/11/2026 (W1: 40 min; W2: 52 min). W0 se calculó en local sobre las
predicciones de V1. La tabla recoge el resultado de cada variante, red y horizonte. [Resultados de la ronda](../reports/experiments/seleccion.csv)

### Decisión según la regla

**Se adopta W1 (sin deriva + recalibración) con RNN simple.** Es la única combinación elegible con
una mejora clara antes de 2022 (+3,8 % de media; la siguiente, W0 con LSTM, +0,3 %).

### La combinación elegida frente al ingenuo

| Horizonte | Antes de 2022 (decide) | Test MAE | Test RMSE | p-valor DM | Años del test en que gana | Factor de encogimiento en test |
|---|---|---|---|---|---|---|
| 1 semana | −0,2 % | **+0,9 %** | +0,6 % | 0,22 | 4 de 5 | ≈ 0,5 |
| 3 meses | −1,9 % | **+4,0 %** | +1,8 % | 0,12 | 4 de 5 | ≈ 0,4 |
| 12 meses | +13,5 % | **−6,8 %** | −20,1 % | 0,12 | 1 de 4 | 1,0 |

(Mejora del error frente al ingenuo; positivo = mejor que el ingenuo.)

Comparada con el modelo de la aplicación actual (GRU original: +1,6 %, −8,0 % y −15,1 % de MAE), la
nueva combinación gana al ingenuo a 1 semana y a 3 meses y pierde mucho menos a 12 meses. En
conjunto su error medio es prácticamente el del ingenuo (1,006 veces), frente a 1,072 de la
GRU original. **Ninguna diferencia frente al ingenuo es estadísticamente significativa** (todos
los p-valores del test de Diebold-Mariano > 0,1).

### Qué funcionó y qué no

- **Quitar la deriva funciona.** La subida media que predecían las redes antes de 2022 baja de
  +4/7 % a +2 % a 3 meses y de +10/19 % a +7 % a 12 meses.
- **La recalibración funciona a 1 semana y a 3 meses.** Las predicciones se encogen a la mitad
  aproximadamente y pasan de perder a ganar al ingenuo, en 4 de los 5 años del test.
- **A 12 meses no hay arreglo.** Antes de 2022 la red acertaba a ese plazo (+13,5 %), así que la
  recalibración la dejó apostar con todo (factor 1,0); en 2022-2026 la relación se invirtió
  (los picos de las guerras se deshicieron) y perdió. Es el cambio de régimen que se anticipó y
  que no se puede corregir sin mirar el test.
- **Historia desde 1987 (W2, solo informativa):** no mejora antes de 2022 en ninguna red. En test,
  la RNN simple a 12 meses gana al ingenuo (+8,7 % de MAE, p = 0,68) pero las otras dos pierden:
  no hay evidencia de que más historia resuelva el plazo largo. No puede adoptarse porque el
  enunciado fija el periodo de estudio desde 2000.
- **El intervalo del 80 %** de la combinación elegida cubre el 79 % / 88 % / 90 % de los casos;
  a 3 y 12 meses es más ancho que el del ingenuo.

### Modelo integrado

La aplicación utiliza la RNN simple W1 seleccionada en esta ronda para los tres plazos. Los pesos por horizonte, los tiempos de entrenamiento y las métricas están publicados. La repetición posterior de W0 confirmó la elección en desarrollo y permitió comparar qué redes fueron más precisas en cada año posterior. [Pesos y metadatos](../models/) · [Comparativa de la aplicación](../reports/comparativa.md) · [Repetición de W0](../reports/experiments/w0/resultado.md)
