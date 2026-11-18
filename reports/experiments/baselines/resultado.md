# Modelos sencillos y elección por horizonte

Comparación retrospectiva con datos y predicciones incluidos en el repositorio. El periodo desde 2022 ya se había consultado. La selección se guarda antes de calcular sus métricas, usando solo orígenes y objetivos anteriores a 2022.

[Protocolo y límites](../../../docs/experimento-baselines.md). La aplicación conserva su modelo publicado.

## Candidato elegido en desarrollo, por horizonte

| Sesiones | Elegido | Mejora desarrollo | MAE histórico ($) | Mejora histórica | IC95 de la mejora | Mejora histórica RNN de la app |
|---|---|---|---|---|---|---|
| 5 | naive | +0.00 % | 3.987 | +0.00 % | [+0.00 %, +0.00 %] | +0.89 % |
| 63 | naive | +0.00 % | 11.944 | +0.00 % | [+0.00 %, +0.00 %] | +3.98 % |
| 252 | rnn | +13.52 % | 17.627 | -6.82 % | [-30.56 %, +39.04 %] | -6.82 % |

Una mejora positiva reduce el MAE frente a mantener el precio actual. El IC95 usa bloques de h previsiones, conserva el emparejamiento con el ingenuo y es orientativo. No es un intervalo del precio futuro ni corrige comparaciones múltiples. Si incluye cero, no establece una ventaja estable con este remuestreo.

## Todos los candidatos

| Sesiones | Modelo | Mejora desarrollo | MAE histórico ($) | RMSE histórico ($) | Mejora histórica | IC95 de la mejora | p DM absoluto |
|---|---|---|---|---|---|---|---|
| 5 | gru | -0.11 % | 3.983 | 5.734 | +0.10 % | [-0.07 %, +0.28 %] | 0.275 |
| 5 | lstm | -0.01 % | 3.987 | 5.738 | +0.00 % | [+0.00 %, +0.00 %] | — |
| 5 | naive | +0.00 % | 3.987 | 5.738 | +0.00 % | — | — |
| 5 | reversion_252 | -0.16 % | 3.978 | 5.727 | +0.21 % | [-0.07 %, +0.52 %] | 0.178 |
| 5 | reversion_63 | +0.00 % | 3.987 | 5.738 | +0.00 % | [+0.00 %, +0.00 %] | — |
| 5 | ridge_0.01 | -1.53 % | 4.051 | 5.876 | -1.62 % | [-4.81 %, +1.91 %] | 0.356 |
| 5 | ridge_0.1 | -0.74 % | 4.006 | 5.767 | -0.48 % | [-2.50 %, +1.72 %] | 0.673 |
| 5 | ridge_1 | -0.37 % | 3.979 | 5.732 | +0.20 % | [-0.70 %, +1.20 %] | 0.688 |
| 5 | ridge_10 | +0.02 % | 3.982 | 5.736 | +0.11 % | [-0.10 %, +0.35 %] | 0.337 |
| 5 | ridge_100 | +0.01 % | 3.986 | 5.738 | +0.02 % | [-0.01 %, +0.05 %] | 0.218 |
| 5 | rnn | -0.22 % | 3.951 | 5.703 | +0.89 % | [-0.03 %, +1.88 %] | 0.064 |
| 63 | gru | -1.29 % | 11.303 | 17.010 | +5.37 % | [-7.00 %, +17.37 %] | 0.368 |
| 63 | lstm | -1.45 % | 11.552 | 17.321 | +3.28 % | [-6.71 %, +12.98 %] | 0.453 |
| 63 | naive | +0.00 % | 11.944 | 17.437 | +0.00 % | — | — |
| 63 | reversion_252 | -2.38 % | 11.588 | 16.685 | +2.98 % | [-4.62 %, +8.91 %] | 0.406 |
| 63 | reversion_63 | +0.00 % | 11.631 | 16.732 | +2.62 % | [-2.42 %, +6.98 %] | 0.286 |
| 63 | ridge_0.01 | -15.90 % | 15.403 | 21.191 | -28.96 % | [-64.11 %, -3.99 %] | 0.017 |
| 63 | ridge_0.1 | -14.32 % | 13.705 | 20.062 | -14.74 % | [-35.79 %, +3.01 %] | 0.107 |
| 63 | ridge_1 | -8.32 % | 11.935 | 18.131 | +0.08 % | [-5.90 %, +8.73 %] | 0.984 |
| 63 | ridge_10 | -1.08 % | 11.807 | 17.490 | +1.14 % | [-0.58 %, +3.89 %] | 0.245 |
| 63 | ridge_100 | -0.05 % | 11.923 | 17.439 | +0.18 % | [-0.07 %, +0.58 %] | 0.205 |
| 63 | rnn | -1.85 % | 11.469 | 17.117 | +3.98 % | [+0.95 %, +7.81 %] | 0.019 |
| 252 | gru | +0.13 % | 16.614 | 24.727 | -0.68 % | [-18.35 %, +33.25 %] | 0.951 |
| 252 | lstm | +0.94 % | 20.244 | 27.523 | -22.67 % | [-67.61 %, +17.51 %] | 0.184 |
| 252 | naive | +0.00 % | 16.502 | 22.527 | +0.00 % | — | — |
| 252 | reversion_252 | -5.37 % | 14.626 | 19.660 | +11.37 % | [-2.26 %, +22.41 %] | 0.159 |
| 252 | reversion_63 | -1.31 % | 15.939 | 21.395 | +3.41 % | [-3.37 %, +7.53 %] | 0.266 |
| 252 | ridge_0.01 | -52.82 % | 32.230 | 37.253 | -95.31 % | [-208.70 %, -39.17 %] | <0.001 |
| 252 | ridge_0.1 | -42.69 % | 26.016 | 31.598 | -57.65 % | [-112.60 %, -33.16 %] | <0.001 |
| 252 | ridge_1 | -13.76 % | 16.477 | 25.323 | +0.15 % | [-21.11 %, +42.31 %] | 0.991 |
| 252 | ridge_10 | +1.40 % | 15.882 | 22.637 | +3.76 % | [-2.53 %, +17.26 %] | 0.305 |
| 252 | ridge_100 | +0.52 % | 16.407 | 22.512 | +0.58 % | [-0.20 %, +2.31 %] | 0.192 |
| 252 | rnn | +13.52 % | 17.627 | 27.061 | -6.82 % | [-30.56 %, +39.04 %] | 0.668 |

Los p-valores DM de este experimento usan error absoluto (MAE); los del informe original usan error cuadrático. No son directamente intercambiables.

## Mejor referencia sencilla en el contraste histórico (exploratorio)

Esta tabla elige después de observar el periodo desde 2022; no cambia la decisión tomada en desarrollo ni propone un nuevo modelo de producción.

| Sesiones | Mejor modelo sencillo histórico | Mejora histórica | Mejora en desarrollo de ese modelo |
|---|---|---|---|
| 5 | reversion_252 | +0.21 % | -0.16 % |
| 63 | reversion_252 | +2.98 % | -2.38 % |
| 252 | reversion_252 | +11.37 % | -5.37 % |

## Sensibilidad al solapamiento de los candidatos elegidos

| Sesiones | Modelo | Mínimo por desplazamiento | Mediana | Máximo | Mínimo de observaciones por submuestra |
|---|---|---|---|---|---|
| 5 | naive | +0.00 % | +0.00 % | +0.00 % | 243 |
| 63 | naive | +0.00 % | +0.00 % | +0.00 % | 18 |
| 252 | rnn | -108.33 % | -6.85 % | +96.82 % | 3 |

Cada submuestra toma un origen cada h observaciones para un desplazamiento distinto. Este rango muestra sensibilidad; no constituye un intervalo de confianza. Los huecos del calendario pueden separar todavía más algunos orígenes.

## Archivos para revisar

- `decision.json`: selección en desarrollo, regla y hashes de las entradas.
- `development.csv` y `test.csv`: todas las métricas por candidato y plazo.
- `annual.csv`: resultados por año del origen, con muestras emparejadas.
- `predictions.csv`: predicciones individuales reproducibles.
- `training_audit.csv`: cortes y último objetivo utilizado en cada entrenamiento.

Los ganadores históricos no se sustituyen por los elegidos en desarrollo. La confirmación independiente sigue pendiente de nuevas predicciones y resultados.
