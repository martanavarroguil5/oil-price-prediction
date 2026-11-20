# Ronda 4: resultados

Mejora frente al ingenuo; positivo significa mejor que el ingenuo. Todas las cifras incluyen la
recalibración del centro, que aprende desde 2008. La decisión se guardó en
[decision.json](decision.json) el 19/11/2026 a las 23:55 UTC, con los hashes de las 18
predicciones, antes de calcular el test. [Protocolo](../../../docs/experimento-ronda4.md)

R4 = red W1 a 5 sesiones + redes nuevas a 63 y 252 sesiones que solo reciben la relación
oro/crudo, con atajo lineal.

| Variante | Red | Desarrollo 2008-2021, MAE (5 / 63 / 252) | Media desarrollo | Media 2008-2013 | Test 2022-2026, MAE (5 / 63 / 252) | Media test MAE | Test RMSE (5 / 63 / 252) | Media test RMSE |
|---|---|---|---:|---:|---|---:|---|---:|
| **R4** | **LSTM** | −0,1 % / +3,0 % / +12,9 % | **+5,3 %** | +5,6 % | 0,0 % / −2,3 % / **+10,0 %** | **+2,6 %** | 0,0 % / +6,6 % / **+20,7 %** | **+9,1 %** |
| R4 | GRU | −0,1 % / +2,0 % / +13,0 % | +5,0 % | +7,9 % | 0,0 % / −5,8 % / +5,7 % | 0,0 % | 0,0 % / +3,2 % / +14,4 % | +5,9 % |
| R4 | RNN simple | −0,2 % / +0,2 % / +13,0 % | +4,3 % | +6,4 % | +0,6 % / −3,1 % / +9,1 % | +2,2 % | +0,5 % / +3,1 % / +20,8 % | +8,1 % |
| W1 | LSTM | −0,1 % / −2,5 % / −8,5 % | −3,7 % | −7,3 % | 0,0 % / +3,2 % / −18,0 % | −4,9 % | 0,0 % / +0,6 % / −19,0 % | −6,1 % |
| W1 publicada | RNN simple | −0,2 % / −1,6 % / −15,4 % | −5,7 % | −17,5 % | +0,6 % / +3,7 % / −6,4 % | −0,7 % | +0,5 % / +1,7 % / −19,8 % | −5,8 % |
| W1 | GRU | −0,1 % / −1,2 % / −16,0 % | −5,8 % | −6,1 % | 0,0 % / +5,7 % / −2,9 % | +0,9 % | 0,0 % / +2,6 % / −13,7 % | −3,7 % |
| Referencia lineal | — | — / +2,2 % / +13,6 % | — | — | — / −1,4 % / +7,5 % | — | — / +3,9 % / +20,0 % | — |

Ordenado por la media de desarrollo, que es el criterio de la regla. La fila en negrita es la
elegida. La referencia lineal usa la misma variable y los mismos reentrenos y no era candidata.
[Desarrollo por periodo](development.csv) · [Test completo](test.csv) ·
[Test por año](annual.csv) · [Referencia lineal](test_lineal.csv)

## Lectura

- **La regla eligió R4 con LSTM** y la adopta: supera a la RNN W1 en 2008-2021 (+5,3 % frente a
  −5,7 %) y en 2008-2013 (+5,6 % frente a −17,5 %).
- **A un año, la previsión pasa a aportar valor.** Las tres redes R4 ganan al ingenuo en los tres
  periodos: 2008-2013 (+12,5 % a +19,2 %), 2014-2021 (+7,8 % a +13,3 %) y el test 2022-2026
  (+5,7 % a +10,0 % de MAE, +14 % a +21 % de RMSE). Con las 20 variables de W1, las 15
  combinaciones de la ronda 3 perdían en 2008-2013 y en el test.
- **La LSTM elegida reduce el RMSE a un año un 20,7 %**: MAE de 14,85 USD/barril frente a 16,50
  del ingenuo y 17,56 de la RNN W1. Su intervalo del 80 % puntúa 90,2 en Winkler, frente a 97,1
  del ingenuo, aunque es demasiado prudente: cubre el 99,6 % de los casos.
- **A tres meses mejora el error cuadrático pero no el absoluto**: +6,6 % de RMSE y −2,3 % de
  MAE. La red W1 ganaba en MAE en el test (+3,2 % a +5,7 %) pero perdía en desarrollo.
- **A una semana no cambia nada**: la LSTM W1 queda igual que el ingenuo (factor del centro 0).

## Límites

- **No es significativo por sí solo.** Con 252 sesiones, el test de 2022-2026 contiene menos de
  cinco años no solapados. El p-valor de Diebold-Mariano a un año es 0,40. La evidencia viene
  de que la ganancia aparece en tres periodos distintos, no de un único contraste.
- **En el test, toda la ganancia a un año viene de los orígenes de 2025.** Por años, la LSTM R4
  gana un 42 % en los orígenes de 2022 y un 43 % en los de 2025, pero pierde un 20 % en 2023 y
  un 97 % en 2024: el oro empezó a subir con fuerza, la red previó una recuperación del crudo y
  en 2025 el Brent bajó. **Sin los orígenes de 2025**, cuyo objetivo cae en la subida de 2026 por
  la guerra con Irán, la LSTM R4 pierde un 11,6 % de MAE (−2,0 % de RMSE) frente al ingenuo,
  y la RNN W1 ganaría un 4,9 %. La red vio el crudo barato frente al oro y la guerra hizo el
  resto; con estos datos no se puede separar el acierto de la suerte. El respaldo de la variable
  viene sobre todo de 2008-2013 y 2014-2021.
- **La variable se eligió entre 20** con los mismos años de desarrollo, y la trayectoria del Brent
  de 2022-2026 ya se conocía. El test de esta variable no se había calculado, pero no es ciego en
  sentido estricto.
- **Riesgo conocido:** si el oro sigue subiendo por motivos propios (compras de bancos centrales),
  el crudo parecerá barato sin que tenga por qué subir.

## Reproducción de la criba y el piloto

- Criba de los 20 predictores: `python scripts/ronda4_criba.py .`
- Piloto con los hiperparámetros de W1: `python scripts/ronda4_piloto.py P4 lstm 252` (variantes
  P1-P4). Puntuaciones de desarrollo de todas las variantes, incluida la referencia lineal:
  [piloto/desarrollo.csv](piloto/desarrollo.csv).
