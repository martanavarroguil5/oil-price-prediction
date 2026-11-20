# Ronda 3: resultados

Mejora del MAE en dólares frente al ingenuo; positivo significa mejor que el ingenuo. Todas las
cifras incluyen la recalibración del centro, que en esta ronda aprende desde 2008. La decisión
se guardó en [decision.json](decision.json) el 19/11/2026 a las 21:31 UTC, con los hashes de
las 45 predicciones, antes de calcular el test. [Protocolo](../../../docs/experimento-ronda3.md)

| Variante | Red | Desarrollo 2008-2021 (5 / 63 / 252) | Media desarrollo | Media 2008-2013 | Test 2022-2026 (5 / 63 / 252) | Media test |
|---|---|---|---:|---:|---|---:|
| **M2 multihorizonte + márgenes** | LSTM | −0,1 % / +0,4 % / +3,9 % | +1,4 % | −0,3 % | +0,2 % / +1,5 % / −14,4 % | −4,2 % |
| S1 una red por plazo, mediana | GRU | +0,0 % / −0,6 % / −1,7 % | −0,8 % | −4,3 % | +0,7 % / +6,0 % / −48,9 % | −14,1 % |
| M0 multihorizonte, pinball | RNN simple | 0,0 % / −1,4 % / −2,0 % | −1,1 % | −5,3 % | 0,0 % / +0,2 % / −5,1 % | −1,6 % |
| M1 multihorizonte, mediana | RNN simple | −0,2 % / −2,0 % / −3,1 % | −1,8 % | −10,4 % | +0,0 % / +0,3 % / −2,9 % | −0,9 % |
| M0 multihorizonte, pinball | LSTM | −0,1 % / −1,1 % / −4,6 % | −1,9 % | −3,4 % | +0,1 % / +1,1 % / −19,7 % | −6,2 % |
| M1 multihorizonte, mediana | LSTM | 0,0 % / −0,5 % / −7,5 % | −2,7 % | −8,2 % | +0,1 % / −1,7 % / −43,4 % | −15,0 % |
| S1 una red por plazo, mediana | RNN simple | −0,1 % / −0,4 % / −7,8 % | −2,8 % | −8,7 % | +0,1 % / +2,4 % / −2,0 % | +0,2 % |
| W1 publicada | LSTM | −0,1 % / −2,5 % / −8,5 % | −3,7 % | −7,3 % | +0,0 % / +3,2 % / −18,0 % | −4,9 % |
| S1 una red por plazo, mediana | LSTM | 0,0 % / −1,1 % / −13,0 % | −4,7 % | −13,5 % | +0,2 % / +2,7 % / −12,1 % | −3,1 % |
| W1 publicada | RNN simple | −0,2 % / −1,6 % / −15,4 % | −5,7 % | −17,5 % | +0,6 % / +3,7 % / −6,4 % | −0,7 % |
| W1 publicada | GRU | −0,1 % / −1,2 % / −16,0 % | −5,8 % | −6,1 % | +0,0 % / +5,7 % / −2,9 % | +0,9 % |
| M1 multihorizonte, mediana | GRU | 0,0 % / −0,8 % / −20,8 % | −7,2 % | −15,5 % | +0,4 % / +5,4 % / −9,6 % | −1,3 % |
| M0 multihorizonte, pinball | GRU | +0,0 % / −0,7 % / −21,5 % | −7,4 % | −12,6 % | +0,1 % / +1,9 % / −3,9 % | −0,6 % |
| M2 multihorizonte + márgenes | RNN simple | 0,0 % / −2,2 % / −22,7 % | −8,3 % | −18,0 % | +0,2 % / +3,5 % / −7,9 % | −1,4 % |
| M2 multihorizonte + márgenes | GRU | +0,0 % / −0,7 % / −25,1 % | −8,6 % | −19,3 % | +0,1 % / +1,2 % / −17,2 % | −5,3 % |

Ordenado por la media de desarrollo, que es el criterio de la regla. La fila en negrita es la
elegida. [Desarrollo por periodo](development.csv) · [Test completo](test.csv) · [Test por año](annual.csv)

## Lectura

- **La regla eligió la red multihorizonte LSTM con márgenes de refino (M2).** Fue la única
  combinación con una media positiva en 2008-2021 (+1,4 %) y la que menos perdió en 2008-2013.
  Cumple las dos condiciones para sustituir a la RNN W1, que en 2008-2013 pierde un 17,5 %.
- **En el test es peor que la RNN W1 publicada.** Su media baja a −4,2 %, frente a −0,7 % de la
  RNN W1 evaluada con la misma historia: +0,2 % a 5 sesiones, +1,5 % a 63 y −14,4 % a 252.
  La ventaja de desarrollo no se trasladó al periodo 2022-2026.
- **A 252 sesiones, ninguna red aporta valor fuera del periodo de ajuste.** Las 15 combinaciones
  pierden frente al ingenuo en 2008-2013 y las 15 pierden en el test. Solo ganan en 2014-2021,
  el periodo que se usó para elegir hiperparámetros. En 2012-2013, con el Brent estable cerca de
  110 dólares, las redes anuales llegaron a prever subidas del 27 % que no ocurrieron.
- **A 63 sesiones la señal no es estable.** En desarrollo pierden 14 de 15 combinaciones, aunque
  por poco. En el test ganan 14 de 15, entre +0,2 % y +6,0 %. La RNN W1 obtiene +3,7 % con
  p = 0,015 en Diebold-Mariano sobre error absoluto, sin corregir por comparaciones múltiples.
- **A 5 sesiones todas empatan prácticamente con el ingenuo**: entre −0,2 % y +0,7 %.
- **La red multihorizonte y la selección por la mediana no mejoraron de forma general.**
  Reducen las pérdidas anuales en 2014-2021, pero no en 2008-2013 ni en el test.
- **Con la historia desde 2008, la RNN W1 cambia ligeramente en el test**: +0,6 %, +3,7 % y
  −6,4 %, frente a +0,9 %, +4,0 % y −6,8 % publicados. Su factor de recalibración aprende con
  más años.

## Límites

Se eligió entre 15 combinaciones y el test 2022-2026 ya se había consultado en rondas
anteriores. La red W1 se amplió con reentrenos en CPU; las demás variantes se entrenaron en la
RTX 3090. Las previsiones a 252 sesiones se solapan casi por completo: cada periodo de seis u
ocho años equivale a muy pocas observaciones independientes.
