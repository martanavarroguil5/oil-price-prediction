# Ronda 5: resultados

La decisión se guardó en [decision.json](decision.json) el 20/11/2026 a las 10:54 UTC, antes de
calcular el test. Incluye los hashes de las 21 predicciones, del protocolo y de las tres
configuraciones. [Protocolo](../../../docs/experimento-ronda5.md)

La mejora se mide frente al ingenuo: positivo significa mejor que el ingenuo. Todas las cifras
incluyen la recalibración del centro.

## Decisión con desarrollo

Una candidata merece la pena si mejora a B en al menos 0,5 puntos en 2008-2021 sin empeorar en
ninguno de los dos subperiodos. Solo se pueden adoptar las RNN.

| Plazo | Candidata | 2008-2013 | 2014-2021 | 2008-2021 | ¿Merece la pena? |
|---|---|---:|---:|---:|---|
| 5 | **B: RNN W1** | −0,40 % | −0,04 % | −0,23 % | Referencia |
| 5 | RNN W1, 20 semillas | −0,39 % | +0,01 % | −0,20 % | No (+0,03 puntos) |
| 5 | RNN con futuros | −0,54 % | +0,13 % | −0,21 % | No (+0,02 puntos) |
| 5 | Mejor no adoptable: GRU con futuros | −0,22 % | +0,09 % | −0,07 % | No (+0,16 puntos) |
| 63 | **B: RNN W1** | −2,01 % | −1,16 % | −1,59 % | Referencia |
| 63 | RNN W1, 20 semillas | −3,30 % | −1,10 % | −2,19 % | No (peor) |
| 63 | Mejor no adoptable: W1, tres redes | −1,28 % | −1,68 % | −1,48 % | No (+0,11 puntos) |
| 252 | **B: RNN oro/crudo** | +19,24 % | +7,84 % | +12,98 % | Referencia |
| 252 | RNN oro/crudo, 20 semillas | +18,93 % | +11,24 % | +14,71 % | No: +1,73 puntos, pero −0,31 en 2008-2013 |
| 252 | Mejor no adoptable: tres redes, 20 semillas | +18,60 % | +9,95 % | +13,85 % | No (−0,64 en 2008-2013) |

**Ninguna candidata merece la pena: B se queda como está en los tres plazos.**
[Todas las candidatas](development.csv)

## Test 2022-2026, calculado una vez después de la decisión

El intervalo es un bootstrap por bloques al 90 % (bloques de h sesiones, al menos 21).
"Fuera de muestra" incluye todos los orígenes desde 2008: desarrollo y test.

| Plazo | Candidata | Test MAE | Intervalo test | Test RMSE | p DM (abs.) | Fuera de muestra 2008-2026 | Intervalo |
|---|---|---:|---|---:|---:|---:|---|
| 5 | **B: RNN W1** | +0,62 % | +0,15 % a +1,12 % | +0,48 % | 0,044 | +0,07 % | −0,22 % a +0,35 % |
| 5 | RNN W1, 20 semillas | +0,75 % | +0,17 % a +1,38 % | +0,57 % | 0,046 | +0,13 % | −0,22 % a +0,48 % |
| 5 | RNN con futuros | +0,94 % | +0,41 % a +1,48 % | +0,76 % | 0,005 | +0,20 % | −0,26 % a +0,64 % |
| 5 | Con futuros, tres redes | +0,57 % | +0,19 % a +0,97 % | +0,51 % | 0,011 | +0,15 % | −0,13 % a +0,43 % |
| 63 | **B: RNN W1** | +3,73 % | +1,33 % a +6,69 % | +1,74 % | 0,015 | −0,15 % | −2,46 % a +2,08 % |
| 63 | RNN W1, 20 semillas | +3,07 % | +0,92 % a +5,71 % | +1,15 % | 0,015 | −0,74 % | −3,28 % a +1,72 % |
| 63 | W1, tres redes | +7,11 % | +1,42 % a +13,68 % | +3,22 % | 0,051 | +0,85 % | −2,20 % a +3,70 % |
| 252 | **B: RNN oro/crudo** | +9,08 % | −59,6 % a +45,2 % | +20,79 % | 0,747 | +10,71 % | −1,73 % a +21,49 % |
| 252 | RNN oro/crudo, 20 semillas | +11,11 % | −53,1 % a +45,0 % | +23,08 % | 0,678 | +12,43 % | +0,23 % a +23,28 % |
| 252 | Oro/crudo, tres redes | +9,51 % | −53,0 % a +41,6 % | +20,98 % | 0,712 | +11,31 % | −0,72 % a +22,35 % |

[Test completo](test.csv) · [Test por año](annual.csv)

| Plazo | Candidata | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---:|---:|---:|---:|---:|
| 5 | B: RNN W1 | +1,0 % | +0,4 % | +1,5 % | +0,7 % | 0,0 % |
| 5 | RNN con futuros | +1,1 % | +0,3 % | +1,8 % | +1,0 % | +0,8 % |
| 252 | B: RNN oro/crudo | +45,1 % | −10,5 % | −122,2 % | +47,4 % | — |
| 252 | RNN oro/crudo, 20 semillas | +44,8 % | −8,9 % | −110,6 % | +47,0 % | — |

## Lectura

- **1 semana: los futuros no aportan a la red antes de 2022.** En desarrollo, la RNN con
  futuros queda igual que B (−0,21 % frente a −0,23 %). En el test gana algo más (+0,94 % frente
  a +0,62 %) y mejora todos los años, pero la regla no puede usar el test para elegir. La
  diferencia es de unos 1,3 céntimos por barril. Coincide con la literatura: la señal del
  futuro es fuerte a 1 día y se diluye en una semana.
- **3 meses: B sigue siendo la mejor RNN.** Más semillas no la mejoran. La media de las tres
  redes llega a +7,1 % en el test, pero en desarrollo es casi igual y no se puede adoptar.
- **1 año: más semillas estabilizan la previsión, pero no superan la regla.** Las 20 semillas
  de la RNN van de −14,3 % a +18,5 % en el test (2 negativas); la media de las 20 da +11,1 %.
  En desarrollo mejora 1,7 puntos, pero queda 0,3 puntos por debajo de B en 2008-2013. En
  todo el periodo fuera de muestra, su intervalo es el único que no incluye el cero
  (+0,2 % a +23,3 %), aunque por muy poco.
- **El intervalo del test a 1 año no se estrecha con ningún modelo.** Todas las candidatas van
  de unos −55 % a +45 %. El ancho viene de que el test solo tiene unos 4 años independientes
  y de 2024, donde todas pierden más de un 100 %. Solo se estrecha con más años de datos.

## Conclusión

Por la regla fijada de antemano, la ronda 5 no cambia la aplicación. Confirma que B es estable:
más semillas, la media de las tres redes y los futuros dan resultados parecidos. Las dos
variantes que el test favorece (futuros a 1 semana, 20 semillas a 1 año) no mostraron mejora
suficiente antes de 2022. Adoptarlas ahora sería elegir mirando el test.
