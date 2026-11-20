# Propuestas de modelo por plazo (análisis posterior al test)

> Análisis del 20/11/2026, hecho **después** de ver el test 2022-2026 de las rondas 1 a 4. Sirve
> para comparar alternativas, no como resultado a ciegas: cualquier combinación elegida aquí
> se ha escogido mirando el test. **La aplicación usa la propuesta B desde el 20/11/2026**, con
> esa salvedad. [Verificación de B](propuesta_b/resultado.md)

Todas las cifras son la mejora del MAE en dólares frente al ingenuo; positivo significa mejor.
Salvo que se indique, usan la recalibración del centro con errores conocidos desde 2008. Las
filas marcadas "historia 2014" se evaluaron con errores desde 2014, como se publicaron
entonces, y difieren unas décimas.

## Propuesta A: LSTM R4 (elegida por la regla de la ronda 4)

Elegida con la regla fijada antes de la búsqueda de la ronda 4. A 5 sesiones usa la LSTM W1
y a 63 y 252 sesiones la LSTM que solo recibe la relación oro/crudo.

| | 5 sesiones | 63 sesiones | 252 sesiones | Media |
|---|---:|---:|---:|---:|
| Test 2022-2026, MAE | 0,0 % | −2,3 % | +10,0 % | +2,6 % |
| Test 2022-2026, RMSE | 0,0 % | +6,6 % | +20,7 % | +9,1 % |
| Desarrollo 2008-2021, MAE | −0,1 % | +3,0 % | +12,9 % | +5,3 % |
| 2008-2013, MAE | — | — | — | +5,6 % |

## Propuesta B: RNN simple en los tres plazos (la de la aplicación)

La RNN W1 publicada a 5 y 63 sesiones (20 variables diarias) y la RNN R4 a 252 sesiones
(relación oro/crudo con atajo lineal). Es una sola arquitectura, pero la combinación se
eligió después de ver el test.

| | 5 sesiones | 63 sesiones | 252 sesiones | Media |
|---|---:|---:|---:|---:|
| Test 2022-2026, MAE | +0,6 % | +3,7 % | +9,1 % | **+4,5 %** |
| Test 2022-2026, RMSE | +0,5 % | +1,7 % | +20,8 % | +7,7 % |
| MAE en USD/barril (ingenuo) | 3,962 (3,987) | 11,499 (11,944) | 15,003 (16,502) | |
| p-valor DM, error absoluto | 0,044 | 0,015 | 0,747 | |
| Desarrollo 2008-2021, MAE | −0,2 % | −1,6 % | +13,0 % | +3,7 % |
| 2008-2013, MAE | −0,4 % | −2,0 % | +19,2 % | +5,6 % |
| 2014-2021, MAE | 0,0 % | −1,2 % | +7,8 % | +2,2 % |

B gana a A en el test (+4,5 % frente a +2,6 %) y pierde en desarrollo (+3,7 % frente a
+5,3 %). A 63 sesiones, la RNN W1 gana en el test pero pierde antes de 2022.

## Las redes anteriores, plazo a plazo

Mejores resultados documentados en el test 2022-2026 entre todas las redes de las rondas 1 a 4:

| Plazo | Red | Test MAE | Los otros dos plazos | Desarrollo en ese plazo |
|---|---|---:|---|---:|
| 5 sesiones | GRU original (antes de W1) | **+1,6 %** | −8,0 % a 63, −15,1 % a 252 | — |
| 5 sesiones | RNN W1 publicada (historia 2014) | +0,9 % | +4,0 % a 63, −6,8 % a 252 | −0,2 % |
| 5 sesiones | RNN W0 (historia 2014) | +0,9 % | +1,8 % a 63, −4,0 % a 252 | −0,1 % |
| 63 sesiones | **GRU W0** (historia 2014) | **+7,1 %** | +0,1 % a 5, −3,1 % a 252 | −0,5 % |
| 63 sesiones | GRU S1, ronda 3 | +6,0 % | +0,7 % a 5, −48,9 % a 252 | −0,6 % |
| 63 sesiones | GRU W1 | +5,7 % | 0,0 % a 5, −2,9 % a 252 | −1,2 % |
| 63 sesiones | RNN W1 (la de B) | +3,7 % | +0,6 % a 5, −6,4 % a 252 | −1,6 % |
| 252 sesiones | **LSTM R4** | **+10,0 %** | 0,0 % a 5, −2,3 % a 63 | +12,9 % |
| 252 sesiones | RNN R4 (la de B) | +9,1 % | +0,6 % a 5, −3,1 % a 63 | +13,0 % |
| 252 sesiones | LSTM W0 (historia 2014) | +2,5 % | +0,1 % a 5, −0,8 % a 63 | 0,0 % |

Fuentes: [W0](w0/resultado.md), [ronda 2](../../docs/experimento-ronda2.md) (GRU original),
[ronda 3](ronda3/test.csv) y [ronda 4](ronda4/test.csv). El desarrollo de W0 cubre 2014-2021;
el resto, 2008-2021.

## Lectura

- **Ninguna red anterior mejora a B a la vez a 5 y a 63 sesiones.** En esos dos plazos B ya
  usa la RNN W1 publicada, la mejor red completa de las rondas anteriores.
- **La GRU W0 es la mejor a 63 sesiones** (+7,1 %, p = 0,038 en error absoluto), pero a 5
  sesiones apenas mejora (+0,1 %) y antes de 2022 perdía en ese plazo (−0,5 %).
- **La GRU original es la mejor a 5 sesiones** (+1,6 %), pero pierde un 8,0 % a 63 sesiones.
- **A 63 sesiones todas las ventajas del test desaparecen en desarrollo.** Las GRU ganan entre un
  5 % y un 7 % en 2022-2026 y pierden antes de 2022. No hay una señal estable a tres meses.
- **A 252 sesiones solo las redes con oro/crudo ganan**, en desarrollo y en test. En el test,
  la ganancia depende de los orígenes de 2025 ([detalle](ronda4/resultado.md)).

Combinar la mejor red de cada plazo (RNN W1, GRU W0 y LSTM R4) daría alrededor de +5,9 % de
media en el test. Mezcla tres arquitecturas y dos historias de evaluación, y cada pieza se
eligió por su resultado en ese mismo test.

## Después de la ronda 5

La [ronda 5](ronda5/resultado.md) probó, con una regla fijada antes de entrenar, el movimiento
de los futuros a 1 semana, 20 semillas y la media de las tres redes en cada plazo. Ninguna RNN
cumplió la regla y B se mantiene. En el test, tres variantes ganan a B en un plazo:

| Plazo | Variante | Test MAE | B | 2008-2021 (variante / B) |
|---|---|---:|---:|---|
| 5 sesiones | RNN con futuros | +0,9 % | +0,6 % | −0,2 % / −0,2 % |
| 63 sesiones | Media de RNN, LSTM y GRU | +7,1 % | +3,7 % | −1,5 % / −1,6 % |
| 252 sesiones | RNN con 20 semillas | +11,1 % | +9,1 % | +14,7 % / +13,0 % (−0,3 puntos en 2008-2013) |

Ninguna mejoraba a B en todos los periodos anteriores a 2022, y la media de las tres redes
mezcla arquitecturas. Adoptarlas sería volver a elegir con el test.

