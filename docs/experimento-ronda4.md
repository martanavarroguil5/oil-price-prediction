# Experimento, ronda 4: la relación oro/crudo

> Protocolo escrito el 20/11/2026, **antes** de la búsqueda de hiperparámetros de esta ronda y
> sin calcular el test de ninguna variante con la nueva variable. La regla de decisión no se
> cambia después de ver los resultados.

## Qué dice la literatura

- **Baumeister y Kilian** (JBES 2015; Baumeister, Kilian y Lee 2014) combinan seis modelos
  (VAR del mercado mundial, futuros, márgenes de refino, materias primas industriales y
  ingenuo). La combinación reduce el error cuadrático medio hasta un 12-18 % frente al
  ingenuo, pero con datos **mensuales** (producción, inventarios, actividad real). El
  enunciado pide datos diarios, así que el VAR queda fuera.
- **Curva de futuros.** Es el mejor predictor a pocos meses (FRBSF). La EIA solo publica los
  contratos 1-4 del WTI y **dejó de actualizarlos en abril de 2024**, por lo que la aplicación
  no podría usarlos.
- **Materias primas industriales** (cobre): señal pequeña e inestable en esta muestra.
- **"Golden constant" de Steve Hanke** (Cato Institute): el precio del crudo medido en onzas
  de oro vuelve a su media, y el ajuste se produce sobre todo por el lado del petróleo. Hanke
  estima que la mitad de la desviación se corrige en unos 14 meses.
- **LSTNet** (Lai et al., 2018) añade a la red recurrente un atajo lineal en paralelo para que
  la red no tenga que aprender la parte lineal de la señal.

## Criba de variables (solo desarrollo, objetivo anterior a 2022)

Se probaron 20 predictores sugeridos por la literatura con una regresión lineal recursiva por
horizonte (reentrenos anuales, purga del objetivo). Solo uno mejoró el MAE frente al ingenuo
en 2008-2013 y en 2014-2021 a 252 sesiones con margen claro: **la desviación del logaritmo de
barriles por onza (oro/Brent) frente a su media de 3 años** (`oro_crudo`). Por bloques de dos
años, a 252 sesiones, mejoró el MAE en 7 de 8 bloques entre 2006 y 2021. Falló en 2014-2015,
cuando el crudo siguió cayendo aunque ya estaba barato frente al oro. El resultado es
parecido con medias de 2 y 5 años.

**Advertencia.** La variable se eligió entre 20 con los mismos años que se usan para decidir,
así que las cifras de desarrollo de esta ronda son optimistas. La prueba que cuenta es el test.

## Piloto (solo desarrollo)

Con los hiperparámetros publicados de W1, reentrenos de calibración de 2008 a 2020 y 5
semillas. Mejora del MAE frente al ingenuo, después de la recalibración del centro:

| 2008-2021 | 63 sesiones (RNN / LSTM / GRU) | 252 sesiones (RNN / LSTM / GRU) |
|---|---|---|
| W1 publicada | −1,6 % / −2,5 % / −1,2 % | −15,4 % / −8,5 % / −16,0 % |
| W1 + `oro_crudo` (21 variables) | −0,2 % / −2,3 % / −1,3 % | −6,0 % / −6,9 % / −10,8 % |
| Igual con atajo lineal | 0,0 % / −2,3 % / −1,4 % | −2,6 % / −10,9 % / −10,4 % |
| Solo `oro_crudo` | +1,3 % / −0,7 % / +1,6 % | +10,4 % / +2,2 % / +6,7 % |
| **Solo `oro_crudo` con atajo lineal** | +1,7 % / +0,5 % / +1,6 % | +13,3 % / +2,6 % / +7,2 % |
| Referencia lineal (solo `oro_crudo`) | +2,2 % | +13,6 % |

Con las 20 variables de W1 la señal se pierde. En el plazo anual hay muy pocas observaciones
independientes y las demás variables solo añaden ruido.

## Variante R4

- **5 sesiones:** la red W1 publicada, sin cambios. Ninguna señal de la criba sirvió a 5
  sesiones.
- **63 y 252 sesiones:** una red por horizonte que recibe 60 sesiones de `oro_crudo`, con
  atajo lineal desde el último día. El atajo empieza en cero. Las tres redes tienen la misma
  entrada, la misma ventana y el mismo espacio de búsqueda que W1, con **80 ensayos** de
  Optuna, pérdida pinball, misma semilla y los bloques de validación 2014-2021.
- Reentrenos de calibración 2008-2020 (2008-2013 nunca se usa para validar), test con
  reentreno anual desde 2022, 5 semillas, recalibración del centro e intervalo conformal como
  en W1. Configuración: `configs/ronda4.yaml`.

## Regla de decisión (fijada de antemano)

Igual que en la ronda 3. Para cada red, R4 combina W1 a 5 sesiones con la nueva red a 63 y 252.

1. Para cada combinación de variante (W1 o R4) y red se calcula la mejora media del MAE
   frente al ingenuo en los tres horizontes, en orígenes de 2008 a 2021 con objetivo anterior
   a 2022.
2. Se elige la combinación con mayor media.
3. Sustituye a la RNN W1 de la aplicación solo si su media en 2008-2021 supera la de la RNN
   W1 y en 2008-2013 no es peor.
4. Antes de calcular el test se guardan la decisión y los hashes de las predicciones.
5. El test 2022-2026 se calcula una sola vez y se publica tal cual, junto con la referencia
   lineal.

## Transparencia sobre el test

- El periodo 2022-2026 se consultó en rondas anteriores, y la trayectoria del Brent se ve en
  la aplicación. El test no es ciego en sentido estricto. Ninguna predicción con `oro_crudo`
  se ha evaluado en él antes de esta decisión.
- Es público que el oro subió con fuerza entre 2024 y 2026, impulsado en parte por las compras
  de los bancos centrales. Eso lleva `oro_crudo` a valores extremos y es un riesgo conocido
  para esta variable: el crudo puede parecer barato frente al oro sin que suba.

## Reproducción

```sh
poetry run python -m oilcast -c configs/ronda4.yaml tune
poetry run python -m oilcast -c configs/ronda4.yaml evaluate
poetry run python -m oilcast -c configs/ronda4.yaml ronda4-decide
poetry run python -m oilcast -c configs/ronda4.yaml ronda4-test
```
