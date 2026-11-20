# Experimento, ronda 5: exprimir la propuesta B

> Protocolo escrito el 20/11/2026, **antes** de entrenar ninguna variante de esta ronda y sin
> calcular el test de ninguna variante nueva. La regla de decisión no se cambia después de ver
> los resultados. El test 2022-2026 ya se consultó en rondas anteriores y la propuesta B se
> eligió mirándolo ([propuestas](../reports/experiments/propuestas.md)).

## Punto de partida

La aplicación usa la propuesta B: RNN W1 a 5 y 63 sesiones y RNN oro/crudo (ronda 4) a 252.
La [verificación](../reports/experiments/propuesta_b/resultado.md) mostró que la mejora a 5
sesiones es mínima y que a 252 depende de pocos episodios. Con datos anteriores a 2022:

- **Futuros (literatura: la base futuro-spot es la señal a corto plazo que más resiste).** El
  Brent spot se valora a las 16:30 de Londres y los futuros cierran a las 19:30. El movimiento
  del futuro después de la valoración anticipa el spot del día siguiente (+3,1 % y +1,4 % de MAE
  en 2008-2013 y 2014-2021 con una regresión lineal), pero a 5 sesiones solo +0,6 % y 0,0 %.
- **Combinación de modelos (Baumeister, Kilian y Lee).** Promediar modelos da una precisión más
  estable. A 252 sesiones, la media de las tres redes oro/crudo da +18,1 % y +10,2 % en
  2008-2013 y 2014-2021, frente a +19,2 % y +7,8 % de la RNN sola.
- **Más semillas.** A 252 sesiones, una de las cinco semillas de la RNN pierde frente al
  ingenuo; más semillas reducen esa variación.

## Candidatas por plazo

Todas usan la misma recalibración del centro y los reentrenos de calibración 2008-2020.

| Plazo | Candidata | Qué cambia |
|---|---|---|
| 5 | B: RNN W1 | Referencia (5 semillas) |
| 5 | W1, tres redes | Media de RNN, LSTM y GRU W1 |
| 5 | RNN W1, 20 semillas | Mismos hiperparámetros |
| 5 | W1, tres redes, 20 semillas | |
| 5 | Futuros: RNN, LSTM, GRU y su media | 20 variables de W1 + `late_bz1`, `late_bz2` y `late_cl1`; búsqueda nueva de 80 ensayos por red, pinball, bloques 2014-2021 |
| 63 | B: RNN W1 | Referencia |
| 63 | W1, tres redes; RNN W1 20 semillas; tres redes 20 semillas | Como a 5 sesiones |
| 252 | B: RNN oro/crudo | Referencia |
| 252 | Oro/crudo, tres redes; RNN 20 semillas; tres redes 20 semillas | Mismos hiperparámetros de la ronda 4 |

`late_bz1` es la rentabilidad del futuro del Brent (BZ=F) menos la del Brent spot el mismo día;
`late_bz2`, su suma de dos días; `late_cl1`, lo mismo con el futuro del WTI (CL=F). BZ=F empieza
en 2007, así que antes vale 0 y un canal indica que falta. Todas son fuentes del enunciado.

## Cumplimiento del enunciado

- Datos: solo series de las fuentes del enunciado (FRED/EIA y Yahoo Finance: BZ=F, CL=F, oro).
- Las tres redes se entrenan en cada plazo con los mismos datos, la misma ventana de 60
  sesiones, la misma partición y el mismo presupuesto de búsqueda (80 ensayos).
- La aplicación integra un único tipo de red (RNN simple), con pesos propios por plazo.
- Escalado ajustado solo con entrenamiento, CV walk-forward y comparación con el ingenuo.

## Regla de decisión (fijada de antemano)

Para cada plazo por separado:

1. Se calcula la mejora del MAE frente al ingenuo en desarrollo: orígenes de 2008-2013, de
   2014-2021 y de 2008-2021, siempre con objetivo anterior a 2022.
2. Una candidata **merece la pena** si en 2008-2021 mejora al componente de B en al menos
   **0,5 puntos** y no es peor que él ni en 2008-2013 ni en 2014-2021.
3. **Solo pueden adoptarse candidatas RNN.** El enunciado pide elegir un modelo para la
   aplicación, y B usa la RNN simple en los tres plazos. Las medias de las tres redes y las
   variantes LSTM y GRU se calculan igual y se publican como referencia, pero no se adoptan.
   Entre las RNN que merecen la pena se elige la de mayor mejora en 2008-2021.
4. Si ninguna merece la pena, B se queda como está en ese plazo.
5. La decisión se guarda con los hashes de las predicciones antes de calcular el test. El test
   se calcula una sola vez, con intervalos bootstrap por bloques, y se publica tal cual.

Implementar el resultado en la aplicación lo decide después la autora.

## Reproducción

```sh
poetry run python -m oilcast -c configs/ronda5_futuros.yaml tune
poetry run python -m oilcast -c configs/ronda5_futuros.yaml evaluate
poetry run python -m oilcast -c configs/ronda5_semillas_w1.yaml evaluate
poetry run python -m oilcast -c configs/ronda5_semillas_oro.yaml evaluate
poetry run python -m oilcast ronda5-decide
poetry run python -m oilcast ronda5-test
```

Las ejecuciones con 20 semillas reutilizan los hiperparámetros publicados, copiados en
`reports/experiments/ronda5/w1_20/tuning` y `reports/experiments/ronda5/oro_20/tuning`.
