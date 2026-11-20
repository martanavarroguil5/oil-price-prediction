# Experimento, ronda 3: una red para los tres horizontes

> Protocolo escrito el 19/11/2026, **antes** de entrenar ninguna variante de esta ronda. La
> regla de decisión no se cambia después de ver los resultados.

## Punto de partida (solo datos anteriores a 2022)

Con las predicciones guardadas de W1 se comprobó, sin entrenar nada, lo siguiente:

1. **Antes de 2022, las redes de corto y medio plazo pierden frente al ingenuo.** Sin la
   recalibración del centro, el MAE de la RNN es un 1,7 % peor a 5 sesiones y un 7,8 % peor
   a 63. La recalibración las devuelve casi al ingenuo.
2. **La señal lenta sí se aprende, pero desde el plazo largo.** La previsión de la RNN a 252
   sesiones, escalada a 63/252, mejora el MAE trimestral un 3,3 % antes de 2022
   (correlación 0,37 con lo ocurrido). La red entrenada solo para 63 sesiones empeora un 7,8 %
   (correlación 0,13). La red específica de 63 sesiones aprende ruido propio.
3. **La búsqueda no se fija en la mediana.** W1 elige hiperparámetros y época con la pérdida
   pinball de los siete cuantiles, dominada por la anchura del rango. Seis de las nueve
   configuraciones elegidas se entrenan entre 2 y 7 épocas.
4. **Más semillas ayudan poco.** Pasar de 1 a 5 semillas mejora el MAE en bruto, pero la
   tendencia apunta a menos de un punto adicional con más semillas.

También se exploraron, solo con datos anteriores a 2022, los márgenes de refino (gasolina
y gasóleo de Nueva York frente al Brent) y la base entre el Brent spot y su futuro. Dieron
señales pequeñas y muy desiguales entre bloques de dos años. Un piloto anterior ya había
probado los márgenes a 63 sesiones sin mejora, así que la variante con márgenes es
secundaria.

**Advertencia de transparencia.** Durante el diagnóstico se calculó sobre el test 2022-2026
la media simple de las tres redes W1. Esa combinación queda **excluida** de los candidatos.
El test 2022-2026 ya se había consultado en rondas anteriores.

## Variantes

Todas usan el mismo dataset, la ventana de 60 sesiones, las 20 variables de W1, el objetivo
sin deriva, el mismo espacio de búsqueda, los mismos bloques de validación de 2014 a 2021 y
**80 ensayos de Optuna por estudio** con la misma semilla para las tres redes.

| Variante | Qué cambia frente a W1 | Hipótesis |
|---|---|---|
| W1 | Nada (modelo publicado). Solo se añaden reentrenos de 2008, 2010 y 2012 para evaluar más años | Referencia |
| S1 | Una red por horizonte; búsqueda y parada temprana por el error de la mediana frente al ingenuo | Seleccionar para la mediana reduce el ruido del centro |
| M0 | **Una red para los tres horizontes** (una salida por horizonte, purga por horizonte); búsqueda por pinball como W1 | Compartir la representación traslada la señal lenta a los plazos cortos |
| M1 | M0 con búsqueda por el error de la mediana (media de los tres horizontes) | Ambos efectos |
| M2 | M1 con los márgenes de gasolina y gasóleo (22 variables) | Información nueva (hipótesis de Verleger) |

En la red multihorizonte, cada origen aporta los objetivos ya realizados antes del corte. Los
demás se enmascaran en la pérdida, igual que la purga de W1, pero por horizonte. Cada horizonte
se estandariza con su desviación de entrenamiento y pesa lo mismo en la pérdida.

## Evaluación de desarrollo

- Reentrenos anteriores al test en 2008, 2010, 2012, 2014, 2016, 2018 y 2020, con las épocas
  fijadas en la búsqueda y 5 semillas. **2008-2013 no se usó nunca para validar ni para elegir
  hiperparámetros** en ninguna variante.
- La recalibración del centro (factor λ entre 0 y 1) y el intervalo conformal usan todos los
  errores fuera de muestra ya conocidos en cada fecha, desde 2008.
- Desarrollo = orígenes de 2008 a 2021 con objetivo realizado antes del 1/1/2022.

## Regla de decisión (fijada de antemano)

1. Para cada combinación de variante y red se calcula la mejora del MAE en dólares frente al
   ingenuo en desarrollo, después de la recalibración, y su media en los tres horizontes.
2. Se elige la combinación con mayor media.
3. Sustituye a la RNN W1 de la aplicación solo si se cumplen las dos condiciones:
   - su media en 2008-2021 supera la de la RNN W1 evaluada igual;
   - en 2008-2013, el periodo que ninguna búsqueda usó, su media no es peor que la de la
     RNN W1.

   Si no se cumplen, se mantiene la RNN W1.
4. Antes de calcular el test se guarda la decisión con los hashes de las predicciones.
5. El test 2022-2026 se calcula una sola vez para todas las variantes y se publica tal cual.
   No se harán más rondas ajustadas mirando el test.

Elegir entre 15 combinaciones con datos de desarrollo favorece a la que tuvo más suerte en ese
periodo. Por eso se publican todas y se informa también 2008-2013 por separado.

## Reproducción

```sh
poetry run python -m oilcast -c configs/ronda3_w1.yaml extend-published
poetry run python -m oilcast -c configs/ronda3_single.yaml tune
poetry run python -m oilcast -c configs/ronda3_single.yaml evaluate
for c in ronda3_pinball ronda3 ronda3_productos; do
  poetry run python -m oilcast -c configs/$c.yaml multi-tune
  poetry run python -m oilcast -c configs/$c.yaml multi-evaluate
done
```

La variante con márgenes necesita antes `oilcast download-products`. La búsqueda se ejecutó
en la RTX 3090 por la duración.
