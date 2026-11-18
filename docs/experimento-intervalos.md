# Mejora de los intervalos: protocolo

Protocolo fijado el 18/11/2026 antes de calcular este experimento. Se reutilizan las
predicciones W1 de las tres redes, sin reentrenar y sin cambiar sus medianas. La familia
publicada sigue siendo RNN W1. Se estudian intervalos nominales del 50 % y 80 %.

## Variantes fijadas

1. `current`: intervalos publicados, CQR con historia expansiva.
2. `cqr_504` y `cqr_1260`: CQR con las últimas 504 o 1.260 puntuaciones ya realizadas.
3. `scaled_expanding` y `scaled_1260`: CQR cuya puntuación se divide por la semianchura
   del intervalo crudo de esa previsión. El ajuste vuelve a multiplicarse por la
   semianchura de la previsión actual; permite adaptar la corrección a la escala que
   anticipa la red. No se utiliza volatilidad futura.
4. `residual_expanding` y `residual_1260`: intervalo simétrico alrededor de la mediana,
   usando el cuantil corregido del error absoluto de rentabilidad logarítmica. No usa
   la anchura propuesta por la red.

Se necesitan 100 errores realizados para calibrar. Antes se conserva el intervalo crudo.
Las variantes contienen la mediana y anidan el intervalo del 50 % dentro del del 80 %.
La referencia `current` se conserva exactamente como está guardada. En cada origen solo
se usan resultados con fecha objetivo estrictamente anterior a ese origen; la ventana
se define en número de puntuaciones realizadas, no en años independientes.

## Elección previa al contraste histórico

El periodo 2014–2015 aporta calentamiento. Se elige una variante por red y horizonte con
orígenes desde 2016 y objetivos realizados antes del 1/1/2022. Todos se comparan en las
mismas fechas y en dólares por barril.

Una alternativa solo es elegible si cumple en desarrollo:

- mejora la puntuación de Winkler del intervalo 80 % al menos un 2 % frente al actual;
- cobertura del 80 % de al menos el 77 %, y del 50 % de al menos el 47 %;
- Winkler del 50 % no más de un 2 % peor que el actual.

Entre las elegibles se elige la menor puntuación de Winkler del 80 %. Dentro de una
diferencia relativa del 0,5 % se prefiere el orden de la lista de configuración. Si
ninguna pasa los requisitos, se conserva `current`. Los márgenes no son contrastes de
significación. La decisión se guarda antes de calcular métricas del periodo desde 2022.

## Evaluación y límites

Se publicarán cobertura, anchura y Winkler del 50 % y 80 %, resultados por año y todas
las variantes, incluso cuando el ganador de desarrollo empeore después. El contraste
desde 2022 ya se consultó en otras rondas: no es un test independiente nuevo. Las redes
ya usaron los bloques de desarrollo para sus hiperparámetros; tampoco es una nueva CV
anidada de toda la cadena. No se harán cambios en los candidatos mirando ese contraste.

El intervalo orientativo del 95 % de la mejora en Winkler 80 usa 1.000 remuestreos
circulares por bloques de h previsiones, semilla 42. No corrige exploración ni múltiples
comparaciones. A 252 sesiones hay pocos bloques. La dependencia temporal impide atribuir
automáticamente a esta calibración la garantía de cobertura de un problema intercambiable.

La evaluación y la comparación se incorporarán a la app como una mejora experimental,
con el método publicado identificado. Los pesos y medianas no cambian. La aplicación
no sustituirá silenciosamente la calibración publicada. Confirmar una mejora estable
requerirá nuevas predicciones prospectivas y sus resultados.

## Reproducción

```sh
poetry run python -m oilcast -c configs/intervals.yaml improve-intervals
```

Salidas independientes en `reports/experiments/intervals/`: decisión en desarrollo,
métricas, comparación anual, predicciones e informe, con configuración y hashes de las
entradas. Los intervalos experimentales de una previsión actual se calcularán con la
misma función y solo resultados ya disponibles al cierre del último dato.
