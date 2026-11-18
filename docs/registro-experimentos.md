# Registro de experimentos del Brent

Se eligió una familia para los tres horizontes con el promedio de las mejoras porcentuales del MAE frente al ingenuo. Así, el error anual, normalmente más alto en dólares, recibe el mismo peso que el semanal y el trimestral. Los resultados desde 2022 sirven para comprobar cómo se comportó después la familia elegida. [Regla de decisión](experimento-ronda2.md)

| Experimento | Qué se observó | Decisión |
|---|---|---|
| Búsqueda ampliada y variables macroeconómicas | Ampliar la búsqueda redujo un 3,4 % la pérdida de validación de los cuantiles y mejoró ocho de nueve combinaciones. Las variables adicionales no aportaron una mejora estable. | Conservar las 20 variables diarias y 80 ensayos por red y horizonte. |
| Deriva y recalibración | W0 modera el cambio previsto según errores pasados. W1 también retira la deriva media del entrenamiento. Antes de 2022, RNN W1 mejoró al ingenuo un **3,82 %** de media. La mejor W0, LSTM, lo mejoró un **0,28 %**. | Integrar RNN W1 en la app para los tres plazos. |
| Ventanas y señales adelantadas | Ampliar la ventana o sumar una rama larga no aportó una ganancia estable. Los avisos de huracán y las divisas apenas cambiaron el MAE semanal. | Mantener la ventana de 60 sesiones. |
| Modelo económico | A 252 sesiones, la LSTM económica obtuvo **15,447 USD/barril** de MAE frente a **16,502** del ingenuo. Su RMSE fue mayor, **23,51** frente a **22,53**. | Conservar el resultado como comparación anual. |
| Combinaciones a 63 sesiones | Mezclar las ramas directa y económica, entrenarlas por separado o ponderarlas por aciertos recientes dejó errores mayores que los **11,469** de la app. Añadir mercados diarios llevó la GRU a **12,134**. | Mantener estas variantes como experimentos. |
| Afinado de la GRU W0 | La candidata recalibrada obtuvo **11,302** de MAE frente a **11,099** de la original. La red afinada por sí sola obtuvo **11,195**. | Conservar la GRU W0 original como referencia trimestral. |
| W0 con las tres familias y el mismo presupuesto | Entre la app y las tres W0, en test ganó la app a 5 sesiones, la GRU W0 a 63 y la LSTM W0 a 252. La GRU W0 tuvo la mejor media posterior del grupo (**+1,33 %** frente al ingenuo). | Documentar los ganadores por plazo y mantener la elección realizada en desarrollo. |

La ventaja más importante de la app en desarrollo se encontraba a 252 sesiones (**+13,52 %** frente al ingenuo). En test pasó a **−6,82 %**. A 63 sesiones la GRU W0 obtuvo una reducción del MAE de **7,07 %** frente al ingenuo. Esos resultados explican por qué la familia elegida para los tres horizontes y la red con menor error trimestral son distintas. [Comparación de variantes en desarrollo y test](../reports/experiments/seleccion.csv) · [Resultados anuales de W0](../reports/experiments/w0/resultado.md) · [Resultados del modelo económico](../reports/experiments/economico.csv)

Se conserva también una previsión trimestral emitida con el último precio disponible del 11 de noviembre de 2026. Quedó registrada junto al ingenuo y dos correcciones antes de conocer el precio objetivo. [Previsión y regla de evaluación](../reports/results/prospectivo/2026-11-17-63d.json)
