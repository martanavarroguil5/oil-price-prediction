# Experimentos de predicción del Brent

La aplicación utiliza una RNN simple para los tres horizontes. En la evaluación con orígenes desde 2022, esta red obtuvo menor error absoluto medio que la LSTM económica a 5 y 63 sesiones. A 252 sesiones, la LSTM económica consiguió **15,45 USD/barril de MAE**, frente a **16,50** del ingenuo y **17,63** de la RNN de la aplicación. La conservamos como candidata anual. El ingenuo, que pronostica el último precio conocido, obtuvo menor error cuadrático medio a ese plazo. [Comparativa de la aplicación](../reports/comparativa.md) · [Evaluación económica](../reports/results/experiments/economic_v2/resultado.md)

## Qué se probó y qué se decidió

| Experimento | Pregunta | Resultado y decisión |
|---|---|---|
| Búsqueda de hiperparámetros y variables externas | ¿Ayudan una búsqueda más amplia, las variables macroeconómicas o los índices de riesgo geopolítico? | La búsqueda ampliada con las variables base redujo un 3,4 % la pérdida de validación de los cuantiles y mejoró 8 de las 9 combinaciones de red y horizonte. Fue la configuración elegida. [Ronda 1](../reports/results/experiments/resumen.md) |
| Deriva y recalibración | ¿Conviene retirar la tendencia media aprendida y moderar el cambio previsto según los aciertos pasados? | La RNN simple con ambos cambios obtuvo una mejora media del MAE del 3,8 % frente al ingenuo antes de 2022. Esta regla de selección dio lugar al modelo de la aplicación. [Ronda 2](../reports/results/experiments/ronda2.md) |
| Ventanas A, B y C | Se compararon 60 sesiones diarias (A), 252 (B) y dos ramas diarias de 60 y 252 (C). | En el piloto con hiperparámetros fijos, ninguna comparación de B o C frente a A cumplió el criterio de mejora y estabilidad. A 63 sesiones, C con RNN simple superó a A en dos de cuatro bloques, lo que motivó una prueba específica. [Protocolo](../reports/results/experiments/pilot_abc/protocol.json) · [Comparaciones](../reports/results/experiments/pilot_abc/comparison.csv) |
| Ajuste de A y C a 63 sesiones | ¿Cambia la conclusión si ambas variantes reciben el mismo presupuesto de búsqueda? | La mejor familia seleccionada para A fue GRU y para C fue RNN simple. En desarrollo, A obtuvo 9,532 de MAE, C obtuvo 9,753 y el ingenuo 9,602. C perdió frente a A en los cuatro bloques. [Criterio](../reports/results/experiments/ac63_search/protocol.json) · [Decisión](../reports/results/experiments/ac63_search/decision.json) |
| Señales adelantadas | ¿Anticipan el precio los avisos de huracán en el golfo de México o las divisas de países exportadores? | A cinco sesiones, añadir una señal redujo el MAE en menos de 0,001 USD/barril en la mejor comparación con su propia red de control. Dar acceso directo a la salida empeoró el MAE en las seis comparaciones. [Pruebas con señales, controles y episodios reales](../reports/results/experiments/leading_signals/resultado.md) |
| Refinados y futuros a 63 sesiones | ¿Ayudan los márgenes de refino, el objetivo en dólares o la pendiente entre dos vencimientos del WTI? | En una prueba exploratoria anterior a 2022, sumar la pendiente de futuros a la GRU redujo su MAE de 9,532 a 9,335. Ganó en dos de cuatro bloques. La serie pública utilizada termina en abril de 2024. [Comparación y disponibilidad de datos](../reports/results/experiments/products63/resultado.md) |
| Primer híbrido económico | ¿Puede la red aprender el error de una previsión económica calculada de antemano? | A cinco sesiones, la LSTM híbrida obtuvo 2,1953 de MAE y la misma LSTM entrenada directamente con las variables obtuvo 2,1916. A 252 apareció una mejora exploratoria al usar toda la corrección neuronal, concentrada en dos de cuatro bloques. [Controles y resultados del primer híbrido](../reports/results/experiments/economic_hybrid/resultado.md) |
| Combinación recurrente de pronósticos económicos | ¿Puede cada red aprender cuánto confiar cada día en distintos indicadores? | La LSTM obtuvo el menor MAE relativo medio en desarrollo. Quedó seleccionada para los tres horizontes antes de puntuar el tramo desde 2022. [Diseño y bibliografía](experimento-economico-v2.md) · [Selección](../reports/results/experiments/economic_v2/selection.json) |
| Amplitud del cambio previsto | ¿Deja margen la combinación económica para anticipar movimientos mayores? | A cinco días, el 95,7 % de los precios objetivo desde 2022 quedó fuera del rango de las cinco propuestas centrales. A un año, ajustar la amplitud a partir de los errores conocidos empeoró las tres redes. [Diagnóstico, prueba de amplitud y combinación con la red directa](../reports/results/experiments/economic_amplitude/resultado.md) |

## Cómo se evaluó la variante económica

Cada día, cuatro modelos estiman por separado cuánto cambiará el Brent en dólares por barril a cada plazo. Usan el mercado físico y los futuros, los márgenes de refino, la actividad y las divisas, y la desviación del precio respecto a su historia. La quinta previsión es un cambio de cero. A partir de 60 sesiones diarias, la RNN simple, la LSTM o la GRU asigna pesos a las cinco previsiones. Comparar esa combinación con cada componente y con su media permite medir qué aporta la red. [Arquitectura y regla de selección](experimento-economico-v2.md)

Las tres redes usaron los mismos datos, horizontes y presupuesto. Los hiperparámetros se ajustaron en dos ventanas anteriores a 2014. La familia se eligió con los objetivos realizados en 2014–2021, dando el mismo peso a cada horizonte. La evaluación posterior usa orígenes desde enero de 2022 y precios objetivo conocidos hasta el 3 de noviembre de 2026. En cada reentrenamiento solo podían entrar ejemplos cuyo precio objetivo ya se conocía. Ese tramo histórico ya se había consultado para otros modelos del proyecto. [Búsqueda](../reports/results/experiments/economic_v2/search.csv) · [Reentrenamientos](../reports/results/experiments/economic_v2/fits.csv)

### MAE por horizonte en las mismas fechas de test

Los valores son errores absolutos medios en USD/barril. Una cifra menor indica mayor precisión. La columna del híbrido anterior utiliza su LSTM con la corrección calibrada.

| Sesiones | Previsiones | Ingenuo | RNN de la app | LSTM híbrida anterior | RNN económica | LSTM económica | GRU económica |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.218 | 3,987 | 3,951 | 3,992 | 3,993 | 3,995 | 3,995 |
| 63 | 1.160 | 11,944 | 11,469 | 12,552 | 11,960 | 11,850 | 11,760 |
| 252 | 971 | 16,502 | 17,627 | 21,977 | 16,041 | 15,447 | 15,618 |

A 63 sesiones la LSTM económica anticipó un movimiento absoluto medio de **1,08 USD/barril** y el Brent se movió **11,94**. Eso cuantifica por qué la curva prevista queda cerca del último precio. La gráfica sitúa cada pronóstico en su fecha objetivo. [Cambios previstos y observados](../reports/results/experiments/economic_v2/cambios_63.png) · [Métricas completas con MSE, RMSE, R², dirección y cobertura](../reports/results/experiments/economic_v2/metrics.csv)

### Qué significa ganar al ingenuo a doce meses

La LSTM económica redujo el MAE un **6,4 %**, equivalente a **1,05 USD/barril**. El RMSE, más sensible a los errores grandes, fue **23,51** para la LSTM y **22,53** para el ingenuo. En el 10 % de las mayores variaciones anuales, el MAE de la LSTM subió a **56,23** frente a **52,05** del ingenuo. Los 971 pronósticos se emitieron en sesiones consecutivas y sus horizontes se solapan. El contraste de Diebold–Mariano sobre error absoluto dio *p* = 0,512. La ganancia media observada tiene, por tanto, evidencia limitada de persistencia en otros periodos. [Errores en movimientos grandes](../reports/results/experiments/economic_v2/large_moves.csv) · [Resultados por año](../reports/results/experiments/economic_v2/annual.csv)

El componente físico y de futuros obtuvo por sí solo un MAE anual de **15,156**, frente a **15,846** de la media de los cinco pronósticos y **15,447** de la LSTM. La ponderación aprendida mejoró la media simple. El componente físico aislado fue más preciso en este tramo. [Resultados por componente](../reports/results/experiments/economic_v2/resultado.md)

## Reproducción y seguimiento

Desde la raíz del repositorio y con el entorno Python del proyecto activo, `python -m scripts.predict_economic_v2` genera las tres previsiones con los pesos guardados. El informe de evaluación incluye los comandos de entrenamiento y de cálculo de métricas. La ejecución en la GPU registró 54 ajustes para seleccionar hiperparámetros y 243 ajustes de evaluación de las redes económicas, con tres semillas. [Ejecución e identificación de los datos](../reports/results/experiments/economic_v2/run.json) · [Pesos y coeficientes](../reports/results/experiments/economic_v2/models/) · [Pasos de reproducción](../reports/results/experiments/economic_v2/resultado.md)

El 17 de noviembre de 2026 se registró, por separado, una previsión prospectiva de la RNN publicada a 63 sesiones, junto con el ingenuo y dos ajustes de sesgo. Se evaluará con la 63.ª observación del Brent posterior al precio de origen del 11 de noviembre. [Predicciones y regla de evaluación fijadas al emitirlas](../reports/results/prospectivo/2026-11-17-63d.json)
