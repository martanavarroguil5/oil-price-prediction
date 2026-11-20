# Mejoras implementadas y siguientes pasos

Estado del proyecto a 18/11/2026. Este registro permite explicar qué se ha añadido y qué
evidencia aporta cada cambio. En esa fecha la aplicación usaba la RNN W1;
desde el 20/11/2026 usa la propuesta B ([qué modelo se utiliza](../README.md#qué-modelo-se-utiliza)).

## Mejoras implementadas

| Área | Cambio realizado | Qué aporta |
|---|---|---|
| Interfaz y uso empresarial | Pantalla Decisiones, volumen equivalente, presupuesto, coste adicional y porcentaje comprado al inicio. | Traduce una previsión del Brent a una exposición económica bajo supuestos visibles. |
| Comparación de compras | Escenarios de compra inicial, compra al final y reparto entre ambos momentos; escenario manual y descarga del resumen. | Permite explorar cómo cambia el coste ante distintos precios finales. No establece una estrategia óptima ni acredita ahorros históricos. |
| Demostración histórica | Selección de red, plazo y origen; revelación posterior del precio realizado y comparación con el ingenuo. | Hace visibles los aciertos y fallos. El resultado no se incluye en la respuesta inicial, pero el periodo histórico ya se había consultado durante el desarrollo. |
| Evaluación de la rentabilidad | MAE y RMSE en puntos porcentuales de rentabilidad simple, junto a las métricas en dólares. | Distingue el error porcentual del impacto monetario. Las redes ya se entrenaban con rentabilidades logarítmicas. |
| Gráficos de mercado | Velas OHLC y volumen de los futuros continuos del Brent, con producto, fuente y fechas identificados. | Añade una lectura de las sesiones observadas. No se inventan apertura, máximo y mínimo para el Brent spot. |
| Distribución de errores | Violines del ingenuo, RNN, LSTM y GRU para los tres horizontes, en dólares y rentabilidad. | Muestra sesgo, dispersión y colas de error que un promedio puede ocultar. |
| Test frente a realidad | Dispersión observado–predicho en precio y rentabilidad, referencia ingenua opcional y diagonal de acierto perfecto. | Permite ver errores extremos y muestra el R² predictivo original, aunque sea negativo; no se ajusta una recta posterior para sustituirlo. |
| Evaluación de intervalos | Tabla de cobertura, anchura y puntuación de Winkler por modelo y horizonte. | Permite juzgar si un rango es informativo, además de comprobar cuántos resultados cubre. |
| Calibración experimental de intervalos | Siete variantes causales y una regla por red y horizonte fijada con objetivos anteriores a 2022; comparación histórica y cálculo actual en la app. | La RNN anual reduce Winkler un 14,08 % frente a su intervalo anterior, con incertidumbre estadística aún amplia. Es una mejora experimental y no sustituye automáticamente la calibración publicada. |
| Referencias sencillas | Comparación reproducible con Ridge y reversión a la media; elección exploratoria por horizonte usando desarrollo. | Comprueba si la complejidad de las redes aporta valor frente a alternativas sencillas. No ha demostrado una sustitución estable del modelo publicado. |

[Guía de demostración](guia-demostracion.md) ·
[Comparación con modelos sencillos](../reports/experiments/baselines/resultado.md) ·
[Métricas en rentabilidad](../reports/rentabilidad.csv)

## Qué resultados podemos afirmar

En el contraste histórico, la RNN publicada reduce el MAE en dólares un 0,89 % a 5 sesiones
y un 3,98 % a 63 sesiones; a 252 lo empeora un 6,82 %. En rentabilidad simple, las cifras
son +0,74 %, +3,09 % y −7,50 %. La nueva interfaz y los gráficos no cambian esas predicciones.

Los intervalos ya existían: las redes predicen cuantiles con pérdida pinball y calibración
conformal. A su evaluación y comunicación se ha añadido un experimento reproducible con
nuevas calibraciones. Las mejoras siguientes describen el método publicado antes de ese
experimento:

Para la RNN, la reducción relativa de la puntuación de Winkler frente al intervalo del
ingenuo es la siguiente; positivo significa mejor y negativo, peor:

| Horizonte | Mejora de Winkler |
|---|---:|
| 5 sesiones | +4,50 % |
| 63 sesiones | −16,12 % |
| 252 sesiones | −32,73 % |

Por tanto, el intervalo semanal aporta una mejora según esta medida; los trimestrales
y anuales actuales no. Aumentar su anchura puede elevar la cobertura, pero no reduce el
MAE o RMSE de la previsión central.

## Mejora experimental implementada: intervalos más informativos

**Implementada y evaluada retrospectivamente.** Se comparan CQR con ventanas recientes,
CQR normalizado por la anchura cruda y rangos basados en errores de la mediana. La regla
previa exige una mejora en desarrollo y mínimos de cobertura del 50 % y 80 %. Se usan
solo errores ya realizados en cada fecha y se conservan las medianas originales.

Para la RNN anual se seleccionó un intervalo basado en errores de la mediana. En el
contraste posterior, Winkler baja de 128,91 a 110,75 (mejora del 14,08 %), la anchura media
de 113,94 a 81,80 USD/barril y la cobertura del 89,7 % al 83,9 %. Sigue por detrás del
intervalo ingenuo (Winkler 97,12), y su IC95 orientativo de mejora incluye cero.
A 5 y 63 sesiones, la regla de desarrollo conserva el procedimiento publicado.
Las variantes de otras redes también se publican, incluida la LSTM trimestral, que empeora.

[Protocolo fijado antes de calcular resultados](experimento-intervalos.md) ·
[Comparación completa](../reports/experiments/intervals/resultado.md)

El periodo desde 2022 ya se ha consultado: no puede presentarse como un test ciego nuevo.
La confirmación independiente requerirá registrar previsiones y esperar nuevos resultados.
Una mejora en los intervalos tampoco demostraría por sí sola una mejora en la mediana ni
la rentabilidad de una política de compras.

## Redacción breve para la presentación

> Se ha ampliado Estrato con simulación presupuestaria, comparación de escenarios,
> demostración histórica y nuevas visualizaciones. La evaluación distingue precisión
> en dólares, error en rentabilidad y calidad de los intervalos. Una calibración elegida
> en desarrollo mejora experimentalmente el intervalo anual de la RNN; su confirmación
> con resultados prospectivos continúa pendiente.

Este contenido debe integrarse dentro del máximo de cinco diapositivas de contenido
establecido para la entrega. No constituye una presentación adicional.
