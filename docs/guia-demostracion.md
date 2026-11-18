# Demostración de Estrato

La propuesta de valor es explorar la exposición de un presupuesto al Brent y contrastar
las previsiones con evidencia histórica. La aplicación no ejecuta compras ni recomienda
un reparto óptimo. El Brent es una referencia; no representa por sí solo una factura real
de combustible o un precio ejecutable de cobertura.

## Un recorrido de cuatro minutos

1. En **Decisiones**, elegir tres meses y explicar qué fecha de origen utiliza la previsión.
   Los tres plazos muestran su mejora histórica frente a mantener el precio constante.
   El año no mostró ventaja: esa limitación permanece visible.
2. Introducir un volumen equivalente y presupuesto. Mover el porcentaje comprado al precio
   de referencia del 0 al 100 %. Observar el coste central, sus límites y el impacto por cada
   dólar de movimiento del precio final. Abrir los supuestos. El reparto del 100 % fija el
   coste en esta simulación, pero no incluye almacenamiento ni financiación.
3. Comparar los tres repartos y mover el escenario manual. Los límites del modelo no
   cambian al mover ese escenario. La marca ámbar significa que un coste supera el
   presupuesto; no es una recomendación de compra. Guardar un resumen para dejar constancia
   de los supuestos explorados.
4. En **Prueba histórica**, elegir red, plazo y fecha. Antes de revelar, mostrar mediana,
   intervalo y referencia ingenua. Pulsar **Ver qué ocurrió**. Comparar los errores de ambas
   previsiones y comprobar si el resultado quedó dentro del intervalo. Recorrer fechas
   distintas: un caso aislado no demuestra una ventaja, y el periodo ya se había consultado.
5. En **Validación**, mostrar que la evaluación en dólares y la evaluación en rentabilidad
   contestan preguntas distintas. En **Redes**, explicar brevemente las tres arquitecturas;
   en **Ciclo histórico**, situar los principales cambios de mercado.

## Qué significan los intervalos

Son intervalos de predicción con nivel nominal del 80 %, no intervalos de confianza para
la media o para el error del modelo. Su cobertura histórica se muestra por horizonte y
puede diferir del 80 %. A mayor plazo suele haber más incertidumbre. Un intervalo ancho
no hace mejor una previsión central: el MAE y el RMSE comparan puntos. Para los intervalos
se necesitan cobertura, anchura y una puntuación como Winkler (incluida en el informe).

Los gráficos muestran la mediana y los límites en la fecha objetivo. No interpolan esos
puntos como si el modelo hubiese calculado una trayectoria diaria.

## Precio, rentabilidad y presupuesto

El objetivo de entrenamiento es `log(P[t+h] / P[t])`. La salida central se convierte a
precio mediante `P[t] * exp(predicción)`. Para evaluar rentabilidades simples se usa
`exp(predicción) - 1`; la referencia ingenua es cero.

El MAE en rentabilidad se expresa en puntos porcentuales. El MAE en dólares equivale a
ponderar el error de rentabilidad simple por el precio de origen, de modo que los dos
criterios pueden ordenar los modelos de forma distinta. Cambiar de unidad no añade señal
ni resuelve por sí solo el horizonte anual. La comparación nueva no modifica el modelo
publicado ni vuelve a seleccionar sus hiperparámetros.

La simulación calcula `volumen * (porcentaje inicial * referencia + porcentaje final *
precio final + coste adicional por barril)`. El escenario central utiliza la mediana
prevista, no un coste esperado. Los escenarios manuales no tienen probabilidad asignada.
