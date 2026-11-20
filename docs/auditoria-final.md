# Auditoría y decisión de cierre

> Revisión anterior a la propuesta B: describe la versión con la RNN W1 en los tres plazos.
> La aplicación usa ahora la propuesta B ([qué modelo se utiliza](../README.md#qué-modelo-se-utiliza)).

Revisión del 19/11/2026 sobre los datos y predicciones ya guardados. No se reentrena ni
se elige otra familia a partir del periodo de test. El propósito es detectar errores
comprobables y evaluar la sensibilidad de los resultados a los años incluidos.

## Qué se ha comprobado

- Fechas objetivo exactamente a 5, 63 y 252 posiciones del calendario observado del Brent.
- Particiones temporales con objetivos de entrenamiento realizados antes de cada corte.
- Escalado de entradas y objetivo ajustado con entrenamiento, según el código revisado.
- Mismos orígenes de test para las tres redes dentro de cada horizonte, sin duplicados.
- Reconstrucción del precio objetivo desde su rentabilidad logarítmica y precio de origen.
- MAE, RMSE y R² recalculados desde las predicciones, coincidentes con las métricas publicadas.

Los controles programáticos pasan en las nueve combinaciones. La discrepancia máxima de
reconstrucción del precio es inferior a 0,0003 USD/barril, coherente con el redondeo de los
CSV. Esta auditoría no reproduce todos los entrenamientos ni certifica la disponibilidad
histórica exacta de cada dato.

## ¿Depende la mejora de un año concreto?

Se recalcula la reducción del MAE frente al ingenuo excluyendo un año cada vez. Es un
análisis de sensibilidad retrospectivo, no una validación independiente ni un intervalo
de confianza. Hay previsiones solapadas y años parciales en el extremo de la muestra.

| Modelo y horizonte | Mejora conjunta | Años positivos | Mejora al excluir un año: mínimo–máximo |
|---|---:|---:|---:|
| RNN · 5 sesiones | +0,89 % | 4/5 | +0,62 % a +1,35 % |
| RNN · 63 sesiones | +3,98 % | 4/5 | +2,50 % a +5,03 % |
| GRU · 63 sesiones | +5,37 % | 4/5 | −0,17 % a +9,44 % |
| RNN · 252 sesiones | −6,82 % | 1/4 | −22,51 % a +4,87 % |

La ventaja trimestral de la RNN no desaparece al retirar individualmente ninguno de los
años. La GRU presenta mejor resultado conjunto, pero mayor sensibilidad al periodo. Esto
ayuda a interpretar la familia publicada; no modifica retrospectivamente su elección.

## Decisiones de diseño y límites

La búsqueda de hiperparámetros minimiza la pérdida pinball de los cuantiles, no el MAE
monetario de la mediana. Es una decisión coherente con la salida probabilística, aunque
no garantiza el mejor valor central. Optimizar explícitamente el error central en
validación sería un experimento distinto, no una corrección de un fallo demostrado.

Las series se alinean por fecha de observación. Los archivos no contienen todos los
instantes de publicación ni versiones históricas de las series. Por tanto, no se puede
certificar con ellos una simulación operativa exacta en tiempo real. No se han inventado
retardos para intentar cambiar los resultados. Se han corregido los comentarios y la
documentación para que esta limitación quede explícita.

## Decisión recomendada

Conservar la RNN W1 publicada y cerrar esta versión para la entrega, explicando su ventaja
modesta a tres meses, el empate aproximado a una semana y la limitación anual. Presentar
la nueva calibración anual como experimental, distinguiendo mejora del intervalo y
precisión de la mediana.

No se recomienda otra ronda abierta de ajustes contra el periodo ya consultado. Para
investigación posterior, priorizar información nueva (curva de futuros e inventarios con
fechas de publicación), o un objetivo central distinto fijado en desarrollo. Ampliar el
registro prospectivo iniciado y esperar sus resultados es la vía para una confirmación
independiente. Nada de esta auditoría demuestra que se haya alcanzado el límite posible.

## Reproducción

```sh
poetry run python scripts/audit_results.py
```

[Controles e identificación de entradas](../reports/auditoria-validacion.json) ·
[Sensibilidad completa de las nueve combinaciones](../reports/auditoria-estabilidad.csv)
