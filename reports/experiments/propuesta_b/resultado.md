# Propuesta B: ¿es real la mejora?

Propuesta B = RNN simple en los tres plazos: la red W1 publicada a 5 y 63 sesiones (20 variables
diarias) y la red de la ronda 4 a 252 sesiones (relación oro/crudo con atajo lineal). Se eligió
**después de ver el test 2022-2026**. [Comparación con las demás redes](../propuestas.md)

Comprobaciones con `python scripts/verificar_propuesta_b.py`.
[Resultados](verificacion.csv) · [Placebos](placebo.csv)

## 1. No hay trampas en el cálculo

| Comprobación | Resultado |
|---|---|
| Las variables calculadas con datos cortados en seis fechas (2008-2025) coinciden con las de la serie completa | Sí, las 20 variables y `oro_crudo` |
| Cada reentreno solo usa objetivos realizados antes de su fecha y predice fechas posteriores | Sí, en los tres plazos |
| Reentrenos de test el 1 de enero de cada año desde 2022 | Sí |
| La búsqueda de hiperparámetros solo validó en 2014-2021 | Sí, en los tres plazos |
| Recalcular el test desde las predicciones en bruto, sin el código de evaluación del proyecto | +0,62 %, +3,73 % y +9,08 %, igual que lo publicado |
| Usar el oro del día anterior (el oro cierra unas horas después que el Brent spot) | Igual: +7,5 % en la referencia lineal |

Las cifras de B están bien calculadas: no hay datos del futuro ni del test en el entrenamiento.

## 2. Solidez de cada plazo en el test

| Plazo | Mejora del MAE | Intervalo bootstrap 90 % | Semillas | Años positivos | Antes de 2022 |
|---|---:|---|---|---|---:|
| 5 sesiones | +0,6 % | +0,2 % a +1,1 % | 5 de 5 positivas (+0,4 % a +0,8 %) | 5 de 5 | −0,2 % |
| 63 sesiones | +3,7 % | +1,1 % a +6,7 % | 5 de 5 positivas (+1,8 % a +4,6 %) | 4 de 5 | −1,6 % |
| 252 sesiones | +9,1 % | −63 % a +46 % | 4 de 5 positivas (−1,1 % a +12,4 %) | 2 de 4 | +13,0 % |

El bootstrap remuestrea bloques de h sesiones, porque las previsiones consecutivas se solapan.

- **1 semana: real pero mínima.** Gana todos los años y con todas las semillas, aunque solo
  reduce el error en 2,5 céntimos por barril. Antes de 2022 no ganaba.
- **3 meses: sólida en 2022-2026.** El intervalo no incluye el cero y las cinco semillas
  ganan (p = 0,015 en error absoluto). Antes de 2022 perdía un 1,6 %, así que la ventaja
  podría ser propia de este periodo.
- **1 año: no se puede distinguir de la suerte en el test.** El intervalo va de −63 % a +46 %.
  La ganancia viene de los orígenes de 2022 (+45 %) y 2025 (+47 %); en 2023 pierde un 10 % y en
  2024 un 122 %. Sin los orígenes de 2025 pierde un 16 %. Su respaldo está antes de 2022: +19 %
  en 2008-2013 y +8 % en 2014-2021.

## 3. Placebo a un año

Se repite la misma regresión lineal recursiva que sirve de referencia (+7,5 % en el test) con
300 variables inventadas de cada tipo:

| Placebo | Mediana | Percentil 95 | Veces que iguala o supera +7,5 % |
|---|---:|---:|---:|
| Ruido puro con la misma persistencia | 0,0 % | +5,2 % | 3 % |
| Oro falso (paseo aleatorio) frente al Brent real | +0,8 % | +16,4 % | 20 % |

Por separado, el oro solo aporta 0,0 % y la vuelta del Brent a su media de 3 años, +2,6 %.
La relación entre ambos da +7,5 %, más que cada parte. Pero un oro inventado frente al Brent
real lo iguala una de cada cinco veces: con cinco años de test, ese resultado no basta para
probar que la relación oro/crudo funcione.

## Conclusión

Las cifras son reales: están bien calculadas y no tienen fugas. La mejora a 1 semana es segura
pero insignificante. La de 3 meses es sólida en 2022-2026, aunque no aparecía antes. La de 1
año depende de pocos episodios y en el test no se distingue del azar; su apoyo viene de los años
anteriores a 2022, donde se eligió la variable. Para la presentación, la frase honesta es:
"a 3 meses mejora de forma consistente en 2022-2026; a 1 año, mejoró en el test pero depende de
pocos episodios".
