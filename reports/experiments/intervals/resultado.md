# Mejora experimental de los intervalos

[Protocolo](../../../docs/experimento-intervalos.md). Decisiones por red y horizonte con datos de desarrollo; el periodo posterior ya se había consultado. Las medianas no cambian. No es un test independiente nuevo ni una sustitución automática de producción.

| Red | Sesiones | Elegido en desarrollo | Cobertura 80 % actual → elegido | Anchura actual → elegida ($) | Winkler actual → elegido | Mejora Winkler80 | IC95 orientativo |
|---|---|---|---|---|---|---|---|
| rnn | 5 | current | 79.0 % → 79.0 % | 11.04 → 11.04 | 19.03 → 19.03 | +0.00 % | [+0.00 %, +0.00 %] |
| rnn | 63 | current | 87.7 % → 87.7 % | 49.66 → 49.66 | 75.72 → 75.72 | +0.00 % | [+0.00 %, +0.00 %] |
| rnn | 252 | residual_expanding | 89.7 % → 83.9 % | 113.94 → 81.80 | 128.91 → 110.75 | +14.08 % | [-3.48 %, +31.60 %] |
| lstm | 5 | current | 75.9 % → 75.9 % | 10.83 → 10.83 | 18.91 → 18.91 | +0.00 % | [+0.00 %, +0.00 %] |
| lstm | 63 | scaled_expanding | 85.7 % → 83.7 % | 45.53 → 47.18 | 72.82 → 77.38 | -6.27 % | [-8.92 %, -3.98 %] |
| lstm | 252 | residual_expanding | 88.7 % → 87.0 % | 104.82 → 95.28 | 125.67 → 117.21 | +6.73 % | [+1.83 %, +13.53 %] |
| gru | 5 | current | 81.1 % → 81.1 % | 11.98 → 11.98 | 19.12 → 19.12 | +0.00 % | [+0.00 %, +0.00 %] |
| gru | 63 | residual_expanding | 88.2 % → 87.0 % | 48.62 → 39.80 | 75.32 → 65.05 | +13.63 % | [+8.05 %, +20.20 %] |
| gru | 252 | residual_expanding | 95.3 % → 90.4 % | 133.76 → 93.70 | 137.71 → 106.07 | +22.98 % | [+1.16 %, +39.09 %] |

Positivo significa menor puntuación de Winkler, no menor MAE de la mediana. El IC95 usa bloques de h previsiones; es orientativo, con pocos bloques a un año, y no corrige exploración ni comparaciones múltiples.

La cobertura nominal es del 80 %; más cobertura no equivale siempre a un mejor intervalo. El criterio de selección incorpora también cobertura y Winkler del 50 %.

## Archivos

- `decision.json`: variantes elegidas antes de calcular test, configuración y hashes.
- `development.csv` y `test.csv`: todos los candidatos.
- `selected.csv`: actual frente a elegido y comparación con el ingenuo.
- `annual.csv`: comparación por año del origen.
- `predictions/`: intervalos elegidos, con medianas originales intactas.
- `app.json`: resumen para la comparación experimental en la aplicación.

La aplicación mantiene identificada la calibración publicada. La opción experimental no demuestra una garantía de cobertura futura. Si una variante seleccionada en desarrollo empeora después, ese resultado se conserva y se muestra.

Selección: {"rnn": {"5": "current", "63": "current", "252": "residual_expanding"}, "lstm": {"5": "current", "63": "scaled_expanding", "252": "residual_expanding"}, "gru": {"5": "current", "63": "residual_expanding", "252": "residual_expanding"}}
