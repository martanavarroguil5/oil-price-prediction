# Una familia para el Brent a 5, 63 y 252 sesiones

La regla temporal elige la **RNN W1 de la app** como modelo único. Entre las tres W0 elige la **LSTM W0**. Cada familia utiliza un ajuste por horizonte. [Protocolo previo](../../../../docs/repeticion-w0-familias.md) · [Selección](selection.json)

## Elección con objetivos conocidos antes de 2022

Reducción porcentual del MAE respecto a mantener el último precio. La columna final da la misma importancia a cada plazo. Los mismos años sirvieron para elegir hiperparámetros en los estudios originales. Esta tabla mide el rendimiento en desarrollo. [Datos de desarrollo](development.csv)

| Familia | 5 sesiones | 63 sesiones | 252 sesiones | Media |
|---|---:|---:|---:|---:|
| RNN W0 | -0,09 % | -1,69 % | +0,01 % | -0,59 % |
| LSTM W0 | -0,02 % | +0,87 % | +0,00 % | +0,28 % |
| GRU W0 | -0,19 % | -0,46 % | +0,00 % | -0,22 % |
| RNN W1 de la app | -0,22 % | -1,85 % | +13,52 % | +3,82 % |
| Ingenuo | +0,00 % | +0,00 % | +0,00 % | +0,00 % |

## Qué pasó desde 2022

El MAE se expresa en USD por barril. En cada fila se destaca el menor error sobre las mismas emisiones y precios reales. La media relativa conjunta de test es descriptiva, ya que este periodo se consultó en experimentos anteriores. [Métricas de test](test.csv)

| Plazo | Emisiones | RNN W0 | LSTM W0 | GRU W0 | RNN W1 de la app | Ingenuo |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 1218 | 3,952 | 3,983 | 3,985 | **3,951** | 3,987 |
| 63 | 1160 | 11,725 | 12,041 | **11,099** | 11,469 | 11,944 |
| 252 | 971 | 17,160 | **16,084** | 17,020 | 17,627 | 16,502 |

| Familia | Mejora media del MAE en test |
|---|---:|
| RNN W0 | -0,42 % |
| LSTM W0 | +0,60 % |
| GRU W0 | +1,33 % |
| RNN W1 de la app | -0,65 % |
| Ingenuo | +0,00 % |

La RNN de la app gana a cinco sesiones por 0,00023 USD/barril frente a la RNN W0. A 63 sesiones la GRU W0 reduce el MAE en 0,845 frente al ingenuo y en 0,370 frente a la app. En el contraste de errores absolutos los valores de *p* son 0,038 y 0,327 respectivamente. A 252 sesiones la LSTM W0 reduce el MAE un 2,53 % frente al ingenuo, aunque su RMSE sube de 22,527 a 23,179 y la diferencia de MAE tiene *p* = 0,739. Son contrastes individuales en un periodo evaluado anteriormente.

La GRU W0 tiene la mayor mejora relativa media en test (+1,33 %), concentrada en 63 sesiones. La RNN de la app fue elegida en desarrollo principalmente por su ventaja a un año (+13,52 %), que pasó a una pérdida del 6,82 % en el test. En la entrega mantendría la familia elegida y documentaría este cambio de comportamiento. El corte de 2022 permite distinguir el rendimiento que motivó la elección del observado después.

### Comparación anual

| Sesiones | Año de emisión | Emisiones | RNN W0 | LSTM W0 | GRU W0 | RNN W1 de la app | Ingenuo |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2022 | 252 | 5,273 | 5,346 | 5,347 | **5,267** | 5,352 |
| 5 | 2023 | 251 | 2,988 | 3,008 | 3,006 | **2,986** | 3,004 |
| 5 | 2024 | 254 | 2,313 | 2,359 | 2,356 | **2,311** | 2,359 |
| 5 | 2025 | 253 | 2,362 | 2,388 | 2,385 | **2,361** | 2,386 |
| 5 | 2026 (parcial) | 208 | 7,448 | **7,433** | 7,449 | 7,461 | 7,453 |
| 63 | 2022 | 252 | 12,620 | 15,060 | **10,619** | 11,803 | 12,938 |
| 63 | 2023 | 251 | 8,450 | 8,630 | 8,857 | **8,342** | 8,621 |
| 63 | 2024 | 254 | 4,540 | **3,884** | 4,146 | 4,350 | 4,817 |
| 63 | 2025 | 253 | 9,075 | 8,699 | **8,457** | 9,168 | 9,068 |
| 63 | 2026 (parcial) | 150 | 32,339 | 32,128 | **31,887** | 32,075 | 32,755 |
| 252 | 2022 | 252 | 23,074 | 21,113 | 22,549 | 24,519 | **19,318** |
| 252 | 2023 | 251 | 7,662 | **7,137** | 7,883 | 7,922 | 7,608 |
| 252 | 2024 | 254 | 10,045 | 6,681 | 8,157 | **4,103** | 11,430 |
| 252 | 2025 (parcial) | 214 | 29,780 | 31,818 | 31,747 | 36,947 | **29,637** |

El último precio objetivo es el del 3 de noviembre de 2026. Las últimas emisiones evaluables corresponden al 27 de octubre de 2026 para 5 sesiones, al 5 de agosto de 2026 para 63 y al 4 de noviembre de 2025 para 252. A un año, los errores de emisiones cercanas comparten gran parte de su horizonte. [Resultados anuales y fechas](annual.csv)

### Precisión y coste por familia y plazo

Los tiempos de W0 son la mediana de cinco entrenamientos individuales en el corte de enero de 2022. Para la app se cita el tiempo publicado sobre el mismo corte. Cada predicción neuronal promedia cinco redes. Los parámetros corresponden a un miembro. El CSV de test incluye el MSE y los contrastes de Diebold–Mariano sobre el error absoluto frente al ingenuo y la app, con corrección por solapamiento de h − 1 retardos. [Tiempos de cada ajuste](fits.csv) · [Métricas y contrastes](test.csv)

| Sesiones | Familia | MAE | RMSE | R² | Mejora MAE | Parámetros | Segundos |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | RNN W0 | 3,952 | 5,701 | 0,876 | +0,88 % | 5959 | 0,87 |
| 5 | LSTM W0 | 3,983 | 5,732 | 0,875 | +0,09 % | 812807 | 0,33 |
| 5 | GRU W0 | 3,985 | 5,735 | 0,875 | +0,06 % | 1004807 | 0,39 |
| 5 | RNN W1 de la app | 3,951 | 5,703 | 0,876 | +0,89 % | 5959 | 0,87 |
| 5 | Ingenuo | 3,987 | 5,738 | 0,875 | +0,00 % | — | — |
| 63 | RNN W0 | 11,725 | 17,298 | -0,187 | +1,83 % | 22599 | 0,21 |
| 63 | LSTM W0 | 12,041 | 17,901 | -0,272 | -0,82 % | 89031 | 0,21 |
| 63 | GRU W0 | 11,099 | 16,780 | -0,117 | +7,07 % | 1004807 | 0,88 |
| 63 | RNN W1 de la app | 11,469 | 17,117 | -0,163 | +3,98 % | 22599 | 0,21 |
| 63 | Ingenuo | 11,944 | 17,437 | -0,207 | +0,00 % | — | — |
| 252 | RNN W0 | 17,160 | 23,751 | -1,742 | -3,99 % | 336135 | 1,87 |
| 252 | LSTM W0 | 16,084 | 23,179 | -1,611 | +2,53 % | 341895 | 0,22 |
| 252 | GRU W0 | 17,020 | 24,393 | -1,892 | -3,14 % | 66887 | 0,08 |
| 252 | RNN W1 de la app | 17,627 | 27,061 | -2,559 | -6,82 % | 204551 | 0,73 |
| 252 | Ingenuo | 16,502 | 22,527 | -1,466 | +0,00 % | — | — |

## Reproducción

En la GPU se completaron 390 entrenamientos en 206 segundos, reutilizando los 80 ensayos de búsqueda originales por combinación. Cada familia recibió 45 ajustes a 5 sesiones, 45 a 63 y 40 a 252. Se verificaron los nueve historiales de Optuna, los mismos orígenes y vencimientos, y cada precio observado contra la serie del Brent. La mayor diferencia absoluta entre cualquier pronóstico reproducido y el archivado fue 0,000076 USD por barril. [Dispositivo, hiperparámetros y presupuesto](run.json) · [Diferencias de reproducción](reproduction.csv)

Desde la raíz del repositorio, el primer comando se ejecuta en la GPU. Los otros dos usan sus archivos de resultados y las referencias publicadas.

```sh
OILCAST_DEVICE=cuda python -m scripts.repeat_w0_families
python -m scripts.compare_w0_families select
python -m scripts.compare_w0_families test
```
