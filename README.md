# Predicción del precio del petróleo (Brent) con RNN, LSTM y GRU

Caso de uso III de Aprendizaje Automático. Predicción del Brent a corto (5 días hábiles),
medio (63) y largo plazo (252) con RNN simple, LSTM y GRU, comparadas con un modelo ingenuo,
e integración del mejor modelo en una app Streamlit.

Enunciado: [`docs/assignment/ENUNCIADO.pdf`](docs/assignment/ENUNCIADO.pdf)

## Ejecución

```bash
poetry install
poetry run python scripts/download_data.py   # genera data/petroleo_dataset.csv
```

## Estructura prevista

```text
docs/        enunciado y material de apoyo
scripts/     descarga de datos (proporcionado)
data/        petroleo_dataset.csv (diario, 2000-hoy, 20 columnas)
notebooks/   EDA, preprocesado y comparativa de modelos   (pendiente)
src/         limpieza, ventanas, modelos, evaluación        (pendiente)
app/         aplicación Streamlit                           (pendiente)
```

## Estado

- [x] Descarga de datos
- [ ] EDA y limpieza (huecos, outliers, WTI negativo)
- [ ] Escalado (solo con train) y ventanas de 60 días
- [ ] RNN / LSTM / GRU con validación walk-forward + baseline ingenuo
- [ ] Tabla comparativa (MAE, RMSE, mejora vs ingenuo, parámetros, tiempo)
- [ ] App Streamlit (3 horizontes, intervalo 80 %, explicación, ciclo económico, reentrenado)
- [ ] Presentación de máximo 5 diapositivas
