# Atajos del proyecto. `make help` lista los objetivos.
PY      := poetry run python
OILCAST := $(PY) -m oilcast
CONFIG  ?= configs/default.yaml

.PHONY: help install data download build tune evaluate baselines returns diagnostics improve-intervals timing summarize train-final report \
        forecast app static test lint format smoke notebooks

help:  ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## Instala dependencias (incluye desarrollo)
	poetry install

data: download build  ## Descarga y construye el dataset

download:  ## Descarga series en bruto a data/raw (FRED, Yahoo, OPEP)
	$(OILCAST) -c $(CONFIG) download

build:  ## Dataset sobre el calendario del Brent -> data/processed
	$(OILCAST) -c $(CONFIG) build

tune:  ## Búsqueda de hiperparámetros (9 estudios, en serie)
	$(OILCAST) -c $(CONFIG) tune

evaluate:  ## Evaluación walk-forward en test
	$(OILCAST) -c $(CONFIG) evaluate

baselines:  ## Modelos sencillos y selección por horizonte, con protocolo separado
	$(OILCAST) -c configs/baselines.yaml baselines

returns:  ## Exporta los errores en rentabilidad simple, sin reentrenar
	$(OILCAST) -c $(CONFIG) returns

diagnostics:  ## Velas OHLC y violines de errores de los modelos guardados
	$(OILCAST) -c $(CONFIG) diagnostics

improve-intervals:  ## Calibración experimental elegida en desarrollo, sin cambiar las medianas
	$(OILCAST) -c configs/intervals.yaml improve-intervals

timing:  ## Tiempos de entrenamiento comparables
	$(OILCAST) -c $(CONFIG) timing

summarize:  ## Métricas, tabla comparativa y elección de modelo
	$(OILCAST) -c $(CONFIG) summarize

train-final:  ## Entrena el modelo de producción (models/)
	$(OILCAST) -c $(CONFIG) train-final

retrain:  ## Reentrena el modelo de producción con los hiperparámetros guardados
	$(OILCAST) -c $(CONFIG) train-final --from-metadata

report:  ## Genera reports/comparativa.md y reports/figures
	$(OILCAST) -c $(CONFIG) report

forecast:  ## Previsión actual en la terminal
	$(OILCAST) -c $(CONFIG) forecast

app:  ## Lanza el observatorio web local
	$(OILCAST) -c $(CONFIG) serve

static:  ## Versión estática de la web para GitHub Pages -> site/
	$(OILCAST) -c $(CONFIG) export-static --out site

test:  ## Tests (CPU, rápidos)
	poetry run pytest

lint:  ## Ruff (lint + formato)
	poetry run ruff check .
	poetry run ruff format --check .

format:  ## Formatea el código
	poetry run ruff format .
	poetry run ruff check --fix .

smoke:  ## Pipeline completo con configuración mínima (CPU, pocos minutos)
	OILCAST_DEVICE=cpu $(MAKE) tune evaluate timing summarize train-final report forecast \
		CONFIG=configs/smoke.yaml

notebooks:  ## Ejecuta los notebooks guardando las salidas
	poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
