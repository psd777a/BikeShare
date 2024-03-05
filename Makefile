install: |
	pip install --upgrade pip &&\
		pip install -r Package/requirements/requirements.txt

format: |
	black ./

lint: |
	pylint --disable=R,C ./Package/bikeshare_model/*.py &&\
	pylint --disable=R,C ./bikeshare_api/app/*.py || true

test: |
	python -m pytest Package/tests/test_*.py

build: |
	python3 bikeshare_model/train_pipeline.py &&\
	   		python3 -m build &&\
				cp dist/bikeshare_model-0.0.1-py3-none-any.whl  bikeshare_model_api

all: install format lint test
