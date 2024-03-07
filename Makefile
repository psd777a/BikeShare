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
	python3 Package/bikeshare_model/train_pipeline.py &&\
		cd Package &&\
	   		python3 -m build &&\
				cd .. &&\
				cp Package/dist/bikeshare_model-0.0.1-py3-none-any.whl  bikeshare_api

all: install format lint test
