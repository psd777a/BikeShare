install: |
	pip install --upgrade pip &&\
		pip install -r Package/requirements/requirements.txt

format: |
	black ./

lint: |
	pylint --disable=R,C ./Package/bikeshare_model
	pylint --disable=R,C ./bikeshare_api/app

test: |
	python -m pytest Package/tests/test_*.py

all: install format lint test
