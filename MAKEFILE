install:
	pip install --upgrade pip &&\
		pip install -r Package/requirements/requirements.txt

format:
	black ./

lint:
	pylint --disable=R,C ./

test:
	python -m pytest Package/tests/test_*.py

all: install format lint test