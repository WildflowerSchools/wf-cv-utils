.PHONY: lint test

deps:
	@pip install --user virtualenv
	@python3 -m venv env
	source env/bin/activate; \
	pip install -e .

test-deps: deps
	source env/bin/activate; \
	pip install pytest pycodestyle pytest-cov

test:
	source env/bin/activate; \
	pytest --cov-report term-missing --cov=cv_utils -v tests/

lint:
	source env/bin/activate; \
	python -m pycodestyle . --ignore=E501,E252 --exclude=env,test --statistics --count
