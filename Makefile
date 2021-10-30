.PHONY: test upload

test:
	@echo "Running doctests and unittests: nose must be installed"
	nosetests -v --logging-level=INFO --with-doctest --where apertools


REPO?=pypi
upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/* -r $(REPO)
