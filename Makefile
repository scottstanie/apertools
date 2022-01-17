.PHONY: test upload

test:
	@echo "Running doctests and unittests: pytest must be installed"
	# nosetests -v --logging-level=INFO --with-doctest --where apertools
	python -m pytest -v  --doctest-modules --ignore=helpers/ --ignore apertools/scripts/


REPO?=pypi
upload:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/* -r $(REPO)
