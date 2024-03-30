.PHONY: test install

test:
	poetry run pytest -vv \
		--cov=dataset_tools \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

install:
	poetry install