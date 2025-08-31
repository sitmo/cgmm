.PHONY: patch minor major docs precommit
SHELL := /bin/bash

define do_bump
	# Capture old/new versions from pyproject via Poetry
	OLD_VERSION=$$(poetry version -s); \
	poetry run bump-my-version bump $(1) --commit false --tag false; \
	NEW_VERSION=$$(poetry version -s); \
	# Run pre-commit; it may modify files
	poetry run pre-commit run -a || true; \
	# Stage any auto-fixes and commit/tag
	git add -A; \
	git commit -m "Bump version: $$OLD_VERSION → $$NEW_VERSION"; \
	git tag v$$NEW_VERSION; \
	git push && git push --tags
endef

patch:
	$(call do_bump,patch)

minor:
	$(call do_bump,minor)

major:
	$(call do_bump,major)

docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"

precommit:
	poetry run pre-commit run --all-files
