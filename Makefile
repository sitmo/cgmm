.PHONY: patch minor major precommit docs test knmi
SHELL := /bin/bash

# Run pre-commit once to apply fixes, then once to verify; abort on failure.
define PRECOMMIT_ENFORCE
  echo "# pre-commit (apply fixes if any)"; \
  poetry run pre-commit run --all-files || true; \
  echo "# pre-commit (verify after fixes)"; \
  if ! poetry run pre-commit run --all-files; then \
    echo "pre-commit verification failed; aborting release." >&2; \
    exit 1; \
  fi
endef

# Core release recipe; arg1 = patch|minor|major
define RELEASE
  set -euo pipefail; \
  OLD_VERSION=$$(poetry version -s); \
  $(PRECOMMIT_ENFORCE); \
  echo "# bump $(1) (allow dirty tree from pre-commit fixes)"; \
  poetry run bump-my-version bump $(1) --no-commit --no-tag --allow-dirty; \
  NEW_VERSION=$$(poetry version -s); \
  git add -A; \
  git commit -m "chore(release): v$$NEW_VERSION (from $$OLD_VERSION)" --no-verify; \
  git tag "v$$NEW_VERSION"; \
  git push; \
  git push --tags; \
  echo "Released v$$NEW_VERSION"
endef

patch:
	@bash -lc '$(call RELEASE,patch)'

minor:
	@bash -lc '$(call RELEASE,minor)'

major:
	@bash -lc '$(call RELEASE,major)'

precommit:
	poetry run pre-commit run --all-files

docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"

test:
	poetry run pytest --cov=cgmm --cov-report=term-missing

knmi:
	poetry run python tools/download_knmi_weather.py --start 2014-01-01 --end 2024-12-31 --outfile docs/examples/data/amsterdam_knmi_hourly.csv
