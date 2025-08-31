.PHONY: patch minor major precommit docs
SHELL := /bin/bash

# Helper: run pre-commit once to apply fixes, then verify. Abort if verification fails.
define PRECOMMIT_ENFORCE
  echo "# pre-commit (apply fixes if any)"; \
  poetry run pre-commit run --all-files || true; \
  echo "# pre-commit (verify after fixes)"; \
  if ! poetry run pre-commit run --all-files; then \
    echo "pre-commit verification failed; aborting release." >&2; \
    exit 1; \
  fi
endef

# Core release recipe param: $(1) is patch|minor|major
define RELEASE
  set -euo pipefail; \
  OLD_VERSION=$$(poetry version -s); \
  $(PRECOMMIT_ENFORCE); \
  echo "# bump $(1) (allow dirty tree from pre-commit fixes)"; \
  poetry run bump-my-version bump $(1) --no-commit --no-tag --allow-dirty; \
  NEW_VERSION=$$(poetry version -s); \
  echo "# create a single release commit (skip hooks: already verified)"; \
  git add -A; \
  git commit -m "chore(release): v$$NEW_VERSION (from $$OLD_VERSION)" --no-verify; \
  git tag "v$$NEW_VERSION"; \
  git push; \
  git push --tags; \
  echo "Released v$$NEW_VERSION"
endef

patch:
	@bash -lc '$(RELEASE)'

minor:
	@bash -lc '$(RELEASE minor)'

major:
	@bash -lc '$(RELEASE major)'

precommit:
	poetry run pre-commit run --all-files

docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"
.PHONY: patch minor major precommit docs
SHELL := /bin/bash

# Helper: run pre-commit once to apply fixes, then verify. Abort if verification fails.
define PRECOMMIT_ENFORCE
  echo "# pre-commit (apply fixes if any)"; \
  poetry run pre-commit run --all-files || true; \
  echo "# pre-commit (verify after fixes)"; \
  if ! poetry run pre-commit run --all-files; then \
    echo "pre-commit verification failed; aborting release." >&2; \
    exit 1; \
  fi
endef

# Core release recipe param: $(1) is patch|minor|major
define RELEASE
  set -euo pipefail; \
  OLD_VERSION=$$(poetry version -s); \
  $(PRECOMMIT_ENFORCE); \
  echo "# bump $(1) (allow dirty tree from pre-commit fixes)"; \
  poetry run bump-my-version bump $(1) --no-commit --no-tag --allow-dirty; \
  NEW_VERSION=$$(poetry version -s); \
  echo "# create a single release commit (skip hooks: already verified)"; \
  git add -A; \
  git commit -m "chore(release): v$$NEW_VERSION (from $$OLD_VERSION)" --no-verify; \
  git tag "v$$NEW_VERSION"; \
  git push; \
  git push --tags; \
  echo "Released v$$NEW_VERSION"
endef

patch:
	@bash -lc '$(RELEASE)'

minor:
	@bash -lc '$(RELEASE minor)'

major:
	@bash -lc '$(RELEASE major)'

precommit:
	poetry run pre-commit run --all-files

docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"
