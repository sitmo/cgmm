.PHONY: patch minor major docs precommit
SHELL := /bin/bash

define ensure_clean_with_precommit
	# Run pre-commit once; it may modify files and exit non-zero
	poetry run pre-commit run -a || true
	# If anything changed, commit the auto-fixes
	if ! git diff --quiet; then \
		git add -A; \
		git commit -m "chore(pre-commit): auto-fixes"; \
	fi
endef

patch:
	$(ensure_clean_with_precommit)
	poetry run bump-my-version bump patch
	git push && git push --tags

minor:
	$(ensure_clean_with_precommit)
	poetry run bump-my-version bump minor
	git push && git push --tags

major:
	$(ensure_clean_with_precommit)
	poetry run bump-my-version bump major
	git push && git push --tags

docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"

precommit:
	poetry run pre-commit run --all-files
