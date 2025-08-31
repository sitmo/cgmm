.PHONY: patch minor major docs precommit

# --- Version bumps (update version, commit, create tag) ---
patch:
	poetry run bump-my-version bump patch
	git push && git push --tags

minor:
	poetry run bump-my-version bump minor
	git push && git push --tags

major:
	poetry run bump-my-version bump major
	git push && git push --tags

# --- Build docs locally ---
docs:
	rm -rf docs/_build .jupyter_cache
	poetry run sphinx-build -b html -E docs docs/_build/html -W --keep-going
	@echo "Open docs/_build/html/index.html in your browser"

# --- Run pre-commit checks on all files ---
precommit:
	poetry run pre-commit run --all-files
