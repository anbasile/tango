#!/bin/bash

set -e

# Read version.py directly rather than importing `tango`, so this works without the package's
# dependencies installed.
TAG="v$(python -c "import runpy; print(runpy.run_path('tango/version.py')['VERSION'])")"

read -p "Creating new release for $TAG. Do you want to continue? [Y/n] " prompt

if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]; then
    python scripts/prepare_changelog.py
    python scripts/prepare_citation_cff.py
    git add -A
    git commit -m "Prepare for release $TAG" || true && git push origin HEAD
    echo "Creating new git tag $TAG"
    git tag "$TAG" -m "$TAG"
    # Push only this tag. `git push --tags` would push every local tag, and this fork's clone
    # still carries allenai/tango's 46 of them — each would fire the release workflow.
    git push origin "$TAG"
else
    echo "Cancelled"
    exit 1
fi
