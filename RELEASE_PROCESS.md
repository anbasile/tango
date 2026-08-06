# Release Process

Releases are published to [GitHub Releases](https://github.com/anbasile/tango/releases) with the
wheel and sdist attached. This fork is not published to PyPI or conda-forge.

## Steps

1. Make sure the changes you want to release are described under "Unreleased" in `CHANGELOG.md`.

2. Update the version in `tango/version.py`.

3. Run the release script:

    ```bash
    ./scripts/release.sh
    ```

    This updates the CHANGELOG and `CITATION.cff`, commits, and pushes a `vX.Y.Z` tag. Pushing the
    tag triggers `.github/workflows/release.yml`, which:

    - rebuilds the sdist and wheel through the same reusable `build.yml` workflow that runs on
      every pull request,
    - checks the tag agrees with `tango/version.py`, and fails the release if it doesn't,
    - generates release notes from the matching `CHANGELOG.md` section,
    - creates the GitHub release with `dist/*` attached.

    Tags containing `rc`, `a` or `b` are published as pre-releases.

## Installing a release

```bash
pip install https://github.com/anbasile/tango/releases/download/v2.0.0/ai2_tango-2.0.0-py3-none-any.whl
```

## Fixing a failed release

Delete both the tag and the corresponding release from GitHub, then push a fix. Remove the tag from
your local clone with:

```bash
git tag -l | xargs git tag -d && git fetch -t
```

Then repeat the steps above.
