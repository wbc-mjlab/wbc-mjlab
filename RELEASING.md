# Releasing

## Pre-release checklist

1. Bump `version` in `pyproject.toml`.
2. Update `version` and `date-released` in `CITATION.cff` (when present).
3. Commit the version bump, then create an annotated tag:

```sh
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Build and verify

```sh
rm -rf dist/
make build
```

This runs `uv build` to produce a wheel and sdist in `dist/`, then smoke-tests
both artifacts in isolated environments.

## Test on TestPyPI (optional)

```sh
UV_PUBLISH_TOKEN=<your-testpypi-token> make publish-test
```

Verify with:

```sh
uvx --extra-index-url https://test.pypi.org/simple/ \
    --index-strategy unsafe-best-match \
    --from wbc-mjlab \
    wbc-mjlab-list-envs
```

## Publish to PyPI

```sh
UV_PUBLISH_TOKEN=<your-pypi-token> make publish
```

## Post-release

```sh
uvx --refresh --from wbc-mjlab wbc-mjlab-list-envs
```

## Releasing from a past tag

```sh
git checkout vX.Y.Z
make build
make publish
git checkout main
```
