# Contributing

This repository is the code companion to a journal article, so changes that alter its
results need a clear reason. Bug fixes, portability fixes and documentation
improvements are welcome.

## Before opening a pull request

```bash
make ci          # environment check and tests (what GitHub Actions runs)
```

If a change can affect results, run the affected stages (or `make reproduce`) before and after
the change and compare the outputs, starting with `outputs/final_results_log.txt`. The pipeline is
deterministic in a fixed environment, so a pure refactor should leave every output file
unchanged. Say in the pull request what you ran and what changed.

## Guidelines

- Keep pull requests focused; describe the problem and how you checked the fix.
- Match the style of the files you touch.
- Don't commit generated files (`outputs/`, `logs/`, `data/processed/`) or secrets.

## Reporting issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) and include the OS, Python version, the
command that failed and the relevant part of the log under `logs/`.

## Releases

1. Update [`CHANGELOG.md`](CHANGELOG.md), and `version` and `date-released` in
   [`CITATION.cff`](CITATION.cff).
2. Tag the release on `main` and publish a GitHub release from the tag, for example
   `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`, then
   `gh release create vX.Y.Z --notes-file <notes>`.

## Licence

Contributions are made under the terms in [`LICENSE`](LICENSE).
