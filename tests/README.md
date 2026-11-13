# Tests

```bash
make ci        # = python scripts/check_env.py && python -m unittest discover -v tests
```

The tests (`test_smoke.py`) check the environment, the configuration files and that every
pipeline entry point imports and parses its arguments. They need no pipeline outputs, so they run in CI.

`NumbersLogTest` also rebuilds the headline-numbers log from an existing run, and is skipped
when there is none. Set `RUN_NAME` to test a run other than `outputs/runs/eval_run`.
