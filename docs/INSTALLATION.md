# Installation

## Requirements

- **Python 3.11.** The pipeline was developed and tested on Linux. TensorFlow 2.20 also supports
  macOS and Windows (WSL2), but those platforms have not been tested with this code.
- **Disk**: about 11 GB of free space for the virtual environment (about 2.5 GB), the processed data
  (about 6.5 GB, mostly intermediate preprocessing files) and the outputs (about 1.5 GB).
- **CPU**: no GPU is needed. The full pipeline takes about 3 hours on a 16-core (32-thread) CPU;
  it runs on fewer cores, more slowly.
- **Network**: only for installing packages. The input data are in the repository.

## Virtual environment

From the repository root:

```bash
make setup                 # python3 -m venv .venv && pip install -r requirements.txt
source .venv/bin/activate
```

or by hand:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `python3` on your system is not 3.11, call `make setup PY=python3.11`.

`requirements.txt` pins every package the code imports to a tested version.
Other versions usually work, but they can change floating-point results and, with them, the trained
model and the numbers that depend on it.

## Check the environment

```bash
make verify-env            # compares installed versions with requirements.txt
make ci                    # environment check plus the test suite (about a minute)
```

`make verify-env` reports packages that are missing (an error) or installed at a different version
than pinned (a warning).

## Conda

A conda environment works too, as long as the pip packages in `requirements.txt` are installed in it:

```bash
conda create -n dietrec python=3.11
conda activate dietrec
pip install -r requirements.txt
```

## Next steps

Run `make reproduce` (see the [README](../README.md#running-the-pipeline)) and look at
[STRUCTURE.md](STRUCTURE.md) for where the outputs go.
