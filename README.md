Python translation of Michael Porter's KRAKEN (not KRAKENC).

> An alternative implementation which provided an internal wave mode equation solver using the same numerical methods as Richard Evans `FORTRAN` model WAVE is found on an archived branch of the code called `rev1_archive`.

The code is provided "as is", with no guarantees of correctness. When in doubt, compare to KRAKEN and to KRAKENC.
Comparisons for a number of test environments are provided in `tests/`.

## Installation
This repository use [uv](https://docs.astral.sh/uv/guides/package/#updating-your-version) to manage its dependency

```sh
uv sync # install necessary dependency
source .venv/bin/activate # activate the virtual environment (you can also use `uv run $SHELL`)
```

### For development and testing

Install development and testing dependencies using
```sh
uv sync --all-groups
```

## Tests
Some basic tests are implemented to compare the results of the modal parameters $k_r$ and $\Psi$ to the original `KRAKEN` program.

``` sh
uv run pytest
```
