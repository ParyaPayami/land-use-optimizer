# Contributing

1. Fork, create a branch, and install: `pip install -e ".[dev]"` (CPU PyTorch:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` first).
2. Keep `ruff check pimaluos tests` and `pytest` green; add a test for every
   behaviour you change.
3. Never commit data, model checkpoints or run outputs. Results belong in a
   Zenodo deposit and are referenced by DOI.
4. Numbers in the manuscript come only from `pimaluos report`; do not type
   results into `paper/` by hand.

## Adding a city

Create `pimaluos/config/cities/<city>.yaml` with `crs` (projected, feet),
`column_mapping` from the source parcel layer to the standard columns used in
`pimaluos/core/data_loader.py` (`lot_area_sqft`, `built_far`, `max_resid_far`,
`max_comm_far`, `max_facil_far`, `land_use`, `zone_district`, ...), and a
land-use code scheme compatible with `LAND_USE_CLASS`, then run
`pimaluos run --config <your config> --pluto <parcel file>`. Capacity-screen
parameters (`CapacityParams`) should be re-calibrated for the new city.

## Reporting issues

Open a GitHub issue with the command you ran, the `manifest.json` of the run,
and the full traceback.
