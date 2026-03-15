# Step 1: Data Preprocessing

This folder contains data collection and preparation notebooks used before training.

## Notebooks

- `get_airbnb_data_al.ipynb`
- `get_airbnb_data_am.ipynb`
- `get_airbnb_data_mo.ipynb`
  - Collect city-specific Airbnb data and save processed CSVs.
- `get_amzn_data.ipynb`
  - Prepares Amazon-side data used in related experiments.
- `get_date.ipynb`
  - Adds date fields to mismatch outputs for analysis.

## How to run

These are notebook workflows (not CLI scripts). Run with Jupyter:

```bash
jupyter lab
```

Open the notebook and run all cells in order.

## Typical outputs

- Cleaned listings/reviews CSVs
- City-specific intermediate files used by later steps (`Space_Formation_Embedding_Extraction` and `Mismatch_Score`)
