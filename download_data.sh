#!/bin/bash

# Create target directory
mkdir -p icu_timeseries_1

# Download all parquet files into the folder
gsutil -m cp gs://a_star_datasets/timeseries_mimic_icu_1/icu_timeseries_*.parquet icu_timeseries_1/

# Create a zip archive
zip icu_timeseries_1.zip icu_timeseries_1/*.parquet

