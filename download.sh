#!/bin/bash

mkdir icu_data && \
gsutil -m cp \
  "gs://a_star_datasets/icu_mortality_data-000000000000.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000001.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000002.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000003.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000004.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000005.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000006.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000007.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000008.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000009.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000010.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000011.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000012.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000013.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000014.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000015.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000016.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000017.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000018.csv" \
  "gs://a_star_datasets/icu_mortality_data-000000000019.csv" \
  icu_data/ && \
(cd icu_data && zip -r ../icu_mortality_data.zip . -x ".*" -x "__MACOSX") && \
rm -rf icu_data

