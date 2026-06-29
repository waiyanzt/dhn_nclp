# Benchmark Results

Results are grouped by accelerator, then dataset and task:

```text
results/
  v100/
    imdb_nc_baseline/
    imdb_lp_baseline/
    dblp_lp_baseline/
    diagnostics/
      fb15k_lp_tail_only/
  h100/
    wordnet_lp_baseline/
```

## Baseline Status

- `v100/imdb_nc_baseline`: IMDb node classification, variants IMDb1-IMDb4.
- `v100/imdb_lp_baseline`: IMDb link prediction for movie-director,
  movie-genre, and movie-link tasks.
- `v100/dblp_lp_baseline`: DBLP paper-venue link prediction. Variants v1 and
  v2 are present; v3 is added by the current rerun.
- `h100/wordnet_lp_baseline`: WordNet link prediction for `no_changes`,
  `all_inverse_edges`, and `transitive_edges`.

DBLP node classification is not currently implemented in the clean benchmark
pipeline, so no `dblp_nc_baseline` directory exists.

## Diagnostics

`v100/diagnostics/fb15k_lp_tail_only` is an obsolete diagnostic run produced
before balanced head/tail negative sampling was implemented. Do not report it
as a final FB15k baseline.

