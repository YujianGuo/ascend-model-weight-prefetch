# Ascend Model Weight Prefetch

A reusable Codex skill and profiling analyzer for designing weight prefetch on Ascend inference workloads. It maps real model source and post-loader weight storage to legal communication windows, then checks device overlap, consumer impact, and service-level evidence separately.

## Contents

- `SKILL.md`: source-driven workflow for Dense, MoE, hybrid-attention, and quantized models.
- `scripts/analyze_prefetch.py`: standard-library-only analyzer for profiler CSV and Chrome trace JSON/JSON.GZ.
- `references/`: model-adaptation guidance, profiling data contract, and a manifest example.
- `tests/`: analyzer unit tests.
- `agents/openai.yaml`: Codex skill metadata.

The repository intentionally excludes customer/model-specific profiling captures, deployment paths, credentials, and historical experiment reports.

## Use as a Codex skill

Copy this repository to the `ascend-model-weight-prefetch` directory under your Codex skills directory, then invoke `$ascend-model-weight-prefetch` with the target model source, runtime configuration, and exact profiling directories.

## Run the analyzer

Python 3.10 or newer is required. The analyzer uses only the Python standard library.

```bash
python -m unittest discover -s tests -v

python scripts/analyze_prefetch.py \
  --baseline /profiles/base/ASCEND_PROFILER_OUTPUT \
  --candidate /profiles/candidate/ASCEND_PROFILER_OUTPUT \
  --manifest /work/model-manifest.json \
  --output /work/prefetch-analysis.json \
  --markdown /work/prefetch-analysis.md
```

The manifest example is a schema starter, not verified model evidence. Profile overlap alone does not establish consumer or service speedup; see `SKILL.md` and `references/profiling-contract.md` for interpretation boundaries.

## License

No license has been specified for this repository.

