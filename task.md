# Tasks

- [ ] Define BenchmarkMetricType, MetricResult, BenchmarkDefinition, BenchmarkResult, BenchmarkTrace, BenchmarkReport, BenchmarkProfile, and BenchmarkProfileRegistry in benchmark_models.py
- [ ] Implement BaseBenchmark, BaseBenchmarkDataset, and BaseMetricCalculator protocols in base.py
- [ ] Implement concrete metric calculators and estimators in metrics.py
- [ ] Implement VerificationBenchmarkRunner and datasets in runner.py / implementations.py
- [ ] Update exceptions.py with Benchmark exceptions
- [ ] Update config.py with Benchmark settings
- [ ] Update bootstrap.py to register benchmark profiles
- [ ] Create tests/unit/test_benchmark.py to fully verify benchmark implementation
- [ ] Run validation commands (ruff, isort, mypy, pytest)
- [ ] Update MILESTONE_STATUS.md and DECISION_LOG.md
- [ ] Stage, commit, push, verify CI
