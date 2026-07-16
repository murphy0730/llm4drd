# Graph Context Benchmark Report

Generated: `2026-07-16T12:53:51.510563+00:00`

## Environment

- Python: `3.14.3`
- Platform: `macOS-26.3-arm64-arm-64bit-Mach-O`
- Processor: `arm`
- CPU count: `10`
- Commit: `44755d657a821290f5557a252a06e18f760cada1`
- Sizes: `80,500,2500`
- Runs / warmups: `7 / 2`
- Seed: `42`

## Results

| Operations | Legacy init (ms) | Cold build + write (ms) | SQLite init (ms) | SQLite speedup | L1 init (ms) | L1 speedup | Runtime regression | Peak memory ratio | Signatures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 80 | 4.601 | 10.201 | 3.873 | 1.19x | 2.910 | 1.58x | -0.97% | 0.827x | match |
| 500 | 18.676 | 42.383 | 10.243 | 1.82x | 7.752 | 2.41x | -2.55% | 0.825x | match |
| 2500 | 91.220 | 210.877 | 42.027 | 2.17x | 31.141 | 2.93x | -2.35% | 0.824x | match |

## Acceptance

Overall: **FAIL**

The acceptance gates are at least 2.0x SQLite and L1 warm initialization speedup for all but the smallest requested fixture, no more than 3% end-to-end runtime regression, matching deterministic output signatures, and no more than 1.25x peak memory for every fixture.

Failures:

- 500: SQLite warm initialization speedup 1.82x < 2.00x

## Baseline Reference

- Generated: `2026-07-16T12:42:20.424278+00:00`
- Commit: `44755d657a821290f5557a252a06e18f760cada1`

Raw timing, memory, and signature samples are preserved in `graph-context-baseline.json` and `graph-context-result.json`.
