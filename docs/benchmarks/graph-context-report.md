# Graph Context Benchmark Report

Generated: `2026-07-16T10:52:06.060752+00:00`

## Environment

- Python: `3.14.3`
- Platform: `macOS-26.3-arm64-arm-64bit-Mach-O`
- Processor: `arm`
- CPU count: `10`
- Commit: `57e57e6227622245acae78e249c1f64167799000`
- Sizes: `80,500,2500`
- Runs / warmups: `7 / 2`
- Seed: `42`

## Results

| Operations | Legacy init (ms) | Cold build + write (ms) | SQLite init (ms) | SQLite speedup | L1 init (ms) | L1 speedup | Runtime regression | Peak memory ratio | Signatures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 80 | 5.979 | 11.008 | 3.643 | 1.64x | 2.649 | 2.26x | -2.63% | 0.825x | match |
| 500 | 19.463 | 43.508 | 9.427 | 2.06x | 6.713 | 2.90x | -3.46% | 0.824x | match |
| 2500 | 92.590 | 211.386 | 36.607 | 2.53x | 24.952 | 3.71x | -0.83% | 0.823x | match |

## Acceptance

Overall: **PASS**

The acceptance gates are at least 2.0x SQLite and L1 warm initialization speedup for all but the smallest requested fixture, no more than 3% end-to-end runtime regression, matching deterministic output signatures, and no more than 1.25x peak memory for every fixture.

## Baseline Reference

- Generated: `2026-07-16T10:38:49.915265+00:00`
- Commit: `57e57e6227622245acae78e249c1f64167799000`

Raw timing, memory, and signature samples are preserved in `graph-context-baseline.json` and `graph-context-result.json`.
