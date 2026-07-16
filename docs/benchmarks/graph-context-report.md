# Graph Context Benchmark Report

Generated: `2026-07-16T11:44:12.456825+00:00`

## Environment

- Python: `3.14.3`
- Platform: `macOS-26.3-arm64-arm-64bit-Mach-O`
- Processor: `arm`
- CPU count: `10`
- Commit: `7804fa49217255ad2abd85814571593568c7cfdd`
- Sizes: `80,500,2500`
- Runs / warmups: `7 / 2`
- Seed: `42`

## Results

| Operations | Legacy init (ms) | Cold build + write (ms) | SQLite init (ms) | SQLite speedup | L1 init (ms) | L1 speedup | Runtime regression | Peak memory ratio | Signatures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 80 | 4.789 | 10.528 | 4.032 | 1.19x | 2.982 | 1.61x | -1.11% | 0.827x | match |
| 500 | 18.836 | 43.112 | 10.426 | 1.81x | 7.774 | 2.42x | -3.63% | 0.825x | match |
| 2500 | 95.079 | 214.198 | 42.250 | 2.25x | 31.582 | 3.01x | -3.42% | 0.824x | match |

## Acceptance

Overall: **FAIL**

The acceptance gates are at least 2.0x SQLite and L1 warm initialization speedup for all but the smallest requested fixture, no more than 3% end-to-end runtime regression, matching deterministic output signatures, and no more than 1.25x peak memory for every fixture.

Failures:

- 500: SQLite warm initialization speedup 1.81x < 2.00x

## Baseline Reference

- Generated: `2026-07-16T11:40:33.200170+00:00`
- Commit: `7804fa49217255ad2abd85814571593568c7cfdd`

Raw timing, memory, and signature samples are preserved in `graph-context-baseline.json` and `graph-context-result.json`.
