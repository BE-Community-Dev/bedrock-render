# Benchmarks

`benches/render.rs` uses Criterion against the optional sample world from the
adjacent `bedrock-world` checkout. If the fixture is missing, the benchmark
returns without measuring world-backed cases.

## Quick Suite

Run the default suite with:

```powershell
cargo bench --bench render --all-features
```

The quick suite measures representative tile renders, chunk baking, and small
batch behavior. It avoids repeating full web-map exports by default.

For interactive work, compare both one-shot batch rendering and session reuse.
The session path should avoid repeated world open/cache construction and should
produce the first `Cached` or `Rendered` event before the full batch completes.
When collecting local numbers, record:

- planned tiles and visible render chunks
- cache hits and misses
- worker count and CPU utilization
- `resolved_backend` and `gpu_fallback_reason`
- `gpu_batches`, `gpu_max_in_flight`, `gpu_queue_wait_ms`, and
  `gpu_worker_threads`
- `cpu_queue_wait_ms`, `gpu_batch_tiles`, `gpu_submit_workers`,
  `gpu_buffer_reuses`, `gpu_buffer_allocations`, `gpu_staging_reuses`, and
  `gpu_staging_allocations`
- time to first tile and time to complete

## Full Export Suite

Full web-map export benchmarks are intentionally opt-in because each sample can
take multiple seconds and can be dominated by disk throughput:

```powershell
$env:BEDROCK_RENDER_FULL_BENCH='1'
cargo bench --bench render --all-features
Remove-Item Env:\BEDROCK_RENDER_FULL_BENCH
```

## Latest Local Baseline

Measured on 2026-05-01 in release Criterion mode with the local sample fixture:

| Benchmark | Typical time |
| --- | ---: |
| `biome_tile_256_rgba` | 22.8 ms |
| `biome_tile_256_webp` | 23.0 ms |
| `fixed_y_tile_256_rgba` | 29.9 ms |
| `raw_biome_tile_256_rgba` | 22.8 ms |
| `surface_tile_256_rgba` | 208.1 ms |
| `bake_chunk_surface` | 6.5 ms |
| `render_tile_surface_from_bake` | 212.3 ms |
| `heightmap_tile_256_rgba` | 38.9 ms |
| `cave_slice_tile_256_rgba` | 31.6 ms |
| `tile_batch_auto_threads` | 210.7 ms |
| `tile_batch_single_thread` | 716.5 ms |

This baseline predates the block-boundary shadow and bounded GPU in-flight
compose queue. New baselines should record whether `block_boundaries` is enabled
and which `RenderCpuPipelineOptions`, `RenderTilePriority`, and
`RenderGpuOptions` values were used.

Full export benchmarks from this audit were around 5 seconds per sample for the
sample web region and are kept outside the default suite.

After the session upgrade, regressions should be triaged by pipeline stage:
render-index scan, chunk load, bake, compose, encode, and cache write. GPU
numbers are only comparable when the same adapter and driver are used; always
include the fallback reason when GPU work falls back to CPU. If
`gpu_queue_wait_ms` dominates, lower `max_in_flight` or increase CPU bake
parallelism only after checking GPU utilization. If buffer allocation counters
rise on every tile, raise `buffer_pool_bytes` / `staging_pool_bytes` or lower
tile concurrency so pool leases can be reused.
