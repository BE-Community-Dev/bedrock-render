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
| `biome_tile_256_rgba` | 27.7 ms |
| `biome_tile_256_webp` | 28.1 ms |
| `fixed_y_tile_256_rgba` | 34.8 ms |
| `raw_biome_tile_256_rgba` | 27.7 ms |
| `surface_tile_256_rgba` | 243.2 ms |
| `bake_chunk_surface` | 8.9 ms |
| `render_tile_surface_from_bake` | 244.3 ms |
| `heightmap_tile_256_rgba` | 48.4 ms |
| `cave_slice_tile_256_rgba` | 34.9 ms |
| `tile_batch_auto_threads` | 279.1 ms |
| `tile_batch_single_thread` | 943.5 ms |

Full export benchmarks from this audit were around 5 seconds per sample for the
sample web region and are kept outside the default suite.
