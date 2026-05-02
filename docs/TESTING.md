# Testing And Benchmarks

## Required Checks

```powershell
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps --all-features
cargo rustdoc --lib --all-features -- -D missing_docs
cargo run --example palette_tool -- audit --check
cargo run --example palette_tool -- generate-clean-room --check
cargo run --example palette_tool -- normalize --check
```

These checks are expected to pass on a fresh checkout without private world data.

## Optional Fixture

The integration test and benchmarks look for the sample Bedrock world in the
adjacent `bedrock-world` checkout:

```text
../bedrock-world/tests/fixtures/sample-bedrock-world
```

The folder should contain `level.dat` and `db/CURRENT`. If it is missing,
fixture tests and benches skip their world-backed cases instead of failing.

Do not commit real worlds. Add small in-memory tests for regressions whenever
possible.

## Feature Coverage

Run `cargo test --no-default-features` to validate CPU-only compilation. Run
`cargo test --all-features` for async, WebP, PNG, and GPU code paths.

GPU execution depends on host hardware and drivers. Tests should assert CPU
fallback behavior and diagnostics rather than requiring a GPU device.

For GPU changes, include a small direct compose test that accepts either a valid
GPU result or a non-empty fallback reason. When a GPU is available, check
`gpu_tiles`, `gpu_batches`, `gpu_batch_tiles`, `gpu_max_in_flight`,
`gpu_submit_workers`, buffer/staging reuse stats, and readback timing in the
streaming/export stats.

For surface rendering changes, cover both `SurfaceRenderOptions::block_boundaries`
enabled and disabled. Tests should include flat terrain, a sharp height step,
small height noise below the threshold, shallow water, and multi-block pixels
where per-block outlines are intentionally skipped.

## Streaming Session Checks

For changes touching `MapRenderSession` or frontend integration, cover:

- cancellation before and during a tile batch
- cached events on a second render with the same cache signature
- rendered and failed events for mixed valid/missing regions
- complete events containing diagnostics and pipeline stats
- CPU fallback when GPU compose is unavailable or rejected
- GPU queue cancellation before tile readback completes
- `RenderTilePriority::DistanceFrom` emits nearer cached/rendered tiles before
  farther tiles, independent of the original planned-tile vector order
- `render_web_tiles_streaming_channel` closes cleanly when the receiver is
  dropped and logs task-level failures

Manual UI testing should open a large world, pan before metadata indexing
finishes, zoom in/out, switch dimension/mode, toggle cache bypass, and verify
that stale generation events are ignored.
