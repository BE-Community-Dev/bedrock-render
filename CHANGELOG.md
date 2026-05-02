# Changelog

All notable changes to `bedrock-render` are tracked here.

## 0.2.0 - Unreleased

### Added

- Added `MapRenderSession` and `MapRenderSessionConfig` for long-lived
  interactive renderers that reuse the world handle, renderer, tile cache, and
  diagnostics state.
- Added cancellable streaming tile rendering through
  `render_web_tiles_streaming_blocking` and the async
  `render_web_tiles_streaming` wrapper.
- Added `TileStreamEvent` with cached, rendered, failed, progress, and complete
  events so frontends can paint visible tiles before a full batch finishes.
- Added render-index culling for missing chunks before baking, backed by
  `bedrock-world` key-only region scans.
- Added `log` facade diagnostics around session streaming, cache hits/misses,
  GPU backend selection, and CPU fallback.
- Added `BlockBoundaryRenderOptions` for default 2D per-block surface outlines
  and height-contact shadows in `SurfaceBlocks`.
- Added `RenderGpuOptions` and GPU scheduling stats for bounded in-flight GPU
  compose, queue wait timing, batch counts, and readback worker visibility.
- Added `RenderCpuPipelineOptions` and `RenderTilePriority` on `RenderOptions`
  for bounded Rayon load/bake/compose scheduling and viewport-center-first tile
  ordering.
- Added `render_web_tiles_streaming_channel`, an async channel API returning
  `tokio::sync::mpsc::Receiver<TileStreamEvent>`.
- Extended GPU scheduling with `batch_pixels`, `submit_workers`,
  `buffer_pool_bytes`, and `staging_pool_bytes`, plus buffer/staging pool reuse
  stats.

### Breaking Changes

- Interactive integrations should migrate from one-shot
  `render_web_tiles_blocking` calls to a reusable `MapRenderSession`. The old
  blocking batch API remains for export tools, but it no longer represents the
  recommended UI path.
- `SurfaceRenderOptions` now includes `block_boundaries`, and `RenderOptions`
  now includes `gpu`, `cpu`, and `priority`. Struct literals must add these
  fields or use `..Default::default()`.
- Renderer and GPU shader cache versions were bumped because default
  `SurfaceBlocks` output now includes block-boundary shadowing.

### Migration Notes

- Replace per-batch world/cache construction with one session per opened world.
- Stream `TileStreamEvent` values into the UI and discard stale events by
  generation when the world, dimension, mode, layout, or cache policy changes.
- Use `RenderCancelFlag` on every viewport batch and cancel the previous flag
  before scheduling a new visible-tile request.
- Use `RenderOptions::gpu` to tune GPU compose. The default zero values select
  profile-aware in-flight, batch, readback, submit-worker, and buffer-pool
  settings.
- Use `RenderOptions::priority = RenderTilePriority::DistanceFrom { .. }` for
  interactive first-screen rendering; keep `RowMajor` for deterministic export
  order.

## 0.1.0 - 2026-05-01

### Added

- Initial public crate-ready tile renderer for Minecraft Bedrock worlds.
- Surface, height map, biome, raw biome, fixed layer, and cave slice render modes.
- Embedded Bedrock block and biome color palette JSON sources, including grass,
  foliage, and water tint tables.
- Palette source schema metadata, source-policy documentation, JSON-only palette
  rebuild support, and a `palette_tool` example for audit/normalize checks.
- Clean-room block and biome palette generation, tainted-source audit checks, and
  optional local resource-pack palette derivation under `target/`.
- Semantic palette guardrails for bamboo, dirt/grass paths, tint masks, biome
  grass separation, and real surface-render biome tint output.
- Region planning, deterministic tile paths, render diagnostics, cancellation,
  progress callbacks, bounded threading, memory-budgeted region baking, and
  optional GPU terrain-light compose.
- PNG/WebP/RGBA output support, preview and static web-map examples, fixture tests,
  Criterion benches, and English/Simplified Chinese documentation.

### Notes

- The crate depends on `bedrock-world` by pinned Git revision until the Bedrock
  crate family is ready for crates.io publishing.
- Real Bedrock worlds and generated render outputs are intentionally excluded
  from version control.
