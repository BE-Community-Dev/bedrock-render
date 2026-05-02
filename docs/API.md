# API Guide

`bedrock-render` exposes four layers:

- `RenderPalette` for color lookup, overrides, and JSON imports.
- `MapRenderer` for tile, batch, region, bake, and web-map rendering.
- `MapRenderSession` for long-lived interactive renderers with cache reuse and
  streaming tile events.
- Value types for layout, threading, memory budgets, diagnostics, and cached
  tile paths.

## Renderer Setup

Create a lazy `bedrock_world::BedrockWorld`, then pass it with a palette to
`MapRenderer::new`:

```rust
use std::sync::Arc;

let renderer = bedrock_render::MapRenderer::new(
    Arc::new(world),
    bedrock_render::RenderPalette::default(),
);
```

`RenderPalette::default()` builds the palette from embedded auditable JSON
sources at startup. Applications that carry their own licensed color data can
merge additional JSON before constructing the renderer.

## Tile And Region Rendering

Use `RenderJob::new` for one tile and `MapRenderer::plan_region_tiles` for
deterministic web-map regions. `RenderLayout` controls world coverage and pixel
density. `RenderOptions` controls output format, backend, threading, memory
budget, CPU/GPU pipeline policy, priority, diagnostics, cancellation, and
surface shading.

`ImageFormat::Rgba` is the cheapest validation format. `ImageFormat::WebP` and
`ImageFormat::Png` populate `TileImage::encoded` when their features are enabled.

## Session Streaming

Use `MapRenderSession` for UI integrations. The session owns a `MapRenderer`,
shares a memory/disk tile cache, and can use render-index culling to avoid
loading chunks that have no renderable records.

```rust
let session = bedrock_render::MapRenderSession::new(
    renderer,
    bedrock_render::MapRenderSessionConfig {
        cache_root: "target/render-cache".into(),
        world_id: "world".into(),
        world_signature: "stable-content-signature".into(),
        cull_missing_chunks: true,
        ..Default::default()
    },
);
let cancel = bedrock_render::RenderCancelFlag::new();
session.render_web_tiles_streaming_blocking(
    &planned_tiles,
    bedrock_render::RenderOptions {
        cancel: Some(cancel),
        cache_policy: bedrock_render::RenderCachePolicy::Use,
        ..Default::default()
    },
    |event| {
        match event {
            bedrock_render::TileStreamEvent::Cached { .. } => {}
            bedrock_render::TileStreamEvent::Rendered { .. } => {}
            bedrock_render::TileStreamEvent::Failed { .. } => {}
            bedrock_render::TileStreamEvent::Progress(_) => {}
            bedrock_render::TileStreamEvent::Complete { .. } => {}
        }
        Ok(())
    },
)?;
```

`render_web_tiles_streaming` is available with the default `async` feature. It
wraps the same blocking pipeline in `tokio::task::spawn_blocking`. For UI code
that prefers an async channel, use `render_web_tiles_streaming_channel`; it
returns a `tokio::sync::mpsc::Receiver<TileStreamEvent>` immediately and lets
the render task continue in the background.

### Replacing older entry points

- `render_web_tiles_blocking` remains suitable for static exports where one
  sink writes all tiles.
- Interactive renderers should use `MapRenderSession` and stream events into
  the UI.
- Full-world chunk scans should move to `bedrock-world` render-index APIs before
  planning visible tiles.
- `render_web_tiles_blocking` remains ordered for export. For first-screen
  interactivity, set `RenderOptions::priority` to
  `RenderTilePriority::DistanceFrom { tile_x, tile_z }`.

## CPU Pipeline

`RenderOptions::cpu` controls the local Rayon pipeline used for load, bake,
compose, and encode coordination:

- `queue_depth=0` picks a bounded queue sized to the worker count and work set.
- `chunk_batch_size=0` lets `bedrock-world` choose render chunk batch sizes.
- `encode_workers=0` keeps encode/write workers profile-aware.

The renderer does not use Rayon global pool state. Long-lived sessions reuse the
same world, renderer, cache, and diagnostics path while each render request
keeps cancellation and progress scoped to its generation.

## Render Modes

- `SurfaceBlocks` is the primary terrain map.
- `HeightMap` renders height gradients from Bedrock height data.
- `Biome { y }` renders resolved biome colors at a sampled Y layer.
- `RawBiomeLayer { y }` renders diagnostic biome-id colors.
- `LayerBlocks { y }` renders a fixed X/Z block layer.
- `CaveSlice { y }` renders air, solid, water, and lava diagnostics for a fixed
  Y layer.

Missing chunks and empty terrain are transparent. Unknown block names are opaque
purple diagnostic pixels and are counted in `RenderDiagnostics`.

## Error Handling

All public fallible APIs return `bedrock_render::Result<T>`.

Match `BedrockRenderError::kind()` for stable categories:

```rust
match error.kind() {
    bedrock_render::BedrockRenderErrorKind::Cancelled => {
        // The caller's cancel flag was observed.
    }
    bedrock_render::BedrockRenderErrorKind::UnsupportedMode => {
        // The requested output format or feature is not enabled.
    }
    _ => eprintln!("{error}"),
}
```

Display strings are for humans and may become more descriptive over time.

## GPU Backend

The default feature set includes `gpu`. `RenderBackend::Auto` uses CPU by default
unless `BEDROCK_RENDER_GPU` requests GPU, or unless callers explicitly set
`RenderBackend::Gpu`. GPU failures are reported through
`RenderPipelineStats::gpu_fallback_reason` and counted as CPU fallback renders.

`RenderOptions::gpu` controls scheduling:

- `min_pixels` is the Auto-mode threshold before GPU compose is attempted.
- `max_in_flight=0` uses profile defaults: up to 2 interactive jobs or 4 export
  jobs.
- `batch_size=0`, `batch_pixels=0`, `submit_workers=0`, and
  `readback_workers=0` use profile-aware automatic defaults.
- `buffer_pool_bytes=0` and `staging_pool_bytes=0` enable automatic reusable
  storage/readback pool budgets. Explicit non-zero values cap each pool.

The GPU path uses a single `wgpu` device with a bounded in-flight queue. CPU
workers continue to load, bake, and prepare tiles while GPU jobs upload,
dispatch, and read back results. Stats expose `gpu_batches`,
`gpu_batch_tiles`, `gpu_max_in_flight`, `gpu_queue_wait_ms`,
`gpu_submit_workers`, `gpu_worker_threads`, `gpu_buffer_reuses`,
`gpu_buffer_allocations`, `gpu_staging_reuses`, and
`gpu_staging_allocations` for tuning.

`SurfaceRenderOptions::block_boundaries` controls the default top-down 2D block
outline and height-contact shadow. It is part of the render cache signature and
is disabled when `height_shading` is disabled.

Streaming complete events include `RenderPipelineStats`, so UI status bars can
show `resolved_backend`, `gpu_fallback_reason`, cache hits/misses, worker count,
and CPU/GPU timing without waiting for a separate export summary.

CPU-only consumers can disable default features and enable the formats they need.
