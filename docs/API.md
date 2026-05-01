# API Guide

`bedrock-render` exposes three layers:

- `RenderPalette` for color lookup, overrides, JSON imports, and binary palette
  cache generation.
- `MapRenderer` for tile, batch, region, bake, and web-map rendering.
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

`RenderPalette::default()` loads the embedded binary palette cache and biome
JSON at startup. `RenderPalette::from_builtin_json_sources()` rebuilds the same
palette from the auditable JSON sources without reading the binary cache, which
is useful for cache validation and palette maintenance tooling. Applications
that carry their own licensed color data can merge JSON or `BRPAL01` palette
caches before constructing the renderer.

## Tile And Region Rendering

Use `RenderJob::new` for one tile and `MapRenderer::plan_region_tiles` for
deterministic web-map regions. `RenderLayout` controls world coverage and pixel
density. `RenderOptions` controls output format, backend, threading, memory
budget, diagnostics, cancellation, and surface shading.

`ImageFormat::Rgba` is the cheapest validation format. `ImageFormat::WebP` and
`ImageFormat::Png` populate `TileImage::encoded` when their features are enabled.

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

CPU-only consumers can disable default features and enable the formats they need.
