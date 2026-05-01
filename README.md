# bedrock-render

[English](README.md) | [简体中文](README.zh-CN.md)

`bedrock-render` is a tile renderer for Minecraft Bedrock worlds. It depends on
`bedrock-world` for Bedrock LevelDB, NBT, chunk, subchunk, height, and biome
queries. Rendering, palettes, tile planning, image encoding, cancellation,
threading, previews, and benchmarks live in this separate crate so the world
parser remains lightweight.

This repository is designed to be checked out independently. The current MSRV is
Rust 1.88, and the default feature set includes `async`, `webp`, and `gpu`.
CPU-only consumers can build with `--no-default-features` and opt into the image
formats they need.

## Design

- Rendering is always an X/Z plane. `Y` is only a sampling parameter.
- Tiles are controlled by `RenderLayout`. The default layout is `16x16` chunks
  per tile, one block per pixel, producing a `256x256` image. Larger
  `blocks_per_pixel` values keep the same world area but render lower-detail
  images for fast large-world previews.
- Arbitrary rectangular chunk regions are supported through `ChunkRegion`.
- Web-map paths are deterministic and cache-safe:
  `<world>/<signature>/r<renderer>-p<palette>/<dimension>/<mode>/<layout>/<tile_x>/<tile_z>.<ext>`.
- Tile batches use bounded worker threads. `Auto` resolves from the selected
  execution profile: `Export` uses host logical CPUs, `Interactive` caps work to
  a smaller foreground-safe pool. Explicit thread counts support `1..=512`.
- Web-map export uses a global chunk-bake queue, per-wave dynamic memory budget,
  parallel tile compose/encode, and a bounded MPMC writer queue so CPU workers
  are not serialized behind `fs::write`.
- `RenderMemoryBudget::Auto` uses a bounded cache budget for chunk bakes and
  export waves. `FixedBytes` and `Disabled` are available for offline tooling.
- Long operations support explicit cancellation and progress callbacks.

## Render Modes

- `RenderMode::Biome { y }`: biome color map sampled at the requested Y layer.
- `RenderMode::RawBiomeLayer { y }`: diagnostic biome-id color map.
- `RenderMode::LayerBlocks { y }`: fixed block layer map at world Y.
- `RenderMode::SurfaceBlocks`: main top-down terrain map. Each X/Z column starts
  from the Bedrock height map, scans down to the highest renderable block, applies
  biome tint, and blends transparent water over the solid block below.
- `RenderMode::HeightMap`: height gradient from Bedrock Data2D/Data3D records.
- `RenderMode::CaveSlice { y }`: fixed Y cave diagnostic map for air/solid/water/lava.

`SurfaceBlocks` follows the same core data flow used by BedrockMap's terrain
bake path: use `chunk.get_height(x,z)`, scan downward in that column until a
renderable block is found, then bake terrain/biome/height data. The default
terrain preview applies lightweight height-normal shading so slopes remain visible;
the height map remains a separate diagnostic image. Missing chunk or
missing height records are treated as absent terrain, not as gray map pixels.
Fixed Y rendering remains available through `LayerBlocks { y }`.
Unknown blocks are rendered as opaque purple diagnostic pixels. Missing chunks,
missing height maps, and empty terrain are transparent and counted separately in
`RenderDiagnostics`.

## API Sketch

```rust
use std::sync::Arc;
use bedrock_render::{
    ChunkRegion, ImageFormat, MapRenderer, RenderExecutionProfile, RenderLayout,
    RenderMemoryBudget, RenderMode, RenderOptions, RenderPalette, RenderThreadingOptions,
};

let renderer = MapRenderer::new(Arc::new(world), RenderPalette::default());
let region = ChunkRegion::new(dimension, -32, -32, 31, 31);
let layout = RenderLayout {
    chunks_per_tile: 16,
    blocks_per_pixel: 1,
    pixels_per_block: 1,
};

let tiles = renderer.render_region_tiles_blocking(
    region,
    RenderMode::HeightMap,
    layout,
    RenderOptions {
        format: ImageFormat::WebP,
        threading: RenderThreadingOptions::Auto,
        execution_profile: RenderExecutionProfile::Export,
        memory_budget: RenderMemoryBudget::Auto,
        ..RenderOptions::default()
    },
)?;
```

## Preview Tool

The preview example generates six atlas PNGs plus web-map tile folders:

```text
cargo run --example render_preview --features png
```

Optional arguments:

```text
render_preview <world_path> <output_dir> <center_tile_x> <center_tile_z> \
  <viewport_tiles> <layer_y> <cave_y> <chunks_per_tile>
```

Preview output layout:

```text
<output_dir>/
  biome-viewport.png
  raw-biome-viewport.png
  layer-y64-viewport.png
  surface-viewport.png
  heightmap-viewport.png
  cave-y32-viewport.png
  web-tiles/sample/signature/r2-p1/overworld/heightmap/16c-1bpp/21/12.png
```

## Web Map Export

`render_web_map` exports WebP tiles plus a self-contained static HTML viewer.
It is intended for validating the renderer and for generating shareable web-map
artifacts without requiring GPUI or a CDN.
When `--region` is omitted, the example discovers and renders the loaded chunk
bounds for each selected dimension.

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map \
  --dimensions overworld,nether,end \
  --mode surface,heightmap,biome,layer \
  --chunks-per-tile 16 \
  --chunks-per-region 32 \
  --blocks-per-pixel 4 \
  --threads auto \
  --profile export \
  --memory-budget auto \
  --pipeline-depth 256 \
  --tile-batch-size auto \
  --writer-threads 2 \
  --write-queue-capacity 256 \
  --stats \
  --force
```

Output layout:

```text
target/bedrock-web-map/
  viewer.html
  map-layout.json
  map-data.js
  tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp
```

`map-layout.json` is the automatically generated map layout for tools and
external frontends. `map-data.js` embeds only the minimal layout constants and
dynamically provides `tileBounds()`, `tileId()`, and `tilePath()`. Tile locations
are derived from the fixed
`tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp` rule, so the HTML
viewer can load the data as a normal script from `file://` without using
`fetch()` or hitting browser CORS restrictions. Export writes `viewer.html`,
`map-layout.json`, and `map-data.js` before opening and scanning the world; as
dimension bounds are discovered and each mode finishes, `map-layout.json` and
`map-data.js` are refreshed with updated layout and stats. The viewer
periodically reloads `map-data.js` and retries visible tiles, so images appear
incrementally while export is still running.
It supports dimension switching, mode switching, drag panning, wheel zoom, tile
coordinates, and load-error reporting. For large
worlds, use `--region` for a bounded export or increase `--blocks-per-pixel`
before exporting the whole loaded chunk bounds. `--profile export --threads auto`
is the recommended default for 8-core/16-thread machines. Use
`--profile interactive` for UI-like previews, and use a fixed thread value up to
512 only for offline exports where the storage device can keep up. `--stats`
prints planned tiles, unique chunks, baked chunks, bake/encode/write time, cache
hits, and peak cache bytes so CPU under-utilization can be traced to I/O,
encoding, or chunk baking. Missing chunks, missing
subchunks, and unloaded fixed-Y areas are transparent. Opaque purple pixels
indicate an actual block name that was read from the world but had no palette
mapping.

## External Palettes

The default palette is embedded in the crate and in compiled binaries. It ships
with two auditable JSON sources plus a prebuilt `BRPAL01` binary cache:

```text
data/colors/bedrock-block-color.json
data/colors/bedrock-biome-color.json
data/colors/bedrock-colors.brpal
```

`RenderPalette::default()` loads the embedded binary cache first, so normal
rendering does not parse JSON or require external palette files. The JSON files
remain available through `RenderPalette::builtin_block_color_json()` and
`RenderPalette::builtin_biome_color_json()` for auditing and rebuild tooling.
`RenderPalette::from_builtin_json_sources()` rebuilds the same palette from the
auditable JSON sources without reading the binary cache.
The built-in data is maintained inside this project for Bedrock rendering; the
loader also understands the public bedrock-level color JSON shape so projects
can regenerate or compare palettes when their licensing policy allows it.

For projects that have their own licensed color data, `RenderPalette` also
supports additional user-provided JSON and a compact binary cache format:

```text
--palette-json target/bedrock-block-color.json
--palette-json target/bedrock-biome-color.json
--palette-cache target/colors.brpal
--rebuild-palette-cache
```

JSON is an import/override format. The embedded `BRPAL01` binary cache is the
recommended hot path for applications because it is a sequential,
allocation-light table of biome ids and block-name RGBA entries. The JSON loader
accepts combined objects
with `schema_version` / `sources` / `blocks` / `defaults` / `biomes`, object maps such as
`{"minecraft:stone":"#7d7d7d"}`, arrays with `name` / `id` plus `color`, and
the bedrock-level color JSON shapes for local user-provided reference data.

Palette source maintenance commands:

```text
cargo run --example palette_tool -- audit --check
cargo run --example palette_tool -- normalize --check
cargo run --example palette_tool -- rebuild-cache --check
```

Source policy and public references are documented in
[docs/PALETTE_SOURCES.md](docs/PALETTE_SOURCES.md).

Latest local reference-palette smoke run:

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map-ref-palette \
  --region 0,0,15,15 \
  --mode surface,heightmap,biome,layer \
  --y 64 \
  --palette-json target/bedrock-block-color.json \
  --palette-json target/bedrock-biome-color.json \
  --palette-cache target/bedrock-colors.brpal \
  --rebuild-palette-cache \
  --force

loaded palette JSON target\bedrock-block-color.json: block_colors=1207 biome_colors=0 skipped=0
loaded palette JSON target\bedrock-biome-color.json: block_colors=0 biome_colors=88 skipped=0
wrote palette cache: target\bedrock-colors.brpal
overworld surface tiles=1 missing=0 transparent=0 unknown=0
overworld layer-y64 tiles=1 missing=0 transparent=12544 unknown=0
```

Latest local debug preview run:

```text
cargo run --example render_preview --features png
Generated 6 atlas images and 54 web-map PNG tiles.
Viewport: 3x3 tiles, 16 chunks per tile, 256x256 px per tile.
surface diagnostics: missing_chunks=0 missing_heightmaps=0 unknown_blocks=0 fallback_pixels=0
```

Latest local WebP web-map smoke run:

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map \
  --region 0,0,15,15 \
  --mode surface,heightmap,biome,layer \
  --threads auto \
  --tile-batch-size auto \
  --writer-threads 2 \
  --write-queue-capacity 64 \
  --force

Generated viewer.html, map-layout.json, map-data.js, and WebP tiles for overworld/nether/end.
LayerBlocks transparent pixels are expected for unloaded fixed-Y areas.
```

## Rendered Examples

These images are generated by the preview tool from the bundled sample fixture
and the current default palette. The terrain view follows the BedrockMap-style
top-down bake flow while remaining part of this standalone public renderer.

### `Biome { y }`

Biome colors sampled on an X/Z plane at the configured Y layer.

![Biome map](docs/images/biome-viewport.png)

### `RawBiomeLayer { y }`

Diagnostic biome-id map. Unknown or sparse ids are intentionally visible.

![Raw biome map](docs/images/raw-biome-viewport.png)

### `LayerBlocks { y }`

Fixed world-Y block layer rendered as an X/Z plane. This is not a side section.

![Fixed Y layer map](docs/images/layer-y64-viewport.png)

### `SurfaceBlocks`

Primary top-down terrain map. Each X/Z column uses the chunk height map, scans
down to the highest renderable block, applies biome tint, and blends transparent
water with the block below. Lightweight height-normal shading is enabled by default
so terrain does not look completely flat; use `HeightMap` for elevation analysis.

![Surface block map](docs/images/surface-viewport.png)

### `HeightMap`

Height gradient derived from Bedrock Data2D/Data3D height records.

![Height map](docs/images/heightmap-viewport.png)

### `CaveSlice { y }`

Fixed Y cave diagnostic map for air, solid blocks, water, and lava.

![Cave slice map](docs/images/cave-y32-viewport.png)

## Performance Model

- `Biome`, `RawBiomeLayer`, `LayerBlocks`, `HeightMap`, and `CaveSlice` prefetch
  only the chunk records needed by the tile.
- Height data is fetched once per chunk through `bedrock-world` and cached as a
  compact `16x16` height array inside the tile context.
- `SurfaceBlocks` does not scan missing-height columns from world top to bottom.
  Missing chunk/height data is counted in diagnostics and emitted as absent
  terrain.
- Subchunk access uses `SubChunk::block_state_at(local_x, local_y, local_z)` so
  renderer code does not duplicate palette index math.
- Web export uses a region-first bake pipeline. `--chunks-per-region` controls
  the bake cache unit; `32` is the default for large exports, while `16` is
  better for small bounded validation runs or lower interactive latency.
- Web export does not keep every baked region for the full map resident at once.
  The renderer splits baking, composition, and writing into waves bounded by
  `--memory-budget`. `--pipeline-depth` and `--write-queue-capacity` only bound
  encoded tiles waiting for disk writes.
- `SurfaceBlocks` uses chunk bake: each chunk is reduced to a `16x16` terrain
  image first, then region planes and WebP tiles are assembled from baked chunk
  pixels.
- The static viewer uses `RenderLayout` auto-scaling. Small worlds default to
  full detail; larger worlds can use 2/4/8 blocks per pixel without changing the
  visible tile coverage.
- Cache keys include world path hash, world file signature, renderer version,
  palette version, dimension, render mode, Y layer, and layout. All-transparent
  stale cached tiles are rejected by the UI and regenerated.
- `ImageFormat::Rgba` is the lowest-latency UI path. WebP/PNG are intended for
  cache/export/preview paths.
- The static web-map example writes WebP tiles under
  `tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp`; it does not leave
  gaps between tiles and uses transparent pixels for absent world data.
- Viewer integrations can start from the world's `level.dat` spawn point,
  support dimension/custom-dimension switching, and store WebP web-map tiles
  under an application cache directory.

## Tests and Benchmarks

```text
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps --all-features
cargo rustdoc --lib --all-features -- -D missing_docs
cargo bench --bench render --all-features
```

The default benchmark suite measures tile rendering, chunk baking, and small
batch behavior. Full web-map export benchmarks are opt-in:

```text
$env:BEDROCK_RENDER_FULL_BENCH='1'
cargo bench --bench render --all-features
Remove-Item Env:\BEDROCK_RENDER_FULL_BENCH
```

More details are in [docs/API.md](docs/API.md), [docs/TESTING.md](docs/TESTING.md),
and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Current Limits

- Surface rendering is top-down and BedrockMap-style. Web-map export now uses a
  global chunk-bake queue, but cross-process persistent chunk-bake reuse is still
  a future optimization.
- V1 does not implement lighting, shadows, elevation blending, entity markers,
  or labels.
