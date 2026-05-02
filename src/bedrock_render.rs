//! Tile rendering utilities for Minecraft Bedrock worlds.
//!
//! `bedrock-render` builds on `bedrock-world` and provides palette management,
//! tile planning, top-down render modes, image encoding, cancellation,
//! diagnostics, and Criterion-backed benchmark support for map tooling.

mod error;
mod palette;
mod renderer;

pub use error::{BedrockRenderError, BedrockRenderErrorKind, Result};
pub use palette::{PaletteImportReport, RenderPalette, RgbaColor};
pub use renderer::{
    BakeDiagnostics, BakeOptions, BlockBoundaryRenderOptions, ChunkBake, ChunkBakePayload,
    ChunkRegion, ChunkTileLayout, DEFAULT_PALETTE_VERSION, DepthPlane, GPU_COMPOSE_SHADER_VERSION,
    HeightPlane, ImageFormat, MAX_RENDER_THREADS, MAX_TILE_SIZE_PIXELS, MapRenderSession,
    MapRenderSessionConfig, MapRenderer, PlannedTile, RENDERER_CACHE_VERSION, RegionBake,
    RegionBakePayload, RegionCoord, RegionLayout, RenderBackend, RenderCachePolicy,
    RenderCancelFlag, RenderCpuPipelineOptions, RenderDiagnostics, RenderDiagnosticsSink,
    RenderExecutionProfile, RenderGpuOptions, RenderJob, RenderLayout, RenderMemoryBudget,
    RenderMode, RenderOptions, RenderPipelineStats, RenderProgress, RenderProgressSink,
    RenderThreadingOptions, RenderTilePriority, RenderWebTilesResult, ResolvedRenderBackend,
    RgbaPlane, SurfacePlane, SurfaceRenderOptions, TerrainLightingOptions, TerrainLightingPreset,
    TileCache, TileCacheKey, TileCoord, TileImage, TilePathScheme, TileSet, TileStreamEvent,
};
