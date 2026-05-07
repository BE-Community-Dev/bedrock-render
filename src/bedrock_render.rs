//! Tile rendering utilities for Minecraft Bedrock worlds.
//!
//! `bedrock-render` provides palette management, tile planning, top-down render
//! modes, image encoding, cancellation, diagnostics, and Criterion-backed
//! benchmark support for map tooling.

pub mod editor;
mod error;
mod palette;
mod renderer;

pub use editor::{MapEditInvalidation, MapWorldEditor};
pub use error::{BedrockRenderError, BedrockRenderErrorKind, Result};
pub use palette::{PaletteImportReport, RenderPalette, RgbaColor};
pub use renderer::{
    AtlasRenderOptions, BakeDiagnostics, BakeOptions, BlockBoundaryRenderOptions,
    BlockVolumeRenderOptions, CHUNK_BAKE_CACHE_VERSION, ChunkBake, ChunkBakeCache,
    ChunkBakeCacheKey, ChunkBakePayload, ChunkBounds, ChunkPos, ChunkRegion, ChunkTileLayout,
    DEFAULT_PALETTE_VERSION, DepthPlane, Dimension, FastRgbaZstdHeader, FastRgbaZstdTile,
    HeightPlane, ImageFormat, LevelDbRenderSource, MAX_RENDER_THREADS, MAX_TILE_SIZE_PIXELS,
    MapRenderSession, MapRenderSessionConfig, MapRenderer, NbtTag, PlannedTile,
    RENDERER_CACHE_VERSION, RegionBake, RegionBakePayload, RegionCoord, RegionLayout,
    RenderBackend, RenderCachePolicy, RenderCancelFlag, RenderChunkSource,
    RenderCpuPipelineOptions, RenderDiagnostics, RenderDiagnosticsSink, RenderExecutionProfile,
    RenderGpuBackend, RenderGpuDiagnostics, RenderGpuFallbackPolicy, RenderGpuOptions,
    RenderGpuPipelineLevel, RenderJob, RenderLayout, RenderMemoryBudget, RenderMode, RenderOptions,
    RenderPerformanceOptions, RenderPerformanceProfile, RenderPipelineStats, RenderProgress,
    RenderProgressSink, RenderSidecarCachePolicy, RenderSurfaceLoadPolicy, RenderTaskControl,
    RenderThreadingOptions, RenderTilePriority, RenderWebTilesResult, ResolvedRenderBackend,
    RgbaPlane, SurfacePlane, SurfacePlaneAtlas, SurfaceRenderOptions, TerrainLightingOptions,
    TerrainLightingPreset, TileCache, TileCacheKey, TileCacheValidationOutcome, TileCoord,
    TileImage, TileManifestProbeRequest, TileManifestProbeResult, TilePathScheme, TileReadySource,
    TileSet, TileStreamEvent, decode_fast_rgba_zstd, decode_fast_rgba_zstd_header,
    encode_fast_rgba_zstd, encode_fast_rgba_zstd_with_validation, tile_cache_validation_value,
};
