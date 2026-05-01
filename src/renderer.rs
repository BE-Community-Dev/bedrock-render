#[cfg(feature = "gpu")]
#[path = "renderer/gpu.rs"]
mod gpu;
#[path = "renderer/pipeline.rs"]
mod pipeline;
#[cfg(not(feature = "gpu"))]
mod gpu {
    use super::pipeline::{GpuTileComposeInput, GpuTileComposeOutput};

    pub(super) fn compose_tile(
        _input: &GpuTileComposeInput<'_>,
    ) -> Result<GpuTileComposeOutput, String> {
        Err("bedrock-render was built without the gpu feature".to_string())
    }

    pub(super) fn feature_enabled() -> bool {
        false
    }
}

pub use pipeline::{
    BakeDiagnostics, BakeOptions, ChunkBake, ChunkBakePayload, ChunkRegion, ChunkTileLayout,
    DEFAULT_PALETTE_VERSION, DepthPlane, GPU_COMPOSE_SHADER_VERSION, HeightPlane, ImageFormat,
    MAX_RENDER_THREADS, MAX_TILE_SIZE_PIXELS, MapRenderer, PlannedTile, RENDERER_CACHE_VERSION,
    RegionBake, RegionBakePayload, RegionCoord, RegionLayout, RenderBackend, RenderCachePolicy,
    RenderCancelFlag, RenderDiagnostics, RenderDiagnosticsSink, RenderExecutionProfile, RenderJob,
    RenderLayout, RenderMemoryBudget, RenderMode, RenderOptions, RenderPipelineStats,
    RenderProgress, RenderProgressSink, RenderThreadingOptions, RenderWebTilesResult,
    ResolvedRenderBackend, RgbaPlane, SurfacePlane, SurfaceRenderOptions, TerrainLightingOptions,
    TerrainLightingPreset, TileCache, TileCacheKey, TileCoord, TileImage, TilePathScheme, TileSet,
};
