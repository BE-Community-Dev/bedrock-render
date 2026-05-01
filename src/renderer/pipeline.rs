use crate::error::{BedrockRenderError, Result};
use crate::palette::{RenderPalette, RgbaColor};
use bedrock_world::{
    BedrockWorld, BlockPos, ChunkPos, Dimension, RenderChunkData, RenderChunkLoadOptions,
    RenderChunkRegion, RenderRegionLoadOptions, RenderSurfaceSubchunkMode, SubChunk,
    SubChunkDecodeMode,
};
#[cfg(feature = "png")]
use image::codecs::png::PngEncoder;
#[cfg(feature = "webp")]
use image::codecs::webp::WebPEncoder;
#[cfg(any(feature = "png", feature = "webp"))]
use image::{ExtendedColorType, ImageEncoder};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::Instant;

/// Renderer cache schema version used in tile cache keys.
pub const RENDERER_CACHE_VERSION: u32 = 3;
/// Default embedded palette version used in tile cache keys.
pub const DEFAULT_PALETTE_VERSION: u32 = 1;
/// Maximum fixed worker thread count accepted by render options.
pub const MAX_RENDER_THREADS: usize = 512;
/// Maximum width or height of a rendered tile in pixels.
pub const MAX_TILE_SIZE_PIXELS: u32 = 4096;
const REGION_BAKE_ESTIMATED_BYTES_PER_CHUNK: usize = 4096;
const DEFAULT_EXPORT_MEMORY_BUDGET_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_INTERACTIVE_MEMORY_BUDGET_BYTES: usize = 512 * 1024 * 1024;
const MIN_AUTO_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;
const MAX_AUTO_MEMORY_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
const MISSING_HEIGHT: i16 = i16::MIN;
const GPU_COMPOSE_MIN_PIXELS: usize = 256 * 256;
/// GPU terrain-lighting shader version used in cache signatures.
pub const GPU_COMPOSE_SHADER_VERSION: u32 = 1;

/// Tile coordinate in chunk-tile space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileCoord {
    /// Tile X coordinate.
    pub x: i32,
    /// Tile Z coordinate.
    pub z: i32,
    /// Bedrock dimension rendered by this tile.
    pub dimension: Dimension,
}

/// Render mode used to sample and color world data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderMode {
    /// Resolved biome color layer sampled at world Y.
    Biome {
        /// World Y coordinate to sample.
        y: i32,
    },
    /// Deterministic diagnostic biome-id layer sampled at world Y.
    RawBiomeLayer {
        /// World Y coordinate to sample.
        y: i32,
    },
    /// Top-down terrain surface render using height maps and subchunks.
    SurfaceBlocks,
    /// Fixed block layer sampled at world Y.
    LayerBlocks {
        /// World Y coordinate to sample.
        y: i32,
    },
    /// Height gradient render from Bedrock height-map data.
    HeightMap,
    /// Cave diagnostic slice sampled at world Y.
    CaveSlice {
        /// World Y coordinate to sample.
        y: i32,
    },
}

/// Requested tile output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// Encode `TileImage::encoded` as WebP.
    WebP,
    /// Encode `TileImage::encoded` as PNG.
    Png,
    /// Return raw RGBA pixels without encoded bytes.
    Rgba,
}

/// A single tile render request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderJob {
    /// Tile coordinate to render.
    pub coord: TileCoord,
    /// Render mode to use.
    pub mode: RenderMode,
    /// Output tile width and height in pixels.
    pub tile_size: u32,
    /// Blocks represented by each output pixel.
    pub scale: u32,
    /// Number of output pixels generated per source block.
    pub pixels_per_block: u32,
}

impl RenderJob {
    /// Creates a default `256x256`, one-block-per-pixel render job.
    #[must_use]
    pub const fn new(coord: TileCoord, mode: RenderMode) -> Self {
        Self {
            coord,
            mode,
            tile_size: 256,
            scale: 1,
            pixels_per_block: 1,
        }
    }

    /// Creates a render job from a chunk-tile layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout is invalid or cannot produce an integer tile size.
    pub fn chunk_tile(coord: TileCoord, mode: RenderMode, layout: ChunkTileLayout) -> Result<Self> {
        validate_layout(layout)?;
        let tile_size = layout.tile_size().ok_or_else(|| {
            BedrockRenderError::Validation(
                "chunks_per_tile * 16 * pixels_per_block must be divisible by blocks_per_pixel"
                    .to_string(),
            )
        })?;
        Ok(Self {
            coord,
            mode,
            tile_size,
            scale: layout.blocks_per_pixel,
            pixels_per_block: layout.pixels_per_block,
        })
    }
}

/// Tile layout in chunks and pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderLayout {
    /// Number of chunks covered by one tile edge.
    pub chunks_per_tile: u32,
    /// Number of world blocks represented by one output pixel.
    pub blocks_per_pixel: u32,
    /// Number of output pixels generated for each world block.
    pub pixels_per_block: u32,
}

impl Default for RenderLayout {
    fn default() -> Self {
        Self {
            chunks_per_tile: 16,
            blocks_per_pixel: 1,
            pixels_per_block: 1,
        }
    }
}

impl RenderLayout {
    /// Computes the output tile size in pixels, if the layout is valid.
    #[must_use]
    pub fn tile_size(self) -> Option<u32> {
        let tile_blocks = self.chunks_per_tile.checked_mul(16)?;
        let output_pixels = tile_blocks.checked_mul(self.pixels_per_block)?;
        if self.blocks_per_pixel == 0
            || self.pixels_per_block == 0
            || output_pixels % self.blocks_per_pixel != 0
        {
            return None;
        }
        let tile_size = output_pixels / self.blocks_per_pixel;
        (tile_size <= MAX_TILE_SIZE_PIXELS).then_some(tile_size)
    }
}

/// Alias for render layouts used specifically by chunk tile planning.
pub type ChunkTileLayout = RenderLayout;

/// Region bake layout in chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionLayout {
    /// Number of chunks covered by one baked region edge.
    pub chunks_per_region: u32,
}

impl Default for RegionLayout {
    fn default() -> Self {
        Self {
            chunks_per_region: 32,
        }
    }
}

impl RegionLayout {
    /// Validates the region layout.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunks_per_region` is zero or greater than `256`.
    pub fn validate(self) -> Result<()> {
        if self.chunks_per_region == 0 {
            return Err(BedrockRenderError::Validation(
                "chunks_per_region must be greater than zero".to_string(),
            ));
        }
        if self.chunks_per_region > 256 {
            return Err(BedrockRenderError::Validation(
                "chunks_per_region must be <= 256".to_string(),
            ));
        }
        Ok(())
    }
}

/// Region coordinate in baked-region space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionCoord {
    /// Region X coordinate.
    pub x: i32,
    /// Region Z coordinate.
    pub z: i32,
    /// Bedrock dimension for this region.
    pub dimension: Dimension,
}

impl RegionCoord {
    /// Converts a chunk position into the region that contains it.
    #[must_use]
    pub fn from_chunk(pos: ChunkPos, layout: RegionLayout) -> Self {
        let span = i32::try_from(layout.chunks_per_region).unwrap_or(32).max(1);
        Self {
            x: pos.x.div_euclid(span),
            z: pos.z.div_euclid(span),
            dimension: pos.dimension,
        }
    }

    /// Returns the inclusive chunk bounds covered by this region.
    #[must_use]
    pub fn chunk_region(self, layout: RegionLayout) -> ChunkRegion {
        let span = i32::try_from(layout.chunks_per_region).unwrap_or(32).max(1);
        let min_chunk_x = self.x.saturating_mul(span);
        let min_chunk_z = self.z.saturating_mul(span);
        ChunkRegion::new(
            self.dimension,
            min_chunk_x,
            min_chunk_z,
            min_chunk_x.saturating_add(span.saturating_sub(1)),
            min_chunk_z.saturating_add(span.saturating_sub(1)),
        )
    }
}

/// Inclusive rectangular chunk region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRegion {
    /// Bedrock dimension for this region.
    pub dimension: Dimension,
    /// Minimum chunk X coordinate.
    pub min_chunk_x: i32,
    /// Minimum chunk Z coordinate.
    pub min_chunk_z: i32,
    /// Maximum chunk X coordinate.
    pub max_chunk_x: i32,
    /// Maximum chunk Z coordinate.
    pub max_chunk_z: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RegionBakeKey {
    coord: RegionCoord,
    mode: RenderMode,
}

#[derive(Debug, Clone)]
struct RegionPlan {
    key: RegionBakeKey,
    region: ChunkRegion,
    chunk_positions: Vec<ChunkPos>,
}

impl ChunkRegion {
    /// Creates an inclusive chunk region.
    #[must_use]
    pub const fn new(
        dimension: Dimension,
        min_chunk_x: i32,
        min_chunk_z: i32,
        max_chunk_x: i32,
        max_chunk_z: i32,
    ) -> Self {
        Self {
            dimension,
            min_chunk_x,
            min_chunk_z,
            max_chunk_x,
            max_chunk_z,
        }
    }
}

/// Path scheme for writing planned tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePathScheme {
    /// Hierarchical web-map layout.
    WebMap,
    /// Single flat filename layout.
    Flat,
}

/// Planned tile plus the chunk region it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTile {
    /// Render job for this tile.
    pub job: RenderJob,
    /// Inclusive chunk bounds needed by the tile.
    pub region: ChunkRegion,
    /// Layout used to produce the job.
    pub layout: ChunkTileLayout,
    /// Optional exact chunk positions for sparse exports.
    pub chunk_positions: Option<Vec<ChunkPos>>,
}

impl PlannedTile {
    /// Builds a relative path for this tile.
    #[must_use]
    pub fn relative_path(&self, scheme: TilePathScheme, extension: &str) -> PathBuf {
        let dimension = dimension_slug(self.job.coord.dimension);
        let mode = mode_slug(self.job.mode);
        let extension = extension.trim_start_matches('.');
        match scheme {
            TilePathScheme::WebMap => PathBuf::from(dimension)
                .join(mode)
                .join(format!(
                    "{}c-{}bpp-{}ppb",
                    self.layout.chunks_per_tile,
                    self.layout.blocks_per_pixel,
                    self.layout.pixels_per_block
                ))
                .join(self.job.coord.x.to_string())
                .join(format!("{}.{}", self.job.coord.z, extension)),
            TilePathScheme::Flat => PathBuf::from(format!(
                "{dimension}_{mode}_cpt{}_bpp{}_ppb{}_x{}_z{}.{}",
                self.layout.chunks_per_tile,
                self.layout.blocks_per_pixel,
                self.layout.pixels_per_block,
                self.job.coord.x,
                self.job.coord.z,
                extension
            )),
        }
    }
}

/// Runtime options used for tile, batch, region, and web-map rendering.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Requested output format.
    pub format: ImageFormat,
    /// Lossy encoder quality when the selected encoder supports quality.
    pub quality: u8,
    /// Requested CPU/GPU backend.
    pub backend: RenderBackend,
    /// Worker-thread policy.
    pub threading: RenderThreadingOptions,
    /// Execution profile used for automatic thread and memory budgets.
    pub execution_profile: RenderExecutionProfile,
    /// Region-bake memory budget policy.
    pub memory_budget: RenderMemoryBudget,
    /// Capacity for the internal compose/write pipeline; zero means automatic.
    pub pipeline_depth: usize,
    /// Optional cancellation flag observed by long-running operations.
    pub cancel: Option<RenderCancelFlag>,
    /// Optional progress callback.
    pub progress: Option<RenderProgressSink>,
    /// Optional diagnostics callback.
    pub diagnostics: Option<RenderDiagnosticsSink>,
    /// Surface render behavior.
    pub surface: SurfaceRenderOptions,
    /// Tile cache behavior.
    pub cache_policy: RenderCachePolicy,
    /// Region-bake layout used by region-backed rendering.
    pub region_layout: RegionLayout,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            format: ImageFormat::WebP,
            quality: 90,
            backend: RenderBackend::Auto,
            threading: RenderThreadingOptions::Auto,
            execution_profile: RenderExecutionProfile::Export,
            memory_budget: RenderMemoryBudget::Auto,
            pipeline_depth: 0,
            cancel: None,
            progress: None,
            diagnostics: None,
            surface: SurfaceRenderOptions::default(),
            cache_policy: RenderCachePolicy::Use,
            region_layout: RegionLayout::default(),
        }
    }
}

/// Requested render backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum RenderBackend {
    /// Resolve backend from options and environment.
    #[default]
    Auto,
    /// Prefer GPU composition when available.
    Gpu,
    /// Force CPU composition.
    Cpu,
}

impl RenderBackend {
    /// Returns the cache slug for this backend request.
    #[must_use]
    pub const fn cache_slug(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }

    fn requested_for_options(options: &RenderOptions) -> Self {
        match options.backend {
            Self::Auto => std::env::var("BEDROCK_RENDER_GPU")
                .ok()
                .and_then(|value| Self::parse_env_value(&value))
                .unwrap_or(Self::Auto),
            backend => backend,
        }
    }

    fn parse_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "1" | "true" | "yes" | "on" | "gpu" => Some(Self::Gpu),
            "0" | "false" | "no" | "off" | "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Backend actually used by a render operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ResolvedRenderBackend {
    /// CPU-only rendering.
    #[default]
    Cpu,
    /// GPU-only composition.
    Gpu,
    /// Mixed CPU and GPU work.
    Mixed,
    /// GPU was requested but CPU fallback was used.
    CpuFallback,
}

impl ResolvedRenderBackend {
    /// Human-readable backend label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Mixed => "mixed",
            Self::CpuFallback => "cpu-fallback",
        }
    }
}

/// Execution profile used to resolve automatic resources.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RenderExecutionProfile {
    /// Offline/export workload that may use all logical CPUs.
    #[default]
    Export,
    /// Foreground workload that reserves more CPU capacity for the caller.
    Interactive,
}

impl RenderExecutionProfile {
    /// Resolves an automatic thread count for a work-item count.
    #[must_use]
    pub fn default_auto_threads(self, work_items: usize) -> usize {
        let logical_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let resolved = match self {
            Self::Export => logical_threads,
            Self::Interactive => (logical_threads / 2).clamp(1, 6),
        };
        resolved.min(work_items.max(1))
    }
}

/// Region-bake memory budget policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RenderMemoryBudget {
    /// Choose a bounded budget from currently available system memory.
    #[default]
    Auto,
    /// Use an explicit byte budget.
    FixedBytes(u64),
    /// Disable memory-budget chunking.
    Disabled,
}

impl RenderMemoryBudget {
    /// Resolves this policy into a concrete byte budget, if enabled.
    #[must_use]
    pub fn resolve_bytes(self, profile: RenderExecutionProfile) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::FixedBytes(bytes) => usize::try_from(bytes).ok(),
            Self::Auto => Some(auto_memory_budget_bytes(profile)),
        }
    }
}

fn auto_memory_budget_bytes(profile: RenderExecutionProfile) -> usize {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available_bytes = usize::try_from(system.available_memory()).unwrap_or(0);
    let budget = available_bytes
        .checked_div(5)
        .unwrap_or(DEFAULT_EXPORT_MEMORY_BUDGET_BYTES);
    let capped = budget.clamp(MIN_AUTO_MEMORY_BUDGET_BYTES, MAX_AUTO_MEMORY_BUDGET_BYTES);
    match profile {
        RenderExecutionProfile::Export => capped,
        RenderExecutionProfile::Interactive => capped.min(DEFAULT_INTERACTIVE_MEMORY_BUDGET_BYTES),
    }
}

/// Aggregated timing and throughput stats from a web-map render.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderPipelineStats {
    /// Number of planned output tiles.
    pub planned_tiles: usize,
    /// Number of planned baked regions.
    pub planned_regions: usize,
    /// Number of unique chunks referenced by the render.
    pub unique_chunks: usize,
    /// Number of chunks baked.
    pub baked_chunks: usize,
    /// Number of regions baked.
    pub baked_regions: usize,
    /// Tile cache hits.
    pub cache_hits: usize,
    /// Tile cache misses.
    pub cache_misses: usize,
    /// Region cache hits.
    pub region_cache_hits: usize,
    /// Region cache misses.
    pub region_cache_misses: usize,
    /// Time spent baking chunks, in milliseconds.
    pub bake_ms: u128,
    /// Time spent baking regions, in milliseconds.
    pub region_bake_ms: u128,
    /// Time spent composing tiles, in milliseconds.
    pub tile_compose_ms: u128,
    /// Time spent encoding images, in milliseconds.
    pub encode_ms: u128,
    /// Time spent writing encoded tiles, in milliseconds.
    pub write_ms: u128,
    /// Aggregate worker idle time, in milliseconds.
    pub worker_idle_ms: u128,
    /// Aggregate queue wait time, in milliseconds.
    pub queue_wait_ms: u128,
    /// Peak region cache memory, in bytes.
    pub peak_cache_bytes: usize,
    /// Peak active task count.
    pub active_tasks_peak: usize,
    /// Peak worker thread count.
    pub peak_worker_threads: usize,
    /// Backend used by tile composition.
    pub resolved_backend: ResolvedRenderBackend,
    /// Number of tiles composed on GPU.
    pub gpu_tiles: usize,
    /// Number of tiles composed on CPU.
    pub cpu_tiles: usize,
    /// Number of tiles that fell back from GPU to CPU.
    pub gpu_fallbacks: usize,
    /// Time spent uploading GPU inputs, in milliseconds.
    pub gpu_upload_ms: u128,
    /// Time spent dispatching GPU work, in milliseconds.
    pub gpu_dispatch_ms: u128,
    /// Time spent reading GPU output, in milliseconds.
    pub gpu_readback_ms: u128,
    /// Name of the GPU adapter used, if any.
    pub gpu_adapter_name: Option<String>,
    /// Last GPU fallback reason, if any.
    pub gpu_fallback_reason: Option<String>,
}

/// Result returned by streaming web-map render APIs.
#[derive(Debug, Clone)]
pub struct RenderWebTilesResult {
    /// Aggregated render diagnostics.
    pub diagnostics: RenderDiagnostics,
    /// Aggregated pipeline statistics.
    pub stats: RenderPipelineStats,
}

/// Built-in terrain lighting presets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerrainLightingPreset {
    /// Disable terrain lighting.
    Off,
    /// Balanced terrain relief.
    Soft,
    /// Higher-contrast terrain relief.
    Strong,
}

/// Terrain lighting and relief parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainLightingOptions {
    /// Whether lighting is enabled.
    pub enabled: bool,
    /// Direction of the light source in degrees.
    pub light_azimuth_degrees: f32,
    /// Elevation of the light source in degrees.
    pub light_elevation_degrees: f32,
    /// Strength of the sampled terrain normal.
    pub normal_strength: f32,
    /// Strength of shadow darkening.
    pub shadow_strength: f32,
    /// Strength of highlight brightening.
    pub highlight_strength: f32,
    /// Ambient occlusion applied on relief.
    pub ambient_occlusion: f32,
    /// Maximum land shadow percentage.
    pub max_shadow: f32,
    /// Softness used to compress steep land slopes.
    pub land_slope_softness: f32,
    /// Strength of local edge relief.
    pub edge_relief_strength: f32,
    /// Minimum neighbor delta before edge relief is applied.
    pub edge_relief_threshold: f32,
    /// Maximum edge-relief shadow percentage.
    pub edge_relief_max_shadow: f32,
    /// Edge-relief highlight multiplier.
    pub edge_relief_highlight: f32,
    /// Whether underwater terrain relief is enabled.
    pub underwater_relief_enabled: bool,
    /// Underwater relief strength multiplier.
    pub underwater_relief_strength: f32,
    /// Depth at which underwater relief fades out.
    pub underwater_depth_fade: f32,
    /// Minimum underwater light contribution.
    pub underwater_min_light: f32,
}

impl TerrainLightingOptions {
    /// Returns a lighting configuration with lighting disabled.
    #[must_use]
    pub const fn off() -> Self {
        Self {
            enabled: false,
            light_azimuth_degrees: 315.0,
            light_elevation_degrees: 45.0,
            normal_strength: 0.0,
            shadow_strength: 0.0,
            highlight_strength: 0.0,
            ambient_occlusion: 0.0,
            max_shadow: 0.0,
            land_slope_softness: 0.0,
            edge_relief_strength: 0.0,
            edge_relief_threshold: 3.0,
            edge_relief_max_shadow: 0.0,
            edge_relief_highlight: 0.0,
            underwater_relief_enabled: false,
            underwater_relief_strength: 0.0,
            underwater_depth_fade: 1.0,
            underwater_min_light: 0.0,
        }
    }

    /// Returns the default balanced terrain lighting preset.
    #[must_use]
    pub const fn soft() -> Self {
        Self {
            enabled: true,
            light_azimuth_degrees: 315.0,
            light_elevation_degrees: 45.0,
            normal_strength: 1.25,
            shadow_strength: 0.42,
            highlight_strength: 0.28,
            ambient_occlusion: 0.04,
            max_shadow: 38.0,
            land_slope_softness: 6.0,
            edge_relief_strength: 0.18,
            edge_relief_threshold: 3.0,
            edge_relief_max_shadow: 18.0,
            edge_relief_highlight: 0.10,
            underwater_relief_enabled: true,
            underwater_relief_strength: 0.75,
            underwater_depth_fade: 8.0,
            underwater_min_light: 0.35,
        }
    }

    /// Returns a high-contrast terrain lighting preset.
    #[must_use]
    pub const fn strong() -> Self {
        Self {
            enabled: true,
            light_azimuth_degrees: 315.0,
            light_elevation_degrees: 42.0,
            normal_strength: 2.2,
            shadow_strength: 0.62,
            highlight_strength: 0.48,
            ambient_occlusion: 0.08,
            max_shadow: 48.0,
            land_slope_softness: 8.0,
            edge_relief_strength: 0.28,
            edge_relief_threshold: 3.0,
            edge_relief_max_shadow: 18.0,
            edge_relief_highlight: 0.10,
            underwater_relief_enabled: true,
            underwater_relief_strength: 1.25,
            underwater_depth_fade: 12.0,
            underwater_min_light: 0.25,
        }
    }

    /// Returns options for a built-in preset.
    #[must_use]
    pub const fn preset(preset: TerrainLightingPreset) -> Self {
        match preset {
            TerrainLightingPreset::Off => Self::off(),
            TerrainLightingPreset::Soft => Self::soft(),
            TerrainLightingPreset::Strong => Self::strong(),
        }
    }
}

impl Default for TerrainLightingOptions {
    fn default() -> Self {
        Self::soft()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerrainHeightNeighborhood {
    center: i16,
    north_west: i16,
    north: i16,
    north_east: i16,
    west: i16,
    east: i16,
    south_west: i16,
    south: i16,
    south_east: i16,
}

impl TerrainHeightNeighborhood {
    fn sobel_gradient(self) -> (f32, f32) {
        let dx =
            (f32::from(self.north_east) + 2.0 * f32::from(self.east) + f32::from(self.south_east)
                - f32::from(self.north_west)
                - 2.0 * f32::from(self.west)
                - f32::from(self.south_west))
                / 8.0;
        let dz =
            (f32::from(self.south_west) + 2.0 * f32::from(self.south) + f32::from(self.south_east)
                - f32::from(self.north_west)
                - 2.0 * f32::from(self.north)
                - f32::from(self.north_east))
                / 8.0;
        (dx, dz)
    }

    fn edge_relief(self, threshold: f32) -> TerrainEdgeRelief {
        let threshold = threshold.max(0.0);
        let center = f32::from(self.center);
        let mut higher_neighbor_delta = 0.0_f32;
        let mut lower_neighbor_delta = 0.0_f32;
        for neighbor in [
            self.north_west,
            self.north,
            self.north_east,
            self.west,
            self.east,
            self.south_west,
            self.south,
            self.south_east,
        ] {
            let delta = f32::from(neighbor) - center;
            higher_neighbor_delta = higher_neighbor_delta.max(delta);
            lower_neighbor_delta = lower_neighbor_delta.max(-delta);
        }

        let max_delta = higher_neighbor_delta.max(lower_neighbor_delta);
        if max_delta <= threshold {
            return TerrainEdgeRelief::default();
        }
        let amount = ((max_delta - threshold) / 16.0).clamp(0.0, 1.0);
        let pit_edge = higher_neighbor_delta >= lower_neighbor_delta;
        TerrainEdgeRelief {
            shadow: if pit_edge { amount } else { amount * 0.45 },
            highlight: if pit_edge { amount * 0.25 } else { amount },
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct TerrainEdgeRelief {
    shadow: f32,
    highlight: f32,
}

/// Options that affect surface block rendering.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceRenderOptions {
    /// Blend water over the block below instead of drawing opaque water.
    pub transparent_water: bool,
    /// Apply biome tinting to grass, foliage, and water.
    pub biome_tint: bool,
    /// Apply simple height shading.
    pub height_shading: bool,
    /// Terrain lighting options.
    pub lighting: TerrainLightingOptions,
    /// Skip air blocks while searching for a surface block.
    pub skip_air: bool,
    /// Draw unknown blocks with the diagnostic unknown-block color.
    pub render_unknown_blocks: bool,
}

impl Default for SurfaceRenderOptions {
    fn default() -> Self {
        Self {
            transparent_water: true,
            biome_tint: true,
            height_shading: true,
            lighting: TerrainLightingOptions::default(),
            skip_air: true,
            render_unknown_blocks: true,
        }
    }
}

/// Worker-thread selection policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RenderThreadingOptions {
    /// Resolve threads automatically from the execution profile.
    #[default]
    Auto,
    /// Use a fixed number of worker threads.
    Fixed(usize),
    /// Force single-threaded rendering.
    Single,
}

impl RenderThreadingOptions {
    /// Resolves a thread count using the export profile.
    #[must_use]
    pub fn resolve(self, work_items: usize) -> usize {
        self.resolve_unchecked(work_items)
    }

    /// Resolves a thread count without validating a fixed count.
    #[must_use]
    pub fn resolve_unchecked(self, work_items: usize) -> usize {
        self.resolve_for_profile_unchecked(RenderExecutionProfile::Export, work_items)
    }

    /// Resolves a thread count for a profile without validating a fixed count.
    #[must_use]
    pub fn resolve_for_profile_unchecked(
        self,
        profile: RenderExecutionProfile,
        work_items: usize,
    ) -> usize {
        match self {
            Self::Single => 1,
            Self::Fixed(threads) => threads.clamp(1, MAX_RENDER_THREADS),
            Self::Auto => profile.default_auto_threads(work_items),
        }
    }

    /// Resolves and validates a thread count using the export profile.
    ///
    /// # Errors
    ///
    /// Returns an error when an explicit fixed thread count is outside `1..=512`.
    pub fn resolve_checked(self, work_items: usize) -> Result<usize> {
        self.resolve_for_profile_checked(RenderExecutionProfile::Export, work_items)
    }

    /// Resolves and validates a thread count for a profile.
    ///
    /// # Errors
    ///
    /// Returns an error when an explicit fixed thread count is outside `1..=512`.
    pub fn resolve_for_profile_checked(
        self,
        profile: RenderExecutionProfile,
        work_items: usize,
    ) -> Result<usize> {
        match self {
            Self::Fixed(0) => Err(BedrockRenderError::Validation(
                "thread count must be in 1..=512".to_string(),
            )),
            Self::Fixed(threads) if threads > MAX_RENDER_THREADS => Err(
                BedrockRenderError::Validation("thread count must be in 1..=512".to_string()),
            ),
            _ => Ok(self.resolve_for_profile_unchecked(profile, work_items)),
        }
    }

    /// Resolves a thread count and applies an optional hard cap and reserved foreground capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when a fixed thread count or maximum thread cap is outside `1..=512`,
    /// or when an explicit fixed thread count exceeds the effective cap.
    pub fn resolve_for_profile_with_limits(
        self,
        profile: RenderExecutionProfile,
        work_items: usize,
        max_threads: Option<usize>,
        reserve_threads: usize,
    ) -> Result<usize> {
        let requested = self.resolve_for_profile_checked(profile, work_items)?;
        let mut allowed = MAX_RENDER_THREADS;
        if let Some(max_threads) = max_threads {
            if max_threads == 0 || max_threads > MAX_RENDER_THREADS {
                return Err(BedrockRenderError::Validation(
                    "max thread count must be in 1..=512".to_string(),
                ));
            }
            allowed = allowed.min(max_threads);
        }
        if reserve_threads > 0 {
            let available = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            allowed = allowed.min(available.saturating_sub(reserve_threads).max(1));
        }
        if matches!(self, Self::Fixed(_)) && requested > allowed {
            return Err(BedrockRenderError::Validation(format!(
                "thread count {requested} exceeds effective limit {allowed}"
            )));
        }
        Ok(requested.min(allowed).max(1))
    }
}

/// Tile cache behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RenderCachePolicy {
    /// Read and write cache entries.
    #[default]
    Use,
    /// Skip cache lookups and writes.
    Bypass,
}

/// Shared cancellation flag for long render operations.
#[derive(Debug, Clone, Default)]
pub struct RenderCancelFlag(Arc<AtomicBool>);

impl RenderCancelFlag {
    /// Creates a new unset cancellation flag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the flag as cancelled.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Callback sink for tile render progress.
#[derive(Clone)]
pub struct RenderProgressSink {
    inner: Arc<dyn Fn(RenderProgress) + Send + Sync>,
}

impl std::fmt::Debug for RenderProgressSink {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RenderProgressSink")
            .finish_non_exhaustive()
    }
}

impl RenderProgressSink {
    /// Creates a new progress sink from a callback.
    #[must_use]
    pub fn new(callback: impl Fn(RenderProgress) + Send + Sync + 'static) -> Self {
        Self {
            inner: Arc::new(callback),
        }
    }

    fn emit(&self, progress: RenderProgress) {
        (self.inner)(progress);
    }
}

/// Progress update emitted during batch rendering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderProgress {
    /// Number of tiles completed.
    pub completed_tiles: usize,
    /// Total number of tiles in the operation.
    pub total_tiles: usize,
}

/// Diagnostic counters emitted during rendering.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderDiagnostics {
    /// Number of sampled positions whose chunk was missing.
    pub missing_chunks: usize,
    /// Number of sampled positions whose height map was missing.
    pub missing_heightmaps: usize,
    /// Number of pixels drawn from unknown block names.
    pub unknown_blocks: usize,
    /// Unknown block counts grouped by block name.
    pub unknown_blocks_by_name: BTreeMap<String, usize>,
    /// Number of pixels that used fallback rendering.
    pub fallback_pixels: usize,
    /// Number of transparent pixels emitted.
    pub transparent_pixels: usize,
    /// Number of purple diagnostic pixels emitted.
    pub purple_error_pixels: usize,
    /// Number of chunks baked while rendering.
    pub baked_chunks: usize,
    /// Number of cache hits.
    pub cache_hits: usize,
    /// Number of cache misses.
    pub cache_misses: usize,
    /// Number of GPU fallback events.
    pub gpu_fallbacks: usize,
}

impl RenderDiagnostics {
    /// Adds another diagnostics value into this one with saturating counters.
    pub fn add(&mut self, other: Self) {
        self.missing_chunks = self.missing_chunks.saturating_add(other.missing_chunks);
        self.missing_heightmaps = self
            .missing_heightmaps
            .saturating_add(other.missing_heightmaps);
        self.unknown_blocks = self.unknown_blocks.saturating_add(other.unknown_blocks);
        for (name, count) in other.unknown_blocks_by_name {
            *self.unknown_blocks_by_name.entry(name).or_default() += count;
        }
        self.fallback_pixels = self.fallback_pixels.saturating_add(other.fallback_pixels);
        self.transparent_pixels = self
            .transparent_pixels
            .saturating_add(other.transparent_pixels);
        self.purple_error_pixels = self
            .purple_error_pixels
            .saturating_add(other.purple_error_pixels);
        self.baked_chunks = self.baked_chunks.saturating_add(other.baked_chunks);
        self.cache_hits = self.cache_hits.saturating_add(other.cache_hits);
        self.cache_misses = self.cache_misses.saturating_add(other.cache_misses);
        self.gpu_fallbacks = self.gpu_fallbacks.saturating_add(other.gpu_fallbacks);
    }

    fn record_unknown_block(&mut self, name: &str) {
        self.unknown_blocks = self.unknown_blocks.saturating_add(1);
        self.purple_error_pixels = self.purple_error_pixels.saturating_add(1);
        *self
            .unknown_blocks_by_name
            .entry(name.to_string())
            .or_default() += 1;
    }

    fn record_transparent_pixel(&mut self) {
        self.fallback_pixels = self.fallback_pixels.saturating_add(1);
        self.transparent_pixels = self.transparent_pixels.saturating_add(1);
    }
}

/// Callback sink for aggregated render diagnostics.
#[derive(Clone)]
pub struct RenderDiagnosticsSink {
    inner: Arc<dyn Fn(RenderDiagnostics) + Send + Sync>,
}

impl std::fmt::Debug for RenderDiagnosticsSink {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RenderDiagnosticsSink")
            .finish_non_exhaustive()
    }
}

impl RenderDiagnosticsSink {
    /// Creates a new diagnostics sink from a callback.
    #[must_use]
    pub fn new(callback: impl Fn(RenderDiagnostics) + Send + Sync + 'static) -> Self {
        Self {
            inner: Arc::new(callback),
        }
    }

    fn emit(&self, diagnostics: RenderDiagnostics) {
        (self.inner)(diagnostics);
    }
}

/// Rendered tile pixels and optional encoded image bytes.
#[derive(Debug, Clone)]
pub struct TileImage {
    /// Tile coordinate rendered by this image.
    pub coord: TileCoord,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw RGBA bytes.
    pub rgba: Vec<u8>,
    /// Encoded image bytes when an encoded format was requested.
    pub encoded: Option<Vec<u8>>,
}

/// Collection of rendered tiles.
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Rendered tile images.
    pub tiles: Vec<TileImage>,
}

/// Options used when baking a chunk before tile composition.
#[derive(Debug, Clone)]
pub struct BakeOptions {
    /// Render mode to bake.
    pub mode: RenderMode,
    /// Surface render behavior used for surface bakes.
    pub surface: SurfaceRenderOptions,
}

impl Default for BakeOptions {
    fn default() -> Self {
        Self {
            mode: RenderMode::SurfaceBlocks,
            surface: SurfaceRenderOptions::default(),
        }
    }
}

/// Diagnostics returned by bake operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BakeDiagnostics {
    /// Render diagnostics collected while baking.
    pub render: RenderDiagnostics,
}

/// Baked data for one chunk.
#[derive(Debug, Clone)]
pub struct ChunkBake {
    /// Chunk position.
    pub pos: ChunkPos,
    /// Render mode used for this bake.
    pub mode: RenderMode,
    /// Baked payload.
    pub payload: ChunkBakePayload,
    /// Diagnostics collected while baking.
    pub diagnostics: RenderDiagnostics,
}

/// Two-dimensional RGBA color plane.
#[derive(Debug, Clone)]
pub struct RgbaPlane {
    /// Plane width.
    pub width: u32,
    /// Plane height.
    pub height: u32,
    /// RGBA colors in row-major order.
    pub pixels: Vec<RgbaColor>,
}

/// Two-dimensional height plane.
#[derive(Debug, Clone)]
pub struct HeightPlane {
    /// Plane width.
    pub width: u32,
    /// Plane height.
    pub height: u32,
    /// Heights in row-major order.
    pub heights: Vec<i16>,
}

/// Two-dimensional water-depth plane.
#[derive(Debug, Clone)]
pub struct DepthPlane {
    /// Plane width.
    pub width: u32,
    /// Plane height.
    pub height: u32,
    /// Depth values in row-major order.
    pub depths: Vec<u8>,
}

/// Baked surface data for color, height, relief, and water depth.
#[derive(Debug, Clone)]
pub struct SurfacePlane {
    /// Surface colors.
    pub colors: RgbaPlane,
    /// Surface heights.
    pub heights: HeightPlane,
    /// Heights used for terrain relief, such as seabed heights under water.
    pub relief_heights: HeightPlane,
    /// Water depths for transparent-water rendering.
    pub water_depths: DepthPlane,
}

/// Payload produced by baking a chunk.
#[derive(Debug, Clone)]
pub enum ChunkBakePayload {
    /// Simple color plane.
    Colors(RgbaPlane),
    /// Surface render planes.
    Surface(SurfacePlane),
    /// Height-map colors and source heights.
    HeightMap {
        /// Height-map colors.
        colors: RgbaPlane,
        /// Source heights.
        heights: HeightPlane,
    },
}

/// Payload produced by baking a region.
#[derive(Debug, Clone)]
pub enum RegionBakePayload {
    /// Simple color plane.
    Colors(RgbaPlane),
    /// Surface render planes.
    Surface(SurfacePlane),
    /// Height-map colors and source heights.
    HeightMap {
        /// Height-map colors.
        colors: RgbaPlane,
        /// Source heights.
        heights: HeightPlane,
    },
}

/// Baked data for one region.
#[derive(Debug, Clone)]
pub struct RegionBake {
    /// Region coordinate.
    pub coord: RegionCoord,
    /// Region layout.
    pub layout: RegionLayout,
    /// Render mode used for this bake.
    pub mode: RenderMode,
    /// Inclusive chunk bounds covered by this bake.
    pub chunk_region: ChunkRegion,
    /// Baked region payload.
    pub payload: RegionBakePayload,
    /// Diagnostics collected while baking.
    pub diagnostics: RenderDiagnostics,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) struct GpuTileComposeInput<'a> {
    pub width: u32,
    pub height: u32,
    pub colors: &'a [u32],
    pub heights: &'a [i32],
    pub water_depths: &'a [u32],
    pub lighting_enabled: bool,
    pub lighting: TerrainLightingOptions,
}

pub(super) struct GpuTileComposeOutput {
    pub rgba: Vec<u8>,
    pub upload_ms: u128,
    pub dispatch_ms: u128,
    pub readback_ms: u128,
    pub adapter_name: String,
}

#[derive(Debug, Clone, Default)]
struct TileComposeStats {
    backend: ResolvedRenderBackend,
    gpu_tiles: usize,
    cpu_tiles: usize,
    gpu_fallbacks: usize,
    gpu_upload_ms: u128,
    gpu_dispatch_ms: u128,
    gpu_readback_ms: u128,
    gpu_adapter_name: Option<String>,
    gpu_fallback_reason: Option<String>,
}

struct PreparedTileCompose {
    colors: Vec<u32>,
    heights: Vec<i32>,
    water_depths: Vec<u32>,
    diagnostics: RenderDiagnostics,
    lighting_enabled: bool,
}

impl TileComposeStats {
    fn cpu() -> Self {
        Self {
            backend: ResolvedRenderBackend::Cpu,
            cpu_tiles: 1,
            ..Self::default()
        }
    }

    fn cpu_fallback(reason: String) -> Self {
        Self {
            backend: ResolvedRenderBackend::CpuFallback,
            cpu_tiles: 1,
            gpu_fallbacks: 1,
            gpu_fallback_reason: Some(reason),
            ..Self::default()
        }
    }

    fn gpu(output: &GpuTileComposeOutput) -> Self {
        Self {
            backend: ResolvedRenderBackend::Gpu,
            gpu_tiles: 1,
            gpu_upload_ms: output.upload_ms,
            gpu_dispatch_ms: output.dispatch_ms,
            gpu_readback_ms: output.readback_ms,
            gpu_adapter_name: Some(output.adapter_name.clone()),
            ..Self::default()
        }
    }

    fn add(&mut self, other: Self) {
        if self.gpu_tiles == 0 && self.cpu_tiles == 0 && self.gpu_fallbacks == 0 {
            self.backend = other.backend;
        } else {
            self.backend = merge_backends(self.backend, other.backend);
        }
        self.gpu_tiles = self.gpu_tiles.saturating_add(other.gpu_tiles);
        self.cpu_tiles = self.cpu_tiles.saturating_add(other.cpu_tiles);
        self.gpu_fallbacks = self.gpu_fallbacks.saturating_add(other.gpu_fallbacks);
        self.gpu_upload_ms = self.gpu_upload_ms.saturating_add(other.gpu_upload_ms);
        self.gpu_dispatch_ms = self.gpu_dispatch_ms.saturating_add(other.gpu_dispatch_ms);
        self.gpu_readback_ms = self.gpu_readback_ms.saturating_add(other.gpu_readback_ms);
        if self.gpu_adapter_name.is_none() {
            self.gpu_adapter_name = other.gpu_adapter_name;
        }
        if self.gpu_fallback_reason.is_none() {
            self.gpu_fallback_reason = other.gpu_fallback_reason;
        }
    }
}

fn merge_backends(
    left: ResolvedRenderBackend,
    right: ResolvedRenderBackend,
) -> ResolvedRenderBackend {
    match (left, right) {
        (ResolvedRenderBackend::CpuFallback, _) | (_, ResolvedRenderBackend::CpuFallback) => {
            ResolvedRenderBackend::CpuFallback
        }
        (ResolvedRenderBackend::Mixed, _) | (_, ResolvedRenderBackend::Mixed) => {
            ResolvedRenderBackend::Mixed
        }
        (ResolvedRenderBackend::Cpu, ResolvedRenderBackend::Cpu) => ResolvedRenderBackend::Cpu,
        (ResolvedRenderBackend::Gpu, ResolvedRenderBackend::Gpu) => ResolvedRenderBackend::Gpu,
        (ResolvedRenderBackend::Cpu, ResolvedRenderBackend::Gpu)
        | (ResolvedRenderBackend::Gpu, ResolvedRenderBackend::Cpu) => ResolvedRenderBackend::Mixed,
    }
}

impl RenderPipelineStats {
    fn add_tile_compose_stats(&mut self, stats: TileComposeStats) {
        if self.gpu_tiles == 0 && self.cpu_tiles == 0 && self.gpu_fallbacks == 0 {
            self.resolved_backend = stats.backend;
        } else {
            self.resolved_backend = merge_backends(self.resolved_backend, stats.backend);
        }
        self.gpu_tiles = self.gpu_tiles.saturating_add(stats.gpu_tiles);
        self.cpu_tiles = self.cpu_tiles.saturating_add(stats.cpu_tiles);
        self.gpu_fallbacks = self.gpu_fallbacks.saturating_add(stats.gpu_fallbacks);
        self.gpu_upload_ms = self.gpu_upload_ms.saturating_add(stats.gpu_upload_ms);
        self.gpu_dispatch_ms = self.gpu_dispatch_ms.saturating_add(stats.gpu_dispatch_ms);
        self.gpu_readback_ms = self.gpu_readback_ms.saturating_add(stats.gpu_readback_ms);
        if self.gpu_adapter_name.is_none() {
            self.gpu_adapter_name = stats.gpu_adapter_name;
        }
        if self.gpu_fallback_reason.is_none() {
            self.gpu_fallback_reason = stats.gpu_fallback_reason;
        }
    }
}

impl RgbaPlane {
    fn new(width: u32, height: u32, fill: RgbaColor) -> Result<Self> {
        Ok(Self {
            width,
            height,
            pixels: vec![fill; plane_len(width, height)?],
        })
    }

    fn color_at(&self, pixel_x: u32, pixel_z: u32) -> Option<RgbaColor> {
        plane_index(self.width, self.height, pixel_x, pixel_z)
            .and_then(|index| self.pixels.get(index).copied())
    }

    fn set_color(&mut self, pixel_x: u32, pixel_z: u32, color: RgbaColor) {
        if let Some(index) = plane_index(self.width, self.height, pixel_x, pixel_z)
            && let Some(pixel) = self.pixels.get_mut(index)
        {
            *pixel = color;
        }
    }
}

impl HeightPlane {
    fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            width,
            height,
            heights: vec![MISSING_HEIGHT; plane_len(width, height)?],
        })
    }

    fn height_at(&self, pixel_x: u32, pixel_z: u32) -> Option<i16> {
        let height = plane_index(self.width, self.height, pixel_x, pixel_z)
            .and_then(|index| self.heights.get(index).copied())?;
        (height != MISSING_HEIGHT).then_some(height)
    }

    fn set_height(&mut self, pixel_x: u32, pixel_z: u32, height: i16) {
        if let Some(index) = plane_index(self.width, self.height, pixel_x, pixel_z)
            && let Some(pixel) = self.heights.get_mut(index)
        {
            *pixel = height;
        }
    }
}

impl DepthPlane {
    fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            width,
            height,
            depths: vec![0; plane_len(width, height)?],
        })
    }

    fn depth_at(&self, pixel_x: u32, pixel_z: u32) -> u8 {
        plane_index(self.width, self.height, pixel_x, pixel_z)
            .and_then(|index| self.depths.get(index).copied())
            .unwrap_or(0)
    }

    fn set_depth(&mut self, pixel_x: u32, pixel_z: u32, depth: u8) {
        if let Some(index) = plane_index(self.width, self.height, pixel_x, pixel_z)
            && let Some(pixel) = self.depths.get_mut(index)
        {
            *pixel = depth;
        }
    }
}

impl RegionBake {
    fn color_at_chunk_local(&self, chunk: ChunkPos, local_x: u8, local_z: u8) -> Option<RgbaColor> {
        let (pixel_x, pixel_z) = self.region_pixel(chunk, local_x, local_z)?;
        self.color_at_region_pixel(pixel_x, pixel_z)
    }

    fn height_at_chunk_local(&self, chunk: ChunkPos, local_x: u8, local_z: u8) -> Option<i16> {
        let (pixel_x, pixel_z) = self.region_pixel(chunk, local_x, local_z)?;
        self.height_at_region_pixel(pixel_x, pixel_z)
    }

    fn region_pixel(&self, chunk: ChunkPos, local_x: u8, local_z: u8) -> Option<(u32, u32)> {
        let relative_chunk_x = chunk.x.checked_sub(self.chunk_region.min_chunk_x)?;
        let relative_chunk_z = chunk.z.checked_sub(self.chunk_region.min_chunk_z)?;
        let chunk_width = self
            .chunk_region
            .max_chunk_x
            .checked_sub(self.chunk_region.min_chunk_x)?
            .checked_add(1)?;
        let chunk_height = self
            .chunk_region
            .max_chunk_z
            .checked_sub(self.chunk_region.min_chunk_z)?
            .checked_add(1)?;
        if relative_chunk_x < 0
            || relative_chunk_z < 0
            || relative_chunk_x >= chunk_width
            || relative_chunk_z >= chunk_height
        {
            return None;
        }
        let pixel_x = u32::try_from(relative_chunk_x).ok()?.saturating_mul(16) + u32::from(local_x);
        let pixel_z = u32::try_from(relative_chunk_z).ok()?.saturating_mul(16) + u32::from(local_z);
        Some((pixel_x, pixel_z))
    }

    fn color_at_region_pixel(&self, pixel_x: u32, pixel_z: u32) -> Option<RgbaColor> {
        match &self.payload {
            RegionBakePayload::Colors(plane) => plane.color_at(pixel_x, pixel_z),
            RegionBakePayload::Surface(plane) => plane.colors.color_at(pixel_x, pixel_z),
            RegionBakePayload::HeightMap { colors, .. } => colors.color_at(pixel_x, pixel_z),
        }
    }

    fn height_at_region_pixel(&self, pixel_x: u32, pixel_z: u32) -> Option<i16> {
        match &self.payload {
            RegionBakePayload::Surface(plane) => plane.relief_heights.height_at(pixel_x, pixel_z),
            RegionBakePayload::HeightMap { heights, .. } => heights.height_at(pixel_x, pixel_z),
            RegionBakePayload::Colors(_) => None,
        }
    }

    fn water_depth_at_region_pixel(&self, pixel_x: u32, pixel_z: u32) -> u8 {
        match &self.payload {
            RegionBakePayload::Surface(plane) => plane.water_depths.depth_at(pixel_x, pixel_z),
            RegionBakePayload::Colors(_) | RegionBakePayload::HeightMap { .. } => 0,
        }
    }
}

fn plane_len(width: u32, height: u32) -> Result<usize> {
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| BedrockRenderError::Validation("plane pixel count overflow".to_string()))
}

fn plane_index(width: u32, height: u32, pixel_x: u32, pixel_z: u32) -> Option<usize> {
    if pixel_x >= width || pixel_z >= height {
        return None;
    }
    usize::try_from(pixel_z.checked_mul(width)?.checked_add(pixel_x)?).ok()
}

/// Cache key for encoded tile images.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileCacheKey {
    /// Stable world identifier.
    pub world_id: String,
    /// World content signature.
    pub world_signature: String,
    /// Renderer cache schema version.
    pub renderer_version: u32,
    /// Palette cache schema version.
    pub palette_version: u32,
    /// Bedrock dimension.
    pub dimension: Dimension,
    /// Render mode slug.
    pub mode: String,
    /// Chunks covered per tile edge.
    pub chunks_per_tile: u32,
    /// Blocks represented per output pixel.
    pub blocks_per_pixel: u32,
    /// Output pixels generated per source block.
    pub pixels_per_block: u32,
    /// Tile X coordinate.
    pub tile_x: i32,
    /// Tile Z coordinate.
    pub tile_z: i32,
    /// Encoded file extension without a leading dot.
    pub extension: String,
}

/// Small memory and disk cache for rendered tiles.
#[derive(Debug)]
pub struct TileCache {
    root: PathBuf,
    memory_limit: usize,
    memory_order: VecDeque<TileCacheKey>,
    memory: BTreeMap<TileCacheKey, TileImage>,
}

impl TileCache {
    /// Creates a tile cache rooted at a filesystem path.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, memory_limit: usize) -> Self {
        Self {
            root: root.into(),
            memory_limit: memory_limit.max(1),
            memory_order: VecDeque::new(),
            memory: BTreeMap::new(),
        }
    }

    /// Returns a tile from the in-memory cache.
    #[must_use]
    pub fn get_memory(&mut self, key: &TileCacheKey) -> Option<TileImage> {
        self.memory.get(key).cloned()
    }

    /// Reads encoded tile bytes from disk, if present.
    #[must_use]
    pub fn get_disk(&self, key: &TileCacheKey) -> Option<Vec<u8>> {
        fs::read(self.path_for_key(key)).ok()
    }

    /// Inserts a tile into memory and writes encoded bytes to disk when present.
    ///
    /// # Errors
    ///
    /// Returns an error if encoded tile bytes cannot be written.
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert(&mut self, key: TileCacheKey, tile: TileImage) -> Result<()> {
        if let Some(encoded) = tile.encoded.as_deref() {
            self.write_encoded(&key, encoded)?;
        }
        self.memory_order.push_back(key.clone());
        self.memory.insert(key.clone(), tile);
        while self.memory.len() > self.memory_limit {
            let Some(old_key) = self.memory_order.pop_front() else {
                break;
            };
            if old_key != key {
                self.memory.remove(&old_key);
            }
        }
        Ok(())
    }

    /// Writes encoded tile bytes to the deterministic disk-cache path.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be created or the file cannot be written.
    pub fn write_encoded(&self, key: &TileCacheKey, encoded: &[u8]) -> Result<()> {
        let path = self.path_for_key(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                BedrockRenderError::io("failed to create tile cache directory", error)
            })?;
        }
        fs::write(path, encoded)
            .map_err(|error| BedrockRenderError::io("failed to write encoded tile cache", error))
    }

    /// Returns the full disk-cache path for a tile key.
    #[must_use]
    pub fn path_for_key(&self, key: &TileCacheKey) -> PathBuf {
        self.root
            .join("web-tiles")
            .join(&key.world_id)
            .join(&key.world_signature)
            .join(format!(
                "r{}-p{}",
                key.renderer_version, key.palette_version
            ))
            .join(dimension_slug(key.dimension))
            .join(&key.mode)
            .join(format!(
                "{}c-{}bpp-{}ppb",
                key.chunks_per_tile, key.blocks_per_pixel, key.pixels_per_block
            ))
            .join(key.tile_x.to_string())
            .join(format!("{}.{}", key.tile_z, key.extension))
    }
}

/// Renderer handle for Bedrock world tile rendering.
#[derive(Clone)]
pub struct MapRenderer {
    world: Arc<BedrockWorld>,
    palette: RenderPalette,
}

impl MapRenderer {
    /// Creates a renderer from a world handle and palette.
    #[must_use]
    pub fn new(world: Arc<BedrockWorld>, palette: RenderPalette) -> Self {
        Self { world, palette }
    }

    /// Renders one tile with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is invalid, world reads fail, rendering is cancelled,
    /// or the requested output cannot be encoded.
    pub fn render_tile_blocking(&self, job: RenderJob) -> Result<TileImage> {
        self.render_tile_with_options_blocking(job, &RenderOptions::default())
    }

    /// Renders one tile with explicit options.
    ///
    /// # Errors
    ///
    /// Returns an error if the job/options are invalid, world reads fail, rendering is
    /// cancelled, or the requested output cannot be encoded.
    pub fn render_tile_with_options_blocking(
        &self,
        job: RenderJob,
        options: &RenderOptions,
    ) -> Result<TileImage> {
        validate_job(&job)?;
        self.render_tile_from_bake_blocking(job, options)
    }

    /// Renders a batch of tiles.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile fails, rendering is cancelled, or the thread options
    /// are invalid.
    #[allow(clippy::needless_pass_by_value)]
    pub fn render_tiles_blocking(
        &self,
        jobs: impl IntoIterator<Item = RenderJob>,
        options: RenderOptions,
    ) -> Result<Vec<TileImage>> {
        let jobs = jobs.into_iter().collect::<Vec<_>>();
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let total_tiles = jobs.len();
        let worker_count = options
            .threading
            .resolve_for_profile_checked(options.execution_profile, total_tiles)?;
        if worker_count == 1 {
            let mut tiles = Vec::with_capacity(total_tiles);
            for job in jobs {
                check_cancelled(&options)?;
                tiles.push(self.render_tile_with_options_blocking(job, &options)?);
                emit_progress(&options, tiles.len(), total_tiles);
            }
            return Ok(tiles);
        }

        let next_job = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::channel::<Result<(usize, TileImage)>>();
        thread::scope(|scope| {
            for _ in 0..worker_count {
                let renderer = self.clone();
                let mut options = options.clone();
                options.threading = RenderThreadingOptions::Single;
                let next_job = Arc::clone(&next_job);
                let sender = sender.clone();
                let jobs = &jobs;
                scope.spawn(move || {
                    loop {
                        if check_cancelled(&options).is_err() {
                            let _ = sender.send(Err(BedrockRenderError::Cancelled));
                            return;
                        }
                        let index = next_job.fetch_add(1, Ordering::Relaxed);
                        if index >= jobs.len() {
                            return;
                        }
                        let job = jobs[index].clone();
                        let result = renderer
                            .render_tile_with_options_blocking(job, &options)
                            .map(|tile| (index, tile));
                        if sender.send(result).is_err() {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            let mut completed = 0usize;
            let mut tiles = Vec::with_capacity(total_tiles);
            for result in receiver {
                let (index, tile) = result?;
                tiles.push((index, tile));
                completed = completed.saturating_add(1);
                emit_progress(&options, completed, total_tiles);
                if completed == total_tiles {
                    break;
                }
            }
            tiles.sort_by_key(|(index, _)| *index);
            Ok(tiles.into_iter().map(|(_, tile)| tile).collect())
        })
    }

    /// Plans all tiles intersecting a chunk region.
    ///
    /// # Errors
    ///
    /// Returns an error if the region or layout is invalid or the resulting plan is too large.
    pub fn plan_region_tiles(
        region: ChunkRegion,
        mode: RenderMode,
        layout: ChunkTileLayout,
    ) -> Result<Vec<PlannedTile>> {
        validate_region(region)?;
        validate_layout(layout)?;
        let chunks_per_tile = i32::try_from(layout.chunks_per_tile).map_err(|_| {
            BedrockRenderError::Validation("chunks_per_tile is outside i32 range".to_string())
        })?;
        let min_tile_x = floor_div(region.min_chunk_x, chunks_per_tile);
        let max_tile_x = floor_div(region.max_chunk_x, chunks_per_tile);
        let min_tile_z = floor_div(region.min_chunk_z, chunks_per_tile);
        let max_tile_z = floor_div(region.max_chunk_z, chunks_per_tile);
        let x_tile_count = i64::from(max_tile_x) - i64::from(min_tile_x) + 1;
        let z_tile_count = i64::from(max_tile_z) - i64::from(min_tile_z) + 1;
        let capacity =
            usize::try_from(x_tile_count.saturating_mul(z_tile_count)).map_err(|_| {
                BedrockRenderError::Validation("planned tile count is too large".to_string())
            })?;
        let mut tiles = Vec::with_capacity(capacity);
        for z in min_tile_z..=max_tile_z {
            for x in min_tile_x..=max_tile_x {
                let job = RenderJob::chunk_tile(
                    TileCoord {
                        x,
                        z,
                        dimension: region.dimension,
                    },
                    mode,
                    layout,
                )?;
                tiles.push(PlannedTile {
                    job,
                    region,
                    layout,
                    chunk_positions: None,
                });
            }
        }
        Ok(tiles)
    }

    /// Renders all tiles intersecting a region and returns them as a tile set.
    ///
    /// # Errors
    ///
    /// Returns an error if planning, rendering, encoding, cancellation, or result
    /// aggregation fails.
    pub fn render_region_tiles_blocking(
        &self,
        region: ChunkRegion,
        mode: RenderMode,
        layout: ChunkTileLayout,
        options: RenderOptions,
    ) -> Result<TileSet> {
        let planned_tiles = Self::plan_region_tiles(region, mode, layout)?;
        let tiles = Arc::new(Mutex::new(Vec::with_capacity(planned_tiles.len())));
        self.render_tiles_from_regions_blocking(&planned_tiles, options, {
            let tiles = Arc::clone(&tiles);
            move |_planned, tile| {
                tiles
                    .lock()
                    .map_err(|_| {
                        BedrockRenderError::Validation(
                            "render region tile result lock failed".to_string(),
                        )
                    })?
                    .push(tile);
                Ok(())
            }
        })?;
        let tiles = Arc::try_unwrap(tiles)
            .map_err(|_| {
                BedrockRenderError::Validation(
                    "render region tile results still shared".to_string(),
                )
            })?
            .into_inner()
            .map_err(|_| {
                BedrockRenderError::Validation("render region tile result lock failed".to_string())
            })?;
        Ok(TileSet { tiles })
    }

    /// Renders an arbitrary iterator of jobs and wraps the result in a tile set.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile render fails.
    pub fn render_region_blocking(
        &self,
        jobs: impl IntoIterator<Item = RenderJob>,
        options: RenderOptions,
    ) -> Result<TileSet> {
        Ok(TileSet {
            tiles: self.render_tiles_blocking(jobs, options)?,
        })
    }

    /// Bakes all chunks in a region.
    ///
    /// # Errors
    ///
    /// Returns an error if the region layout is invalid, world reads fail, or baking is cancelled.
    pub fn bake_region_blocking(
        &self,
        coord: RegionCoord,
        options: &RenderOptions,
        mode: RenderMode,
    ) -> Result<RegionBake> {
        options.region_layout.validate()?;
        let chunk_region = coord.chunk_region(options.region_layout);
        self.bake_region_chunk_region_blocking(coord, chunk_region, options, mode)
    }

    fn bake_region_chunk_region_blocking(
        &self,
        coord: RegionCoord,
        chunk_region: ChunkRegion,
        options: &RenderOptions,
        mode: RenderMode,
    ) -> Result<RegionBake> {
        options.region_layout.validate()?;
        let fixed_y = match mode {
            RenderMode::LayerBlocks { y } | RenderMode::CaveSlice { y } => Some(y),
            _ => None,
        };
        let biome_y = match mode {
            RenderMode::Biome { y } | RenderMode::RawBiomeLayer { y } => Some(y),
            _ => Some(64),
        };
        let uses_surface = matches!(mode, RenderMode::SurfaceBlocks);
        let loads_biome_plane = matches!(
            mode,
            RenderMode::SurfaceBlocks | RenderMode::Biome { .. } | RenderMode::RawBiomeLayer { .. }
        );
        let data = self.world.load_render_region_blocking(
            RenderChunkRegion {
                dimension: chunk_region.dimension,
                min_chunk_x: chunk_region.min_chunk_x,
                min_chunk_z: chunk_region.min_chunk_z,
                max_chunk_x: chunk_region.max_chunk_x,
                max_chunk_z: chunk_region.max_chunk_z,
            },
            RenderRegionLoadOptions {
                surface: uses_surface,
                surface_subchunks: if uses_surface {
                    RenderSurfaceSubchunkMode::Full
                } else {
                    RenderSurfaceSubchunkMode::Needed
                },
                fixed_y,
                biome_y,
                load_all_biomes: loads_biome_plane,
                subchunk_decode: SubChunkDecodeMode::FullIndices,
            },
        )?;
        let chunk_width = u32::try_from(
            chunk_region
                .max_chunk_x
                .saturating_sub(chunk_region.min_chunk_x)
                .saturating_add(1),
        )
        .map_err(|_| BedrockRenderError::Validation("region width overflow".to_string()))?;
        let chunk_height = u32::try_from(
            chunk_region
                .max_chunk_z
                .saturating_sub(chunk_region.min_chunk_z)
                .saturating_add(1),
        )
        .map_err(|_| BedrockRenderError::Validation("region height overflow".to_string()))?;
        let width = chunk_width.saturating_mul(16);
        let height = chunk_height.saturating_mul(16);
        let mut payload =
            empty_region_payload(mode, width, height, self.palette.missing_chunk_color())?;
        let mut diagnostics = RenderDiagnostics::default();
        let base_region = coord.chunk_region(options.region_layout);
        for chunk_data in data.chunks {
            check_cancelled(options)?;
            let bake = self.bake_chunk_data(
                chunk_data,
                BakeOptions {
                    mode,
                    surface: options.surface,
                },
            )?;
            diagnostics.add(bake.diagnostics.clone());
            let relative_chunk_x = bake.pos.x.saturating_sub(base_region.min_chunk_x);
            let relative_chunk_z = bake.pos.z.saturating_sub(base_region.min_chunk_z);
            let Some(region_x) = u32::try_from(relative_chunk_x)
                .ok()
                .map(|x| x.saturating_mul(16))
            else {
                continue;
            };
            let Some(region_z) = u32::try_from(relative_chunk_z)
                .ok()
                .map(|z| z.saturating_mul(16))
            else {
                continue;
            };
            copy_chunk_bake_to_region(&bake, &mut payload, region_x, region_z);
        }
        Ok(RegionBake {
            coord,
            layout: options.region_layout,
            mode,
            chunk_region: base_region,
            payload,
            diagnostics,
        })
    }

    fn bake_region_chunk_positions_blocking(
        &self,
        coord: RegionCoord,
        chunk_region: ChunkRegion,
        chunk_positions: &[ChunkPos],
        options: &RenderOptions,
        mode: RenderMode,
    ) -> Result<RegionBake> {
        options.region_layout.validate()?;
        let fixed_y = match mode {
            RenderMode::LayerBlocks { y } | RenderMode::CaveSlice { y } => Some(y),
            _ => None,
        };
        let biome_y = match mode {
            RenderMode::Biome { y } | RenderMode::RawBiomeLayer { y } => Some(y),
            _ => Some(64),
        };
        let uses_surface = matches!(mode, RenderMode::SurfaceBlocks);
        let loads_biome_plane = matches!(
            mode,
            RenderMode::SurfaceBlocks | RenderMode::Biome { .. } | RenderMode::RawBiomeLayer { .. }
        );
        let data = self.world.load_render_chunks_blocking(
            chunk_positions.iter().copied(),
            RenderChunkLoadOptions {
                surface: uses_surface,
                surface_subchunks: if uses_surface {
                    RenderSurfaceSubchunkMode::Full
                } else {
                    RenderSurfaceSubchunkMode::Needed
                },
                fixed_y,
                biome_y,
                load_all_biomes: loads_biome_plane,
                subchunk_decode: SubChunkDecodeMode::FullIndices,
            },
        )?;
        self.bake_loaded_region_chunks(coord, chunk_region, data, options, mode)
    }

    fn bake_loaded_region_chunks(
        &self,
        coord: RegionCoord,
        _chunk_region: ChunkRegion,
        chunks: Vec<RenderChunkData>,
        options: &RenderOptions,
        mode: RenderMode,
    ) -> Result<RegionBake> {
        let width = options.region_layout.chunks_per_region.saturating_mul(16);
        let height = width;
        let mut payload =
            empty_region_payload(mode, width, height, self.palette.missing_chunk_color())?;
        let mut diagnostics = RenderDiagnostics::default();
        let base_region = coord.chunk_region(options.region_layout);
        for chunk_data in chunks {
            check_cancelled(options)?;
            let bake = self.bake_chunk_data(
                chunk_data,
                BakeOptions {
                    mode,
                    surface: options.surface,
                },
            )?;
            diagnostics.add(bake.diagnostics.clone());
            let relative_chunk_x = bake.pos.x.saturating_sub(base_region.min_chunk_x);
            let relative_chunk_z = bake.pos.z.saturating_sub(base_region.min_chunk_z);
            let Some(region_x) = u32::try_from(relative_chunk_x)
                .ok()
                .map(|x| x.saturating_mul(16))
            else {
                continue;
            };
            let Some(region_z) = u32::try_from(relative_chunk_z)
                .ok()
                .map(|z| z.saturating_mul(16))
            else {
                continue;
            };
            copy_chunk_bake_to_region(&bake, &mut payload, region_x, region_z);
        }
        Ok(RegionBake {
            coord,
            layout: options.region_layout,
            mode,
            chunk_region: base_region,
            payload,
            diagnostics,
        })
    }

    /// Renders planned tiles from region bakes and streams each tile to a sink.
    ///
    /// # Errors
    ///
    /// Returns an error if region baking, tile composition, encoding, cancellation,
    /// or the sink callback fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn render_tiles_from_regions_blocking<F>(
        &self,
        planned_tiles: &[PlannedTile],
        options: RenderOptions,
        sink: F,
    ) -> Result<RenderWebTilesResult>
    where
        F: Fn(PlannedTile, TileImage) -> Result<()> + Send + Sync,
    {
        self.render_web_regions_blocking(planned_tiles, &options, sink)
    }

    /// Renders web-map planned tiles and streams each tile to a sink.
    ///
    /// # Errors
    ///
    /// Returns an error if region baking, tile composition, encoding, cancellation,
    /// or the sink callback fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn render_web_tiles_blocking<F>(
        &self,
        planned_tiles: &[PlannedTile],
        options: RenderOptions,
        sink: F,
    ) -> Result<RenderWebTilesResult>
    where
        F: Fn(PlannedTile, TileImage) -> Result<()> + Send + Sync,
    {
        self.render_web_regions_blocking(planned_tiles, &options, sink)
    }

    fn render_web_regions_blocking<F>(
        &self,
        planned_tiles: &[PlannedTile],
        options: &RenderOptions,
        sink: F,
    ) -> Result<RenderWebTilesResult>
    where
        F: Fn(PlannedTile, TileImage) -> Result<()> + Send + Sync,
    {
        if planned_tiles.is_empty() {
            return Ok(RenderWebTilesResult {
                diagnostics: RenderDiagnostics::default(),
                stats: RenderPipelineStats::default(),
            });
        }
        options.region_layout.validate()?;
        let region_plans = collect_region_plans(planned_tiles, options.region_layout)?;
        let planned_chunk_count = region_plans.iter().fold(0usize, |total, plan| {
            total.saturating_add(plan.chunk_positions.len())
        });
        let mut stats = RenderPipelineStats {
            planned_tiles: planned_tiles.len(),
            planned_regions: region_plans.len(),
            unique_chunks: planned_chunk_count,
            ..RenderPipelineStats::default()
        };
        let region_plan_by_key = region_plans
            .iter()
            .map(|plan| (plan.key, plan.clone()))
            .collect::<BTreeMap<_, _>>();
        let tile_region_keys = planned_tiles
            .iter()
            .map(|planned| tile_region_keys(planned, options.region_layout))
            .collect::<Result<Vec<_>>>()?;
        let memory_budget = options
            .memory_budget
            .resolve_bytes(options.execution_profile)
            .unwrap_or(usize::MAX);
        let mut pending_tiles = (0..planned_tiles.len()).collect::<Vec<_>>();
        let mut diagnostics = RenderDiagnostics::default();
        let worker_count = options
            .threading
            .resolve_for_profile_checked(options.execution_profile, planned_tiles.len())?;
        stats.active_tasks_peak = 1;
        stats.peak_worker_threads = worker_count;

        while !pending_tiles.is_empty() {
            check_cancelled(options)?;
            let wave_keys = select_region_wave(
                &pending_tiles,
                &tile_region_keys,
                &region_plan_by_key,
                memory_budget,
            )?;
            let wave_plans = wave_keys
                .iter()
                .map(|key| region_plan_by_key.get(key).cloned())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    BedrockRenderError::Validation("planned tile references missing region".into())
                })?;
            let region_start = Instant::now();
            let regions = self.bake_web_regions_blocking(&wave_plans, options)?;
            let region_bake_ms = region_start.elapsed().as_millis();
            stats.region_bake_ms = stats.region_bake_ms.saturating_add(region_bake_ms);
            stats.bake_ms = stats.bake_ms.saturating_add(region_bake_ms);
            stats.baked_regions = stats.baked_regions.saturating_add(regions.len());
            stats.region_cache_misses = stats.region_cache_misses.saturating_add(regions.len());
            let wave_cache_bytes = wave_plans.iter().fold(0usize, |total, plan| {
                total.saturating_add(
                    plan.chunk_positions
                        .len()
                        .saturating_mul(REGION_BAKE_ESTIMATED_BYTES_PER_CHUNK),
                )
            });
            stats.peak_cache_bytes = stats.peak_cache_bytes.max(wave_cache_bytes);
            for bake in regions.values() {
                diagnostics.add(bake.diagnostics.clone());
            }
            stats.baked_chunks = diagnostics.baked_chunks;
            let render_tile_indexes = pending_tiles
                .iter()
                .copied()
                .filter(|index| {
                    tile_region_keys[*index]
                        .iter()
                        .all(|key| regions.contains_key(key))
                })
                .collect::<Vec<_>>();
            let compose_start = Instant::now();
            let compose_stats = render_web_tile_indexes(
                self,
                planned_tiles,
                &render_tile_indexes,
                options,
                &regions,
                worker_count,
                &sink,
            )?;
            stats.add_tile_compose_stats(compose_stats);
            let compose_ms = compose_start.elapsed().as_millis();
            stats.tile_compose_ms = stats.tile_compose_ms.saturating_add(compose_ms);
            stats.encode_ms = stats.encode_ms.saturating_add(compose_ms);
            let rendered = render_tile_indexes.into_iter().collect::<BTreeSet<_>>();
            pending_tiles.retain(|index| !rendered.contains(index));
        }
        Ok(RenderWebTilesResult { diagnostics, stats })
    }

    fn bake_web_regions_blocking(
        &self,
        plans: &[RegionPlan],
        options: &RenderOptions,
    ) -> Result<BTreeMap<RegionBakeKey, RegionBake>> {
        if plans.is_empty() {
            return Ok(BTreeMap::new());
        }
        let worker_count = options
            .threading
            .resolve_for_profile_checked(options.execution_profile, plans.len())?;
        if worker_count == 1 {
            let mut regions = BTreeMap::new();
            for plan in plans {
                check_cancelled(options)?;
                regions.insert(
                    plan.key,
                    self.bake_region_chunk_positions_blocking(
                        plan.key.coord,
                        plan.region,
                        &plan.chunk_positions,
                        options,
                        plan.key.mode,
                    )?,
                );
            }
            return Ok(regions);
        }
        let next_key = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::channel::<Result<(RegionBakeKey, RegionBake)>>();
        thread::scope(|scope| {
            for _ in 0..worker_count {
                let next_key = Arc::clone(&next_key);
                let sender = sender.clone();
                let renderer = self.clone();
                let options = options.clone();
                scope.spawn(move || {
                    loop {
                        if check_cancelled(&options).is_err() {
                            let send_result = sender.send(Err(BedrockRenderError::Cancelled));
                            drop(send_result);
                            return;
                        }
                        let index = next_key.fetch_add(1, Ordering::Relaxed);
                        if index >= plans.len() {
                            return;
                        }
                        let plan = plans[index].clone();
                        let result = renderer
                            .bake_region_chunk_positions_blocking(
                                plan.key.coord,
                                plan.region,
                                &plan.chunk_positions,
                                &options,
                                plan.key.mode,
                            )
                            .map(|bake| (plan.key, bake));
                        if sender.send(result).is_err() {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            let mut regions = BTreeMap::new();
            for message in receiver {
                let (key, region) = message?;
                regions.insert(key, region);
            }
            Ok(regions)
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn render_tile_from_cached_regions_with_stats(
        &self,
        job: RenderJob,
        options: &RenderOptions,
        regions: &BTreeMap<RegionBakeKey, RegionBake>,
    ) -> Result<(TileImage, TileComposeStats)> {
        validate_job(&job)?;
        check_cancelled(options)?;
        let pixel_count = usize::try_from(job.tile_size)
            .ok()
            .and_then(|size| size.checked_mul(size))
            .ok_or_else(|| {
                BedrockRenderError::Validation("tile pixel count overflow".to_string())
            })?;
        let request = RenderBackend::requested_for_options(options);
        let should_try_gpu =
            should_try_gpu_compose(request, pixel_count, job.mode, options.surface);
        let (rgba, diagnostics, compose_stats) = if should_try_gpu {
            let prepared = prepare_region_tile_compose(
                self.palette.missing_chunk_color(),
                &job,
                options,
                regions,
            )?;
            let input = GpuTileComposeInput {
                width: job.tile_size,
                height: job.tile_size,
                colors: &prepared.colors,
                heights: &prepared.heights,
                water_depths: &prepared.water_depths,
                lighting_enabled: prepared.lighting_enabled,
                lighting: options.surface.lighting,
            };
            match super::gpu::compose_tile(&input) {
                Ok(output) => {
                    let stats = TileComposeStats::gpu(&output);
                    (output.rgba, prepared.diagnostics, stats)
                }
                Err(message) => {
                    let mut diagnostics = prepared.diagnostics.clone();
                    diagnostics.gpu_fallbacks = diagnostics.gpu_fallbacks.saturating_add(1);
                    (
                        compose_region_tile_from_prepared(
                            &prepared,
                            job.tile_size,
                            options.surface,
                        ),
                        diagnostics,
                        TileComposeStats::cpu_fallback(message),
                    )
                }
            }
        } else {
            let (rgba, diagnostics) =
                compose_region_tile_cpu(self, &job, options, regions, pixel_count)?;
            (rgba, diagnostics, TileComposeStats::cpu())
        };
        if let Some(sink) = &options.diagnostics {
            sink.emit(diagnostics);
        }
        let encoded = encode_image(&rgba, job.tile_size, job.tile_size, options.format)?;
        Ok((
            TileImage {
                coord: job.coord,
                width: job.tile_size,
                height: job.tile_size,
                rgba,
                encoded,
            },
            compose_stats,
        ))
    }

    /// Bakes one chunk into reusable render payloads.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk cannot be loaded or decoded for the requested mode.
    pub fn bake_chunk_blocking(&self, pos: ChunkPos, options: BakeOptions) -> Result<ChunkBake> {
        let fixed_y = match options.mode {
            RenderMode::LayerBlocks { y } | RenderMode::CaveSlice { y } => Some(y),
            _ => None,
        };
        let biome_y = match options.mode {
            RenderMode::Biome { y } | RenderMode::RawBiomeLayer { y } => Some(y),
            _ => Some(64),
        };
        let uses_surface = matches!(options.mode, RenderMode::SurfaceBlocks);
        let loads_biome_plane = matches!(
            options.mode,
            RenderMode::SurfaceBlocks | RenderMode::Biome { .. } | RenderMode::RawBiomeLayer { .. }
        );
        let data = self.world.load_render_chunk_blocking(
            pos,
            RenderChunkLoadOptions {
                surface: uses_surface,
                surface_subchunks: if uses_surface {
                    RenderSurfaceSubchunkMode::Full
                } else {
                    RenderSurfaceSubchunkMode::Needed
                },
                fixed_y,
                biome_y,
                load_all_biomes: loads_biome_plane,
                subchunk_decode: SubChunkDecodeMode::FullIndices,
            },
        )?;
        self.bake_chunk_data(data, options)
    }

    fn bake_chunk_data(&self, data: RenderChunkData, options: BakeOptions) -> Result<ChunkBake> {
        let mode = options.mode;
        let pos = data.pos;
        let mut context = ChunkBakeContext {
            palette: &self.palette,
            options,
            data,
            diagnostics: RenderDiagnostics::default(),
        };
        let payload = context.bake_payload()?;
        let mut diagnostics = context.diagnostics;
        diagnostics.baked_chunks = diagnostics.baked_chunks.saturating_add(1);
        Ok(ChunkBake {
            pos,
            mode,
            payload,
            diagnostics,
        })
    }

    /// Renders a tile by baking the tile's chunks first.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is invalid, chunk loading or baking fails, rendering is
    /// cancelled, or image encoding fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn render_tile_from_bake_blocking(
        &self,
        job: RenderJob,
        options: &RenderOptions,
    ) -> Result<TileImage> {
        validate_job(&job)?;
        check_cancelled(options)?;
        let pixel_count = usize::try_from(job.tile_size)
            .ok()
            .and_then(|size| size.checked_mul(size))
            .ok_or_else(|| {
                BedrockRenderError::Validation("tile pixel count overflow".to_string())
            })?;
        let positions = tile_chunk_positions(&job)?;
        let bake_options = BakeOptions {
            mode: job.mode,
            surface: options.surface,
        };
        let bake_results = prefetch_parallel(positions, options, |pos| {
            self.bake_chunk_blocking(pos, bake_options.clone())
        })?;
        let mut diagnostics = RenderDiagnostics::default();
        let mut bakes = BTreeMap::new();
        for bake in bake_results {
            diagnostics.add(bake.diagnostics.clone());
            bakes.insert(bake.pos, bake);
        }

        let mut rgba = vec![0; pixel_count.saturating_mul(4)];
        let mut pixel_index = 0usize;
        for pixel_z in 0..job.tile_size {
            for pixel_x in 0..job.tile_size {
                if (pixel_count > 4096) && pixel_index.is_multiple_of(4096) {
                    check_cancelled(options)?;
                }
                let (block_x, block_z) = tile_pixel_to_block(&job, pixel_x, pixel_z)?;
                let block_pos = BlockPos {
                    x: block_x,
                    y: 0,
                    z: block_z,
                };
                let chunk_pos = block_pos.to_chunk_pos(job.coord.dimension);
                let (local_x, _, local_z) = block_pos.in_chunk_offset();
                let color = if let Some(color) = bakes
                    .get(&chunk_pos)
                    .and_then(|bake| chunk_bake_color(bake, u32::from(local_x), u32::from(local_z)))
                {
                    shade_chunk_bake_color(
                        color,
                        &bakes,
                        chunk_pos,
                        local_x,
                        local_z,
                        options.surface,
                    )
                } else {
                    diagnostics.missing_chunks = diagnostics.missing_chunks.saturating_add(1);
                    diagnostics.record_transparent_pixel();
                    self.palette.missing_chunk_color()
                };
                write_rgba_pixel(&mut rgba, pixel_index, color);
                pixel_index += 1;
            }
        }
        if let Some(sink) = &options.diagnostics {
            sink.emit(diagnostics);
        }
        let encoded = encode_image(&rgba, job.tile_size, job.tile_size, options.format)?;
        Ok(TileImage {
            coord: job.coord,
            width: job.tile_size,
            height: job.tile_size,
            rgba,
            encoded,
        })
    }

    #[cfg(feature = "async")]
    /// Asynchronously renders one tile by running the blocking renderer on a Tokio
    /// blocking task.
    ///
    /// # Errors
    ///
    /// Returns an error if the blocking task fails or if tile rendering fails.
    pub async fn render_tile(&self, job: RenderJob) -> Result<TileImage> {
        let renderer = self.clone();
        tokio::task::spawn_blocking(move || renderer.render_tile_blocking(job))
            .await
            .map_err(|error| BedrockRenderError::Join(error.to_string()))?
    }

    #[cfg(feature = "async")]
    /// Asynchronously renders a batch of tiles on a Tokio blocking task.
    ///
    /// # Errors
    ///
    /// Returns an error if the blocking task fails or if batch rendering fails.
    pub async fn render_tiles(
        &self,
        jobs: Vec<RenderJob>,
        options: RenderOptions,
    ) -> Result<Vec<TileImage>> {
        let renderer = self.clone();
        tokio::task::spawn_blocking(move || renderer.render_tiles_blocking(jobs, options))
            .await
            .map_err(|error| BedrockRenderError::Join(error.to_string()))?
    }
}

#[allow(dead_code)]
struct TileRenderContext<'a> {
    world: &'a BedrockWorld,
    palette: &'a RenderPalette,
    biome_cache: BTreeMap<(ChunkPos, i32), Option<bedrock_world::ParsedBiomeStorage>>,
    subchunk_cache: BTreeMap<(ChunkPos, i8), Option<SubChunk>>,
    height_cache: BTreeMap<ChunkPos, [[Option<i16>; 16]; 16]>,
    diagnostics: RenderDiagnostics,
    surface_options: SurfaceRenderOptions,
}

#[allow(dead_code)]
impl<'a> TileRenderContext<'a> {
    fn new(
        world: &'a BedrockWorld,
        palette: &'a RenderPalette,
        surface_options: SurfaceRenderOptions,
    ) -> Self {
        Self {
            world,
            palette,
            biome_cache: BTreeMap::new(),
            subchunk_cache: BTreeMap::new(),
            height_cache: BTreeMap::new(),
            diagnostics: RenderDiagnostics::default(),
            surface_options,
        }
    }

    fn diagnostics(&self) -> RenderDiagnostics {
        self.diagnostics.clone()
    }

    fn color_at(&mut self, job: &RenderJob, block_x: i32, block_z: i32) -> Result<RgbaColor> {
        match job.mode {
            RenderMode::Biome { y } => self.biome_color_at(job, block_x, block_z, y, false),
            RenderMode::RawBiomeLayer { y } => self.biome_color_at(job, block_x, block_z, y, true),
            RenderMode::LayerBlocks { y } => self.layer_block_color_at(job, block_x, block_z, y),
            RenderMode::SurfaceBlocks => self.surface_block_color_at(job, block_x, block_z),
            RenderMode::HeightMap => self.height_color_at(job, block_x, block_z),
            RenderMode::CaveSlice { y } => self.cave_color_at(job, block_x, block_z, y),
        }
    }

    fn prefetch_for_job(&mut self, job: &RenderJob, options: &RenderOptions) -> Result<()> {
        match job.mode {
            RenderMode::Biome { y } | RenderMode::RawBiomeLayer { y } => {
                self.prefetch_biomes(job, y, options)
            }
            RenderMode::LayerBlocks { y } | RenderMode::CaveSlice { y } => {
                self.prefetch_subchunks(job, y, options)
            }
            RenderMode::HeightMap | RenderMode::SurfaceBlocks => {
                self.prefetch_heights(job, options)
            }
        }
    }

    fn prefetch_biomes(&mut self, job: &RenderJob, y: i32, options: &RenderOptions) -> Result<()> {
        let bucket_y = biome_storage_bucket_y(y);
        let tasks = tile_chunk_positions(job)?
            .into_iter()
            .filter(|pos| !self.biome_cache.contains_key(&(*pos, bucket_y)))
            .collect::<Vec<_>>();
        let results = prefetch_parallel(tasks, options, |pos| {
            self.world
                .get_biome_storage_blocking(pos, y)
                .map(|storage| ((pos, bucket_y), storage))
                .map_err(BedrockRenderError::World)
        })?;
        for (key, storage) in results {
            self.biome_cache.insert(key, storage);
        }
        Ok(())
    }

    fn prefetch_heights(&mut self, job: &RenderJob, options: &RenderOptions) -> Result<()> {
        let tasks = tile_chunk_positions(job)?
            .into_iter()
            .filter(|pos| !self.height_cache.contains_key(pos))
            .collect::<Vec<_>>();
        let results = prefetch_parallel(tasks, options, |pos| {
            self.world
                .get_height_map_blocking(pos)
                .map(|heights| (pos, heights.unwrap_or([[None; 16]; 16])))
                .map_err(BedrockRenderError::World)
        })?;
        for (pos, heights) in results {
            self.height_cache.insert(pos, heights);
        }
        Ok(())
    }

    fn prefetch_subchunks(
        &mut self,
        job: &RenderJob,
        y: i32,
        options: &RenderOptions,
    ) -> Result<()> {
        let subchunk_y = block_y_to_subchunk_y(y)?;
        let tasks = tile_chunk_positions(job)?
            .into_iter()
            .filter(|pos| !self.subchunk_cache.contains_key(&(*pos, subchunk_y)))
            .collect::<Vec<_>>();
        let results = prefetch_parallel(tasks, options, |pos| {
            self.world
                .parse_subchunk_blocking(
                    pos,
                    subchunk_y,
                    bedrock_world::WorldParseOptions {
                        subchunk_decode_mode: SubChunkDecodeMode::FullIndices,
                        ..bedrock_world::WorldParseOptions::summary()
                    },
                )
                .map(|subchunk| ((pos, subchunk_y), subchunk))
                .map_err(BedrockRenderError::World)
        })?;
        for (key, subchunk) in results {
            self.subchunk_cache.insert(key, subchunk);
        }
        Ok(())
    }

    fn biome_color_at(
        &mut self,
        job: &RenderJob,
        block_x: i32,
        block_z: i32,
        y: i32,
        raw: bool,
    ) -> Result<RgbaColor> {
        let block_pos = BlockPos {
            x: block_x,
            y,
            z: block_z,
        };
        let chunk_pos = block_pos.to_chunk_pos(job.coord.dimension);
        let (local_x, _, local_z) = block_pos.in_chunk_offset();
        let storage_key = (chunk_pos, biome_storage_bucket_y(y));
        if !self.biome_cache.contains_key(&storage_key) {
            let storage = self.world.get_biome_storage_blocking(chunk_pos, y)?;
            self.biome_cache.insert(storage_key, storage);
        }
        let Some(Some(storage)) = self.biome_cache.get(&storage_key) else {
            return Ok(self.palette.unknown_biome_color());
        };
        Ok(storage
            .biome_id_at(local_x, local_biome_y(storage, y)?, local_z)
            .map_or_else(
                || self.palette.unknown_biome_color(),
                |id| {
                    if raw {
                        self.palette.raw_biome_color(id)
                    } else {
                        self.palette.biome_color(id)
                    }
                },
            ))
    }

    fn layer_block_color_at(
        &mut self,
        job: &RenderJob,
        block_x: i32,
        block_z: i32,
        y: i32,
    ) -> Result<RgbaColor> {
        let block_pos = BlockPos {
            x: block_x,
            y,
            z: block_z,
        };
        let chunk_pos = block_pos.to_chunk_pos(job.coord.dimension);
        let subchunk_y = block_y_to_subchunk_y(y)?;
        let block_name = self
            .cached_subchunk(chunk_pos, subchunk_y)?
            .and_then(|subchunk| block_name_at(subchunk, block_pos))
            .map(str::to_owned);
        let Some(name) = block_name else {
            self.diagnostics.record_transparent_pixel();
            return Ok(self.palette.missing_chunk_color());
        };
        if !self.palette.has_block_color(&name) {
            self.diagnostics.record_unknown_block(&name);
        }
        Ok(self.palette.block_color(&name))
    }

    fn cave_color_at(
        &mut self,
        job: &RenderJob,
        block_x: i32,
        block_z: i32,
        y: i32,
    ) -> Result<RgbaColor> {
        let block_pos = BlockPos {
            x: block_x,
            y,
            z: block_z,
        };
        let chunk_pos = block_pos.to_chunk_pos(job.coord.dimension);
        let subchunk_y = block_y_to_subchunk_y(y)?;
        let block_name = self
            .cached_subchunk(chunk_pos, subchunk_y)?
            .and_then(|subchunk| block_name_at(subchunk, block_pos))
            .map(str::to_owned);
        Ok(self.palette.cave_color(block_name.as_deref()))
    }

    fn height_color_at(
        &mut self,
        job: &RenderJob,
        block_x: i32,
        block_z: i32,
    ) -> Result<RgbaColor> {
        let block_pos = BlockPos {
            x: block_x,
            y: 0,
            z: block_z,
        };
        let chunk_pos = block_pos.to_chunk_pos(job.coord.dimension);
        let (local_x, _, local_z) = block_pos.in_chunk_offset();
        let height = self
            .cached_height(chunk_pos, local_x, local_z)?
            .unwrap_or(-64);
        Ok(self.palette.height_color(height, -64, 320))
    }

    fn surface_block_color_at(
        &mut self,
        job: &RenderJob,
        block_x: i32,
        block_z: i32,
    ) -> Result<RgbaColor> {
        let base_pos = BlockPos {
            x: block_x,
            y: 0,
            z: block_z,
        };
        let chunk_pos = base_pos.to_chunk_pos(job.coord.dimension);
        let (local_x, _, local_z) = base_pos.in_chunk_offset();
        let (min_y, max_y) = chunk_pos.y_range(bedrock_world::ChunkVersion::New);
        let Some(height) = self.cached_height(chunk_pos, local_x, local_z)? else {
            self.diagnostics.missing_heightmaps =
                self.diagnostics.missing_heightmaps.saturating_add(1);
            self.diagnostics.missing_chunks = self.diagnostics.missing_chunks.saturating_add(1);
            self.diagnostics.record_transparent_pixel();
            return Ok(self.palette.missing_chunk_color());
        };
        let start_y = i32::from(height).clamp(min_y, max_y);
        let mut visited_subchunk = false;
        if start_y < max_y {
            let (color, visited) = self.find_surface_block_color(
                chunk_pos,
                block_x,
                block_z,
                local_x,
                local_z,
                start_y.saturating_add(1),
                max_y,
                min_y,
            )?;
            visited_subchunk |= visited;
            if let Some(color) = color {
                return Ok(color);
            }
        }
        let (color, visited) = self.find_surface_block_color(
            chunk_pos, block_x, block_z, local_x, local_z, min_y, start_y, min_y,
        )?;
        visited_subchunk |= visited;
        if let Some(color) = color {
            return Ok(color);
        }
        if !visited_subchunk {
            self.diagnostics.missing_chunks = self.diagnostics.missing_chunks.saturating_add(1);
        }
        self.diagnostics.record_transparent_pixel();
        Ok(self.palette.missing_chunk_color())
    }

    #[allow(clippy::too_many_arguments)]
    fn find_surface_block_color(
        &mut self,
        chunk_pos: ChunkPos,
        block_x: i32,
        block_z: i32,
        local_x: u8,
        local_z: u8,
        min_y: i32,
        start_y: i32,
        world_min_y: i32,
    ) -> Result<(Option<RgbaColor>, bool)> {
        let mut visited_subchunk = false;
        for y in (min_y..=start_y).rev() {
            let subchunk_y = block_y_to_subchunk_y(y)?;
            let Some(subchunk) = self.cached_subchunk(chunk_pos, subchunk_y)? else {
                continue;
            };
            visited_subchunk = true;
            let block_pos = BlockPos {
                x: block_x,
                y,
                z: block_z,
            };
            let Some(name) = block_name_at(subchunk, block_pos).map(str::to_owned) else {
                continue;
            };
            if self.surface_options.skip_air && self.palette.is_air_block(&name) {
                continue;
            }
            let biome_id = self.cached_biome_id(chunk_pos, local_x, local_z, y)?;
            if !self.palette.has_block_color(&name) {
                self.diagnostics.record_unknown_block(&name);
                if !self.surface_options.render_unknown_blocks {
                    self.diagnostics.record_transparent_pixel();
                    return Ok((Some(self.palette.void_color()), visited_subchunk));
                }
            }
            let color = if self.palette.is_water_block(&name)
                && self.surface_options.transparent_water
            {
                let (depth, under_name) =
                    self.find_under_water_block(chunk_pos, block_x, block_z, y, world_min_y)?;
                self.palette.transparent_water_color(
                    &name,
                    under_name.as_deref(),
                    biome_id,
                    depth,
                    self.surface_options.biome_tint,
                )
            } else {
                self.palette
                    .surface_block_color(&name, biome_id, self.surface_options.biome_tint)
            };
            let shade_y = i16::try_from(y).map_err(|_| {
                BedrockRenderError::Validation(format!(
                    "surface block y={y} cannot be represented as i16"
                ))
            })?;
            return Ok((
                Some(self.surface_shaded_color(chunk_pos, local_x, local_z, shade_y, color)?),
                visited_subchunk,
            ));
        }
        Ok((None, visited_subchunk))
    }

    fn cached_height(
        &mut self,
        chunk_pos: ChunkPos,
        local_x: u8,
        local_z: u8,
    ) -> Result<Option<i16>> {
        if !self.height_cache.contains_key(&chunk_pos) {
            let heights = self
                .world
                .get_height_map_blocking(chunk_pos)?
                .unwrap_or([[None; 16]; 16]);
            self.height_cache.insert(chunk_pos, heights);
        }
        Ok(self
            .height_cache
            .get(&chunk_pos)
            .and_then(|heights| heights[usize::from(local_z)][usize::from(local_x)]))
    }

    fn cached_subchunk(&mut self, pos: ChunkPos, subchunk_y: i8) -> Result<Option<&SubChunk>> {
        if !self.subchunk_cache.contains_key(&(pos, subchunk_y)) {
            let subchunk = self.world.parse_subchunk_blocking(
                pos,
                subchunk_y,
                bedrock_world::WorldParseOptions {
                    subchunk_decode_mode: SubChunkDecodeMode::FullIndices,
                    ..bedrock_world::WorldParseOptions::summary()
                },
            )?;
            self.subchunk_cache.insert((pos, subchunk_y), subchunk);
        }
        Ok(self
            .subchunk_cache
            .get(&(pos, subchunk_y))
            .and_then(Option::as_ref))
    }

    fn cached_biome_id(
        &mut self,
        chunk_pos: ChunkPos,
        local_x: u8,
        local_z: u8,
        y: i32,
    ) -> Result<Option<u32>> {
        let storage_key = (chunk_pos, biome_storage_bucket_y(y));
        if !self.biome_cache.contains_key(&storage_key) {
            let storage = self.world.get_biome_storage_blocking(chunk_pos, y)?;
            self.biome_cache.insert(storage_key, storage);
        }
        let Some(Some(storage)) = self.biome_cache.get(&storage_key) else {
            return Ok(None);
        };
        Ok(storage.biome_id_at(local_x, local_biome_y(storage, y)?, local_z))
    }

    fn find_under_water_block(
        &mut self,
        chunk_pos: ChunkPos,
        block_x: i32,
        block_z: i32,
        water_y: i32,
        min_y: i32,
    ) -> Result<(u8, Option<String>)> {
        let mut depth = 0_u8;
        for y in (min_y..water_y).rev() {
            let subchunk_y = block_y_to_subchunk_y(y)?;
            let Some(subchunk) = self.cached_subchunk(chunk_pos, subchunk_y)? else {
                continue;
            };
            let block_pos = BlockPos {
                x: block_x,
                y,
                z: block_z,
            };
            let Some(name) = block_name_at(subchunk, block_pos).map(str::to_owned) else {
                continue;
            };
            if self.palette.is_air_block(&name) || self.palette.is_water_block(&name) {
                depth = depth.saturating_add(1);
                continue;
            }
            depth = depth.saturating_add(1);
            return Ok((depth, Some(name)));
        }
        Ok((depth, None))
    }

    fn surface_shaded_color(
        &mut self,
        chunk_pos: ChunkPos,
        local_x: u8,
        local_z: u8,
        height: i16,
        color: RgbaColor,
    ) -> Result<RgbaColor> {
        if !self.surface_options.height_shading {
            return Ok(color);
        }
        let west_x = local_x.saturating_sub(1);
        let east_x = local_x.saturating_add(1).min(15);
        let north_z = local_z.saturating_sub(1);
        let south_z = local_z.saturating_add(1).min(15);
        let west = self
            .cached_height(chunk_pos, west_x, local_z)?
            .unwrap_or(height);
        let east = self
            .cached_height(chunk_pos, east_x, local_z)?
            .unwrap_or(height);
        let north = self
            .cached_height(chunk_pos, local_x, north_z)?
            .unwrap_or(height);
        let south = self
            .cached_height(chunk_pos, local_x, south_z)?
            .unwrap_or(height);
        Ok(self
            .palette
            .height_normal_shaded_color(color, west, east, north, south))
    }
}

struct ChunkBakeContext<'a> {
    palette: &'a RenderPalette,
    options: BakeOptions,
    data: RenderChunkData,
    diagnostics: RenderDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct SurfaceSample {
    color: RgbaColor,
    height: Option<i16>,
    underwater_height: Option<i16>,
    water_depth: u8,
    is_water: bool,
}

#[derive(Debug, Clone)]
struct UnderWaterBlock {
    name: String,
    height: i16,
}

impl ChunkBakeContext<'_> {
    fn bake_payload(&mut self) -> Result<ChunkBakePayload> {
        match self.options.mode {
            RenderMode::SurfaceBlocks => self.bake_surface_payload(),
            RenderMode::HeightMap => Ok(self.bake_heightmap_payload()),
            RenderMode::Biome { .. }
            | RenderMode::RawBiomeLayer { .. }
            | RenderMode::LayerBlocks { .. }
            | RenderMode::CaveSlice { .. } => self.bake_color_payload(),
        }
    }

    fn bake_color_payload(&mut self) -> Result<ChunkBakePayload> {
        let mut colors = RgbaPlane::new(16, 16, self.palette.missing_chunk_color())?;
        for local_z in 0..16_u8 {
            for local_x in 0..16_u8 {
                colors.set_color(
                    u32::from(local_x),
                    u32::from(local_z),
                    self.color_at(local_x, local_z),
                );
            }
        }
        Ok(ChunkBakePayload::Colors(colors))
    }

    fn bake_surface_payload(&mut self) -> Result<ChunkBakePayload> {
        let mut colors = RgbaPlane::new(16, 16, self.palette.missing_chunk_color())?;
        let mut heights = HeightPlane::new(16, 16)?;
        let mut relief_heights = HeightPlane::new(16, 16)?;
        let mut water_depths = DepthPlane::new(16, 16)?;
        for local_z in 0..16_u8 {
            for local_x in 0..16_u8 {
                let sample = self.surface_sample_at(local_x, local_z);
                colors.set_color(u32::from(local_x), u32::from(local_z), sample.color);
                if let Some(height) = sample.height {
                    heights.set_height(u32::from(local_x), u32::from(local_z), height);
                    relief_heights.set_height(
                        u32::from(local_x),
                        u32::from(local_z),
                        sample.underwater_height.unwrap_or(height),
                    );
                }
                if sample.is_water {
                    water_depths.set_depth(
                        u32::from(local_x),
                        u32::from(local_z),
                        sample.water_depth,
                    );
                }
            }
        }
        Ok(ChunkBakePayload::Surface(SurfacePlane {
            colors,
            heights,
            relief_heights,
            water_depths,
        }))
    }

    fn bake_heightmap_payload(&self) -> ChunkBakePayload {
        let mut colors = RgbaPlane {
            width: 16,
            height: 16,
            pixels: Vec::with_capacity(256),
        };
        let mut heights = HeightPlane {
            width: 16,
            height: 16,
            heights: Vec::with_capacity(256),
        };
        for local_z in 0..16_u8 {
            for local_x in 0..16_u8 {
                let height = self
                    .data
                    .height_map
                    .as_ref()
                    .and_then(|height_map| height_map[usize::from(local_z)][usize::from(local_x)])
                    .unwrap_or(-64);
                colors
                    .pixels
                    .push(self.palette.height_color(height, -64, 320));
                heights.heights.push(height);
            }
        }
        ChunkBakePayload::HeightMap { colors, heights }
    }

    fn color_at(&mut self, local_x: u8, local_z: u8) -> RgbaColor {
        if !self.data.is_loaded {
            self.diagnostics.missing_chunks = self.diagnostics.missing_chunks.saturating_add(1);
            self.diagnostics.record_transparent_pixel();
            return self.palette.missing_chunk_color();
        }
        match self.options.mode {
            RenderMode::SurfaceBlocks => self.surface_sample_at(local_x, local_z).color,
            RenderMode::HeightMap => self.height_color_at(local_x, local_z),
            RenderMode::Biome { y } => self.biome_color_at(local_x, local_z, y, false),
            RenderMode::RawBiomeLayer { y } => self.biome_color_at(local_x, local_z, y, true),
            RenderMode::LayerBlocks { y } => self.layer_color_at(local_x, local_z, y),
            RenderMode::CaveSlice { y } => self.cave_color_at(local_x, local_z, y),
        }
    }

    fn surface_sample_at(&mut self, local_x: u8, local_z: u8) -> SurfaceSample {
        if !self.data.is_loaded {
            self.diagnostics.missing_chunks = self.diagnostics.missing_chunks.saturating_add(1);
            self.diagnostics.record_transparent_pixel();
            return SurfaceSample {
                color: self.palette.missing_chunk_color(),
                height: None,
                underwater_height: None,
                water_depth: 0,
                is_water: false,
            };
        }
        let Some(height) = self
            .data
            .height_map
            .as_ref()
            .and_then(|heights| heights[usize::from(local_z)][usize::from(local_x)])
        else {
            self.diagnostics.missing_heightmaps =
                self.diagnostics.missing_heightmaps.saturating_add(1);
            self.diagnostics.missing_chunks = self.diagnostics.missing_chunks.saturating_add(1);
            self.diagnostics.record_transparent_pixel();
            return SurfaceSample {
                color: self.palette.missing_chunk_color(),
                height: None,
                underwater_height: None,
                water_depth: 0,
                is_water: false,
            };
        };

        let (min_y, max_y) = self.data.pos.y_range(bedrock_world::ChunkVersion::New);
        if i32::from(height) < max_y
            && let Some(sample) =
                self.find_surface_sample(local_x, local_z, i32::from(height) + 1, max_y)
        {
            return sample;
        }
        if let Some(sample) = self.find_surface_sample(local_x, local_z, min_y, i32::from(height)) {
            return sample;
        }

        self.diagnostics.record_transparent_pixel();
        SurfaceSample {
            color: self.palette.missing_chunk_color(),
            height: None,
            underwater_height: None,
            water_depth: 0,
            is_water: false,
        }
    }

    fn find_surface_sample(
        &mut self,
        local_x: u8,
        local_z: u8,
        min_y: i32,
        start_y: i32,
    ) -> Option<SurfaceSample> {
        let (_, max_y) = self.data.pos.y_range(bedrock_world::ChunkVersion::New);
        for y in (min_y..=start_y.clamp(min_y, max_y)).rev() {
            let Some(name) = self.block_name_at(local_x, y, local_z) else {
                continue;
            };
            if self.options.surface.skip_air && self.palette.is_air_block(name) {
                continue;
            }
            let name = name.to_string();
            let biome_id = self.biome_id_at_or_top(local_x, local_z, y);
            if !self.palette.has_block_color(&name) {
                self.diagnostics.record_unknown_block(&name);
                if !self.options.surface.render_unknown_blocks {
                    self.diagnostics.record_transparent_pixel();
                    return Some(SurfaceSample {
                        color: self.palette.void_color(),
                        height: None,
                        underwater_height: None,
                        water_depth: 0,
                        is_water: false,
                    });
                }
            }
            let is_water = self.palette.is_water_block(&name);
            let (water_depth, underwater) = if is_water && self.options.surface.transparent_water {
                self.find_under_water_block(local_x, local_z, y, min_y)
            } else {
                (0, None)
            };
            let color = if is_water && self.options.surface.transparent_water {
                self.palette.transparent_water_color(
                    &name,
                    underwater.as_ref().map(|block| block.name.as_str()),
                    biome_id,
                    water_depth,
                    self.options.surface.biome_tint,
                )
            } else {
                self.palette
                    .surface_block_color(&name, biome_id, self.options.surface.biome_tint)
            };
            let shade_y = i16::try_from(y).unwrap_or(if y < 0 { i16::MIN } else { i16::MAX });
            return Some(SurfaceSample {
                color,
                height: Some(shade_y),
                underwater_height: underwater.as_ref().map(|block| block.height),
                water_depth,
                is_water,
            });
        }
        None
    }

    fn height_color_at(&self, local_x: u8, local_z: u8) -> RgbaColor {
        let height = self
            .data
            .height_map
            .as_ref()
            .and_then(|heights| heights[usize::from(local_z)][usize::from(local_x)])
            .unwrap_or(-64);
        self.palette.height_color(height, -64, 320)
    }

    fn biome_color_at(&self, local_x: u8, local_z: u8, y: i32, raw: bool) -> RgbaColor {
        let Some(id) = self.biome_id_at_or_top(local_x, local_z, y) else {
            return self.palette.unknown_biome_color();
        };
        if raw {
            self.palette.raw_biome_color(id)
        } else {
            self.palette.biome_color(id)
        }
    }

    fn layer_color_at(&mut self, local_x: u8, local_z: u8, y: i32) -> RgbaColor {
        let Some(name) = self.block_name_at(local_x, y, local_z).map(str::to_string) else {
            self.diagnostics.record_transparent_pixel();
            return self.palette.missing_chunk_color();
        };
        if !self.palette.has_block_color(&name) {
            self.diagnostics.record_unknown_block(&name);
        }
        self.palette.block_color(&name)
    }

    fn cave_color_at(&self, local_x: u8, local_z: u8, y: i32) -> RgbaColor {
        self.palette
            .cave_color(self.block_name_at(local_x, y, local_z))
    }

    fn block_name_at(&self, local_x: u8, y: i32, local_z: u8) -> Option<&str> {
        let subchunk_y = block_y_to_subchunk_y(y).ok()?;
        let subchunk = self.data.subchunks.get(&subchunk_y)?;
        let local_y = u8::try_from(y - i32::from(subchunk_y) * 16).ok()?;
        subchunk
            .block_state_at(local_x, local_y, local_z)
            .map(|state| state.name.as_str())
    }

    fn biome_id_at(&self, local_x: u8, local_z: u8, y: i32) -> Option<u32> {
        let storage = self
            .data
            .biome_data
            .get(&biome_storage_bucket_y(y))
            .or_else(|| self.data.biome_data.values().next())?;
        non_empty_biome_id(storage.biome_id_at(local_x, local_biome_y(storage, y).ok()?, local_z))
    }

    fn biome_id_at_or_top(&self, local_x: u8, local_z: u8, y: i32) -> Option<u32> {
        if let Some(id) = self.biome_id_at(local_x, local_z, y) {
            return Some(id);
        }
        self.top_biome_id_at(local_x, local_z)
    }

    fn top_biome_id_at(&self, local_x: u8, local_z: u8) -> Option<u32> {
        for storage in self.data.biome_data.values().rev() {
            if storage.y.is_none() {
                if let Some(id) = non_empty_biome_id(storage.biome_id_at(local_x, 0, local_z)) {
                    return Some(id);
                }
                continue;
            }
            for local_y in (0..16_u8).rev() {
                if let Some(id) = non_empty_biome_id(storage.biome_id_at(local_x, local_y, local_z))
                {
                    return Some(id);
                }
            }
        }
        None
    }

    fn find_under_water_block(
        &self,
        local_x: u8,
        local_z: u8,
        water_y: i32,
        min_y: i32,
    ) -> (u8, Option<UnderWaterBlock>) {
        let mut depth = 0_u8;
        for y in (min_y..water_y).rev() {
            let Some(name) = self.block_name_at(local_x, y, local_z) else {
                continue;
            };
            if self.palette.is_air_block(name) || self.palette.is_water_block(name) {
                depth = depth.saturating_add(1);
                continue;
            }
            depth = depth.saturating_add(1);
            let height = i16::try_from(y).unwrap_or(if y < 0 { i16::MIN } else { i16::MAX });
            return (
                depth,
                Some(UnderWaterBlock {
                    name: name.to_string(),
                    height,
                }),
            );
        }
        (depth, None)
    }
}

fn should_try_gpu_compose(
    request: RenderBackend,
    pixel_count: usize,
    mode: RenderMode,
    surface: SurfaceRenderOptions,
) -> bool {
    match request {
        RenderBackend::Cpu => false,
        RenderBackend::Gpu => true,
        RenderBackend::Auto => {
            super::gpu::feature_enabled()
                && pixel_count >= GPU_COMPOSE_MIN_PIXELS
                && lighting_enabled_for(mode, surface)
        }
    }
}

fn compose_region_tile_cpu(
    renderer: &MapRenderer,
    job: &RenderJob,
    options: &RenderOptions,
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    pixel_count: usize,
) -> Result<(Vec<u8>, RenderDiagnostics)> {
    let mut diagnostics = RenderDiagnostics::default();
    let mut rgba = vec![0; pixel_count.saturating_mul(4)];
    let mut pixel_index = 0usize;
    for pixel_z in 0..job.tile_size {
        for pixel_x in 0..job.tile_size {
            if (pixel_count > 4096) && pixel_index.is_multiple_of(4096) {
                check_cancelled(options)?;
            }
            let (block_x, block_z) = tile_pixel_to_block(job, pixel_x, pixel_z)?;
            let color = if let Some(color) = region_color_at_block(
                regions,
                options.region_layout,
                job.coord.dimension,
                job.mode,
                block_x,
                block_z,
            ) {
                shade_region_color(
                    color,
                    regions,
                    options.region_layout,
                    job.coord.dimension,
                    job.mode,
                    block_x,
                    block_z,
                    options.surface,
                )
            } else {
                diagnostics.missing_chunks = diagnostics.missing_chunks.saturating_add(1);
                diagnostics.record_transparent_pixel();
                renderer.palette.missing_chunk_color()
            };
            write_rgba_pixel(&mut rgba, pixel_index, color);
            pixel_index += 1;
        }
    }
    Ok((rgba, diagnostics))
}

fn prepare_region_tile_compose(
    missing_color: RgbaColor,
    job: &RenderJob,
    options: &RenderOptions,
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
) -> Result<PreparedTileCompose> {
    let pixel_count = usize::try_from(job.tile_size)
        .ok()
        .and_then(|size| size.checked_mul(size))
        .ok_or_else(|| BedrockRenderError::Validation("tile pixel count overflow".to_string()))?;
    let lighting_enabled = lighting_enabled_for(job.mode, options.surface);
    let mut diagnostics = RenderDiagnostics::default();
    let mut colors = Vec::with_capacity(pixel_count);
    let mut water_depths = Vec::with_capacity(pixel_count);
    let mut heights = Vec::with_capacity(pixel_count.saturating_mul(9));

    for pixel_z in 0..job.tile_size {
        for pixel_x in 0..job.tile_size {
            let pixel_index = colors.len();
            if (pixel_count > 4096) && pixel_index.is_multiple_of(4096) {
                check_cancelled(options)?;
            }
            let (block_x, block_z) = tile_pixel_to_block(job, pixel_x, pixel_z)?;
            if let Some(color) = region_color_at_block(
                regions,
                options.region_layout,
                job.coord.dimension,
                job.mode,
                block_x,
                block_z,
            ) {
                colors.push(pack_rgba_color(color));
                water_depths.push(u32::from(region_water_depth_at_block(
                    regions,
                    options.region_layout,
                    job.coord.dimension,
                    job.mode,
                    block_x,
                    block_z,
                )));
                push_region_height_neighborhood(
                    &mut heights,
                    regions,
                    options.region_layout,
                    job.coord.dimension,
                    job.mode,
                    block_x,
                    block_z,
                    lighting_enabled,
                );
            } else {
                diagnostics.missing_chunks = diagnostics.missing_chunks.saturating_add(1);
                diagnostics.record_transparent_pixel();
                colors.push(pack_rgba_color(missing_color));
                water_depths.push(0);
                push_missing_height_neighborhood(&mut heights);
            }
        }
    }

    Ok(PreparedTileCompose {
        colors,
        heights,
        water_depths,
        diagnostics,
        lighting_enabled,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_region_height_neighborhood(
    heights: &mut Vec<i32>,
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
    lighting_enabled: bool,
) {
    if !lighting_enabled {
        push_missing_height_neighborhood(heights);
        return;
    }
    let Some(center) =
        region_height_at_block(regions, region_layout, dimension, mode, block_x, block_z)
    else {
        push_missing_height_neighborhood(heights);
        return;
    };
    let height_at = |x, z| {
        i32::from(
            region_height_at_block(regions, region_layout, dimension, mode, x, z).unwrap_or(center),
        )
    };
    heights.extend_from_slice(&[
        i32::from(center),
        height_at(block_x - 1, block_z - 1),
        height_at(block_x, block_z - 1),
        height_at(block_x + 1, block_z - 1),
        height_at(block_x - 1, block_z),
        height_at(block_x + 1, block_z),
        height_at(block_x - 1, block_z + 1),
        height_at(block_x, block_z + 1),
        height_at(block_x + 1, block_z + 1),
    ]);
}

fn push_missing_height_neighborhood(heights: &mut Vec<i32>) {
    heights.extend_from_slice(&[i32::from(MISSING_HEIGHT); 9]);
}

fn compose_region_tile_from_prepared(
    prepared: &PreparedTileCompose,
    tile_size: u32,
    surface: SurfaceRenderOptions,
) -> Vec<u8> {
    let pixel_count = usize::try_from(tile_size)
        .ok()
        .and_then(|size| size.checked_mul(size))
        .unwrap_or(0);
    let mut rgba = vec![0; pixel_count.saturating_mul(4)];
    for (pixel_index, packed_color) in prepared.colors.iter().copied().enumerate() {
        let mut color = unpack_rgba_color(packed_color);
        if prepared.lighting_enabled {
            let offset = pixel_index.saturating_mul(9);
            if let Some(heights) = prepared.heights.get(offset..offset.saturating_add(9))
                && heights.first().copied() != Some(i32::from(MISSING_HEIGHT))
            {
                let neighborhood = TerrainHeightNeighborhood {
                    center: i16_from_i32(heights[0]),
                    north_west: i16_from_i32(heights[1]),
                    north: i16_from_i32(heights[2]),
                    north_east: i16_from_i32(heights[3]),
                    west: i16_from_i32(heights[4]),
                    east: i16_from_i32(heights[5]),
                    south_west: i16_from_i32(heights[6]),
                    south: i16_from_i32(heights[7]),
                    south_east: i16_from_i32(heights[8]),
                };
                let water_depth = u8_from_u32(
                    prepared
                        .water_depths
                        .get(pixel_index)
                        .copied()
                        .unwrap_or(0)
                        .min(u32::from(u8::MAX)),
                );
                color = terrain_lit_color(color, neighborhood, water_depth, surface.lighting);
            }
        }
        write_rgba_pixel(&mut rgba, pixel_index, color);
    }
    rgba
}

fn pack_rgba_color(color: RgbaColor) -> u32 {
    u32::from(color.red)
        | (u32::from(color.green) << 8)
        | (u32::from(color.blue) << 16)
        | (u32::from(color.alpha) << 24)
}

fn unpack_rgba_color(color: u32) -> RgbaColor {
    RgbaColor::new(
        u8_from_u32(color & 0xff),
        u8_from_u32((color >> 8) & 0xff),
        u8_from_u32((color >> 16) & 0xff),
        u8_from_u32((color >> 24) & 0xff),
    )
}

fn validate_job(job: &RenderJob) -> Result<()> {
    if job.tile_size == 0 {
        return Err(BedrockRenderError::Validation(
            "tile_size must be greater than zero".to_string(),
        ));
    }
    if job.scale == 0 {
        return Err(BedrockRenderError::Validation(
            "scale must be greater than zero".to_string(),
        ));
    }
    if job.pixels_per_block == 0 {
        return Err(BedrockRenderError::Validation(
            "pixels_per_block must be greater than zero".to_string(),
        ));
    }
    if job.tile_size > MAX_TILE_SIZE_PIXELS {
        return Err(BedrockRenderError::Validation(format!(
            "tile_size must be <= {MAX_TILE_SIZE_PIXELS}"
        )));
    }
    let tile_scaled_pixels = job
        .tile_size
        .checked_mul(job.scale)
        .ok_or_else(|| BedrockRenderError::Validation("tile block span overflow".to_string()))?;
    if tile_scaled_pixels % job.pixels_per_block != 0 {
        return Err(BedrockRenderError::Validation(
            "tile_size * scale must be divisible by pixels_per_block".to_string(),
        ));
    }
    Ok(())
}

fn validate_layout(layout: ChunkTileLayout) -> Result<()> {
    if layout.chunks_per_tile == 0 {
        return Err(BedrockRenderError::Validation(
            "chunks_per_tile must be greater than zero".to_string(),
        ));
    }
    if layout.blocks_per_pixel == 0 {
        return Err(BedrockRenderError::Validation(
            "blocks_per_pixel must be greater than zero".to_string(),
        ));
    }
    if layout.pixels_per_block == 0 {
        return Err(BedrockRenderError::Validation(
            "pixels_per_block must be greater than zero".to_string(),
        ));
    }
    if layout.tile_size().is_none() {
        return Err(BedrockRenderError::Validation(format!(
            "chunks_per_tile * 16 * pixels_per_block must be divisible by blocks_per_pixel and produce a tile <= {MAX_TILE_SIZE_PIXELS}px"
        )));
    }
    Ok(())
}

fn validate_region(region: ChunkRegion) -> Result<()> {
    if region.min_chunk_x > region.max_chunk_x || region.min_chunk_z > region.max_chunk_z {
        return Err(BedrockRenderError::Validation(format!(
            "invalid chunk region: min=({}, {}) max=({}, {})",
            region.min_chunk_x, region.min_chunk_z, region.max_chunk_x, region.max_chunk_z
        )));
    }
    Ok(())
}

fn floor_div(value: i32, divisor: i32) -> i32 {
    value.div_euclid(divisor)
}

fn dimension_slug(dimension: Dimension) -> String {
    match dimension {
        Dimension::Overworld => "overworld".to_string(),
        Dimension::Nether => "nether".to_string(),
        Dimension::End => "end".to_string(),
        Dimension::Unknown(id) => format!("dimension-{id}"),
    }
}

fn mode_slug(mode: RenderMode) -> String {
    match mode {
        RenderMode::Biome { y } => format!("biome-y{y}"),
        RenderMode::RawBiomeLayer { y } => format!("raw-biome-y{y}"),
        RenderMode::SurfaceBlocks => "surface".to_string(),
        RenderMode::LayerBlocks { y } => format!("layer-y{y}"),
        RenderMode::HeightMap => "heightmap".to_string(),
        RenderMode::CaveSlice { y } => format!("cave-y{y}"),
    }
}

fn tile_pixel_to_block(job: &RenderJob, pixel_x: u32, pixel_z: u32) -> Result<(i32, i32)> {
    let tile_span =
        i64::from(job.tile_size) * i64::from(job.scale) / i64::from(job.pixels_per_block);
    let pixel_block_x = i64::from(pixel_x)
        .checked_mul(i64::from(job.scale))
        .and_then(|value| value.checked_div(i64::from(job.pixels_per_block)))
        .ok_or_else(|| BedrockRenderError::Validation("tile x coordinate overflow".to_string()))?;
    let pixel_block_z = i64::from(pixel_z)
        .checked_mul(i64::from(job.scale))
        .and_then(|value| value.checked_div(i64::from(job.pixels_per_block)))
        .ok_or_else(|| BedrockRenderError::Validation("tile z coordinate overflow".to_string()))?;
    let block_x = i64::from(job.coord.x)
        .checked_mul(tile_span)
        .and_then(|value| value.checked_add(pixel_block_x))
        .ok_or_else(|| BedrockRenderError::Validation("tile x coordinate overflow".to_string()))?;
    let block_z = i64::from(job.coord.z)
        .checked_mul(tile_span)
        .and_then(|value| value.checked_add(pixel_block_z))
        .ok_or_else(|| BedrockRenderError::Validation("tile z coordinate overflow".to_string()))?;
    let block_x = i32::try_from(block_x).map_err(|_| {
        BedrockRenderError::Validation("tile x coordinate is outside i32 range".to_string())
    })?;
    let block_z = i32::try_from(block_z).map_err(|_| {
        BedrockRenderError::Validation("tile z coordinate is outside i32 range".to_string())
    })?;
    Ok((block_x, block_z))
}

fn write_rgba_pixel(rgba: &mut [u8], pixel_index: usize, color: RgbaColor) {
    let Some(offset) = pixel_index.checked_mul(4) else {
        return;
    };
    let Some(pixel) = rgba.get_mut(offset..offset.saturating_add(4)) else {
        return;
    };
    pixel[0] = color.red;
    pixel[1] = color.green;
    pixel[2] = color.blue;
    pixel[3] = color.alpha;
}

fn lighting_enabled_for(mode: RenderMode, surface: SurfaceRenderOptions) -> bool {
    surface.height_shading
        && surface.lighting.enabled
        && matches!(mode, RenderMode::SurfaceBlocks | RenderMode::HeightMap)
}

#[allow(clippy::too_many_arguments)]
fn shade_region_color(
    color: RgbaColor,
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
    surface: SurfaceRenderOptions,
) -> RgbaColor {
    if !lighting_enabled_for(mode, surface) {
        return color;
    }
    let Some(height) =
        region_height_at_block(regions, region_layout, dimension, mode, block_x, block_z)
    else {
        return color;
    };
    let heights = region_height_neighborhood_at_block(
        regions,
        region_layout,
        dimension,
        mode,
        block_x,
        block_z,
        height,
    );
    let water_depth =
        region_water_depth_at_block(regions, region_layout, dimension, mode, block_x, block_z);
    terrain_lit_color(color, heights, water_depth, surface.lighting)
}

fn shade_chunk_bake_color(
    color: RgbaColor,
    bakes: &BTreeMap<ChunkPos, ChunkBake>,
    chunk_pos: ChunkPos,
    local_x: u8,
    local_z: u8,
    surface: SurfaceRenderOptions,
) -> RgbaColor {
    let Some(bake) = bakes.get(&chunk_pos) else {
        return color;
    };
    if !lighting_enabled_for(bake.mode, surface) {
        return color;
    }
    let Some(height) = chunk_bake_relief_height(bake, u32::from(local_x), u32::from(local_z))
    else {
        return color;
    };
    let block_x = chunk_pos.x.saturating_mul(16) + i32::from(local_x);
    let block_z = chunk_pos.z.saturating_mul(16) + i32::from(local_z);
    let heights = chunk_bake_height_neighborhood_at_block(
        bakes,
        chunk_pos.dimension,
        bake.mode,
        block_x,
        block_z,
        height,
    );
    let water_depth = chunk_bake_water_depth(bake, u32::from(local_x), u32::from(local_z));
    terrain_lit_color(color, heights, water_depth, surface.lighting)
}

fn region_color_at_block(
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
) -> Option<RgbaColor> {
    let block_pos = BlockPos {
        x: block_x,
        y: 0,
        z: block_z,
    };
    let chunk_pos = block_pos.to_chunk_pos(dimension);
    let (local_x, _, local_z) = block_pos.in_chunk_offset();
    let region_key = RegionBakeKey {
        coord: RegionCoord::from_chunk(chunk_pos, region_layout),
        mode,
    };
    regions
        .get(&region_key)
        .and_then(|region| region.color_at_chunk_local(chunk_pos, local_x, local_z))
}

fn region_height_at_block(
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
) -> Option<i16> {
    let block_pos = BlockPos {
        x: block_x,
        y: 0,
        z: block_z,
    };
    let chunk_pos = block_pos.to_chunk_pos(dimension);
    let (local_x, _, local_z) = block_pos.in_chunk_offset();
    let region_key = RegionBakeKey {
        coord: RegionCoord::from_chunk(chunk_pos, region_layout),
        mode,
    };
    regions
        .get(&region_key)
        .and_then(|region| region.height_at_chunk_local(chunk_pos, local_x, local_z))
}

fn region_height_neighborhood_at_block(
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
    fallback: i16,
) -> TerrainHeightNeighborhood {
    let height_at = |x, z| {
        region_height_at_block(regions, region_layout, dimension, mode, x, z).unwrap_or(fallback)
    };
    TerrainHeightNeighborhood {
        center: fallback,
        north_west: height_at(block_x - 1, block_z - 1),
        north: height_at(block_x, block_z - 1),
        north_east: height_at(block_x + 1, block_z - 1),
        west: height_at(block_x - 1, block_z),
        east: height_at(block_x + 1, block_z),
        south_west: height_at(block_x - 1, block_z + 1),
        south: height_at(block_x, block_z + 1),
        south_east: height_at(block_x + 1, block_z + 1),
    }
}

fn region_water_depth_at_block(
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    region_layout: RegionLayout,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
) -> u8 {
    let block_pos = BlockPos {
        x: block_x,
        y: 0,
        z: block_z,
    };
    let chunk_pos = block_pos.to_chunk_pos(dimension);
    let (local_x, _, local_z) = block_pos.in_chunk_offset();
    let region_key = RegionBakeKey {
        coord: RegionCoord::from_chunk(chunk_pos, region_layout),
        mode,
    };
    regions
        .get(&region_key)
        .and_then(|region| {
            let (pixel_x, pixel_z) = region.region_pixel(chunk_pos, local_x, local_z)?;
            Some(region.water_depth_at_region_pixel(pixel_x, pixel_z))
        })
        .unwrap_or(0)
}

fn chunk_bake_height_at_block(
    bakes: &BTreeMap<ChunkPos, ChunkBake>,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
) -> Option<i16> {
    let block_pos = BlockPos {
        x: block_x,
        y: 0,
        z: block_z,
    };
    let chunk_pos = block_pos.to_chunk_pos(dimension);
    let (local_x, _, local_z) = block_pos.in_chunk_offset();
    bakes.get(&chunk_pos).and_then(|bake| {
        (bake.mode == mode)
            .then(|| chunk_bake_relief_height(bake, u32::from(local_x), u32::from(local_z)))
            .flatten()
    })
}

fn chunk_bake_height_neighborhood_at_block(
    bakes: &BTreeMap<ChunkPos, ChunkBake>,
    dimension: Dimension,
    mode: RenderMode,
    block_x: i32,
    block_z: i32,
    fallback: i16,
) -> TerrainHeightNeighborhood {
    let height_at =
        |x, z| chunk_bake_height_at_block(bakes, dimension, mode, x, z).unwrap_or(fallback);
    TerrainHeightNeighborhood {
        center: fallback,
        north_west: height_at(block_x - 1, block_z - 1),
        north: height_at(block_x, block_z - 1),
        north_east: height_at(block_x + 1, block_z - 1),
        west: height_at(block_x - 1, block_z),
        east: height_at(block_x + 1, block_z),
        south_west: height_at(block_x - 1, block_z + 1),
        south: height_at(block_x, block_z + 1),
        south_east: height_at(block_x + 1, block_z + 1),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn terrain_lit_color(
    color: RgbaColor,
    heights: TerrainHeightNeighborhood,
    water_depth: u8,
    lighting: TerrainLightingOptions,
) -> RgbaColor {
    if color.alpha == 0 || !lighting.enabled {
        return color;
    }
    let mut normal_strength = lighting.normal_strength.max(0.0);
    let mut shadow_strength = lighting.shadow_strength;
    let mut highlight_strength = lighting.highlight_strength;
    let mut ambient_occlusion = lighting.ambient_occlusion;
    let mut edge_relief_strength = lighting.edge_relief_strength.max(0.0);
    if water_depth > 0 {
        if !lighting.underwater_relief_enabled {
            return color;
        }
        let fade = underwater_depth_factor(water_depth, lighting);
        let minimum_light = lighting.underwater_min_light.clamp(0.0, 1.0);
        let light_factor = fade.max(minimum_light);
        normal_strength *= lighting.underwater_relief_strength.max(0.0) * fade;
        shadow_strength *= light_factor;
        highlight_strength *= light_factor;
        ambient_occlusion *= fade;
        edge_relief_strength *= fade;
    }
    let max_shadow = if water_depth > 0 {
        70.0
    } else {
        lighting.max_shadow.max(0.0)
    };
    if normal_strength == 0.0 {
        return color;
    }
    let (mut dx, mut dz) = heights.sobel_gradient();
    if water_depth == 0 {
        dx = compress_land_slope(dx, lighting.land_slope_softness);
        dz = compress_land_slope(dz, lighting.land_slope_softness);
    }
    let dx = dx * normal_strength;
    let dz = dz * normal_strength;
    let normal_length = (dx.mul_add(dx, dz.mul_add(dz, 4.0)))
        .sqrt()
        .max(f32::EPSILON);
    let normal_x = -dx / normal_length;
    let normal_y = 2.0 / normal_length;
    let normal_z = -dz / normal_length;
    let azimuth = lighting.light_azimuth_degrees.to_radians();
    let elevation = lighting
        .light_elevation_degrees
        .to_radians()
        .clamp(0.01, 1.55);
    let light_horizontal = elevation.cos();
    let light_x = azimuth.sin() * light_horizontal;
    let light_y = elevation.sin();
    let light_z = -azimuth.cos() * light_horizontal;
    let dot = normal_x.mul_add(light_x, normal_y.mul_add(light_y, normal_z * light_z));
    let flat_dot = light_y;
    let relief = (dx.abs() + dz.abs()).min(24.0) / 24.0;
    let relative_light = dot - flat_dot;
    let mut factor = if relative_light >= 0.0 {
        relative_light * highlight_strength * 100.0
    } else {
        relative_light * shadow_strength * 100.0
    };
    factor -= relief * ambient_occlusion * 100.0;
    if edge_relief_strength > 0.0 {
        let edge = heights.edge_relief(lighting.edge_relief_threshold);
        let edge_shadow =
            edge.shadow * edge_relief_strength * lighting.edge_relief_max_shadow.max(0.0);
        let edge_highlight = if relative_light >= 0.0 {
            edge.highlight * edge_relief_strength * lighting.edge_relief_highlight.max(0.0) * 100.0
        } else {
            0.0
        };
        factor += edge_highlight;
        factor -= edge_shadow;
    }
    shade_color_percent(color, factor.round().clamp(-max_shadow, 55.0) as i32)
}

fn compress_land_slope(delta: f32, softness: f32) -> f32 {
    let softness = softness.max(0.0);
    if softness <= f32::EPSILON {
        return 0.0;
    }
    delta / (1.0 + delta.abs() / softness)
}

fn underwater_depth_factor(water_depth: u8, lighting: TerrainLightingOptions) -> f32 {
    if water_depth == 0 {
        return 1.0;
    }
    let fade = lighting.underwater_depth_fade.max(1.0);
    (1.0 - (f32::from(water_depth.saturating_sub(1)) / fade)).clamp(0.0, 1.0)
}

fn shade_color_percent(color: RgbaColor, factor: i32) -> RgbaColor {
    RgbaColor::new(
        shade_channel_percent(color.red, factor),
        shade_channel_percent(color.green, factor),
        shade_channel_percent(color.blue, factor),
        color.alpha,
    )
}

fn shade_channel_percent(channel: u8, factor: i32) -> u8 {
    if factor >= 0 {
        let value = i32::from(channel) + ((255 - i32::from(channel)) * factor / 100);
        u8_from_i32(value.clamp(0, 255))
    } else {
        let value = i32::from(channel) * (100 + factor) / 100;
        u8_from_i32(value.clamp(0, 255))
    }
}

fn u8_from_u32(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn u8_from_i32(value: i32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn i16_from_i32(value: i32) -> i16 {
    i16::try_from(value).unwrap_or_else(|_| {
        if value.is_negative() {
            i16::MIN
        } else {
            i16::MAX
        }
    })
}

fn tile_chunk_positions(job: &RenderJob) -> Result<Vec<ChunkPos>> {
    let (start_x, start_z) = tile_pixel_to_block(job, 0, 0)?;
    let last_pixel = job.tile_size.checked_sub(1).ok_or_else(|| {
        BedrockRenderError::Validation("tile_size must be greater than zero".to_string())
    })?;
    let (end_x, end_z) = tile_pixel_to_block(job, last_pixel, last_pixel)?;
    let min_x = start_x.min(end_x);
    let max_x = start_x.max(end_x);
    let min_z = start_z.min(end_z);
    let max_z = start_z.max(end_z);
    let min_chunk = BlockPos {
        x: min_x,
        y: 0,
        z: min_z,
    }
    .to_chunk_pos(job.coord.dimension);
    let max_chunk = BlockPos {
        x: max_x,
        y: 0,
        z: max_z,
    }
    .to_chunk_pos(job.coord.dimension);
    let x_chunk_count = i64::from(max_chunk.x) - i64::from(min_chunk.x) + 1;
    let z_chunk_count = i64::from(max_chunk.z) - i64::from(min_chunk.z) + 1;
    let capacity = usize::try_from(x_chunk_count.saturating_mul(z_chunk_count)).map_err(|_| {
        BedrockRenderError::Validation("tile chunk coverage is too large".to_string())
    })?;
    let mut positions = Vec::with_capacity(capacity);
    for z in min_chunk.z..=max_chunk.z {
        for x in min_chunk.x..=max_chunk.x {
            positions.push(ChunkPos {
                x,
                z,
                dimension: job.coord.dimension,
            });
        }
    }
    Ok(positions)
}

fn collect_region_plans(
    planned_tiles: &[PlannedTile],
    region_layout: RegionLayout,
) -> Result<Vec<RegionPlan>> {
    region_layout.validate()?;
    let mut regions = BTreeMap::<RegionBakeKey, RegionPlan>::new();
    for planned in planned_tiles {
        for pos in planned_chunk_positions(planned)?.iter().copied() {
            let key = RegionBakeKey {
                coord: RegionCoord::from_chunk(pos, region_layout),
                mode: planned.job.mode,
            };
            regions
                .entry(key)
                .and_modify(|plan| {
                    plan.region.min_chunk_x = plan.region.min_chunk_x.min(pos.x);
                    plan.region.min_chunk_z = plan.region.min_chunk_z.min(pos.z);
                    plan.region.max_chunk_x = plan.region.max_chunk_x.max(pos.x);
                    plan.region.max_chunk_z = plan.region.max_chunk_z.max(pos.z);
                    plan.chunk_positions.push(pos);
                })
                .or_insert_with(|| RegionPlan {
                    key,
                    region: ChunkRegion::new(pos.dimension, pos.x, pos.z, pos.x, pos.z),
                    chunk_positions: vec![pos],
                });
        }
    }
    Ok(regions
        .into_values()
        .map(|mut plan| {
            plan.chunk_positions.sort();
            plan.chunk_positions.dedup();
            plan
        })
        .collect())
}

fn planned_chunk_positions(planned: &PlannedTile) -> Result<Cow<'_, [ChunkPos]>> {
    if let Some(chunk_positions) = &planned.chunk_positions {
        return Ok(Cow::Borrowed(chunk_positions));
    }
    tile_chunk_positions(&planned.job).map(Cow::Owned)
}

fn tile_region_keys(
    planned: &PlannedTile,
    region_layout: RegionLayout,
) -> Result<Vec<RegionBakeKey>> {
    let mut keys = BTreeSet::new();
    for pos in planned_chunk_positions(planned)?.iter().copied() {
        keys.insert(RegionBakeKey {
            coord: RegionCoord::from_chunk(pos, region_layout),
            mode: planned.job.mode,
        });
    }
    Ok(keys.into_iter().collect())
}

fn select_region_wave(
    pending_tiles: &[usize],
    tile_region_keys: &[Vec<RegionBakeKey>],
    region_plan_by_key: &BTreeMap<RegionBakeKey, RegionPlan>,
    memory_budget: usize,
) -> Result<BTreeSet<RegionBakeKey>> {
    let Some(first_tile) = pending_tiles.first().copied() else {
        return Ok(BTreeSet::new());
    };
    let mut selected = BTreeSet::new();
    let mut selected_bytes = 0usize;

    for key in tile_region_keys
        .get(first_tile)
        .ok_or_else(|| BedrockRenderError::Validation("missing tile region keys".to_string()))?
    {
        selected_bytes =
            selected_bytes.saturating_add(region_estimated_bytes(*key, region_plan_by_key)?);
        selected.insert(*key);
    }

    for tile_index in pending_tiles.iter().copied().skip(1) {
        let keys = tile_region_keys.get(tile_index).ok_or_else(|| {
            BedrockRenderError::Validation("missing tile region keys".to_string())
        })?;
        let missing_bytes = keys.iter().try_fold(0usize, |total, key| {
            if selected.contains(key) {
                Ok::<usize, BedrockRenderError>(total)
            } else {
                Ok::<usize, BedrockRenderError>(
                    total.saturating_add(region_estimated_bytes(*key, region_plan_by_key)?),
                )
            }
        })?;
        if selected_bytes.saturating_add(missing_bytes) > memory_budget {
            continue;
        }
        for key in keys {
            selected.insert(*key);
        }
        selected_bytes = selected_bytes.saturating_add(missing_bytes);
    }

    Ok(selected)
}

fn region_estimated_bytes(
    key: RegionBakeKey,
    region_plan_by_key: &BTreeMap<RegionBakeKey, RegionPlan>,
) -> Result<usize> {
    let plan = region_plan_by_key.get(&key).ok_or_else(|| {
        BedrockRenderError::Validation("planned tile references missing region".to_string())
    })?;
    Ok(plan
        .chunk_positions
        .len()
        .saturating_mul(REGION_BAKE_ESTIMATED_BYTES_PER_CHUNK))
}

fn render_web_tile_indexes<F>(
    renderer: &MapRenderer,
    planned_tiles: &[PlannedTile],
    tile_indexes: &[usize],
    options: &RenderOptions,
    regions: &BTreeMap<RegionBakeKey, RegionBake>,
    worker_count: usize,
    sink: &F,
) -> Result<TileComposeStats>
where
    F: Fn(PlannedTile, TileImage) -> Result<()> + Send + Sync,
{
    if tile_indexes.is_empty() {
        return Ok(TileComposeStats::default());
    }
    if worker_count == 1 {
        let mut stats = TileComposeStats::default();
        for index in tile_indexes {
            let planned = planned_tiles.get(*index).ok_or_else(|| {
                BedrockRenderError::Validation("planned tile index is out of range".to_string())
            })?;
            let (tile, tile_stats) = renderer.render_tile_from_cached_regions_with_stats(
                planned.job.clone(),
                options,
                regions,
            )?;
            stats.add(tile_stats);
            sink(planned.clone(), tile)?;
        }
        return Ok(stats);
    }

    let next_tile = Arc::new(AtomicUsize::new(0));
    let queue_capacity = options
        .pipeline_depth
        .max(worker_count.saturating_mul(2))
        .max(1);
    let (sender, receiver) =
        mpsc::sync_channel::<Result<(PlannedTile, TileImage, TileComposeStats)>>(queue_capacity);
    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_tile = Arc::clone(&next_tile);
            let sender = sender.clone();
            let renderer = renderer.clone();
            let options = options.clone();
            scope.spawn(move || {
                loop {
                    if check_cancelled(&options).is_err() {
                        let send_result = sender.send(Err(BedrockRenderError::Cancelled));
                        drop(send_result);
                        return;
                    }
                    let index = next_tile.fetch_add(1, Ordering::Relaxed);
                    let Some(tile_index) = tile_indexes.get(index).copied() else {
                        return;
                    };
                    let Some(planned) = planned_tiles.get(tile_index).cloned() else {
                        let send_result = sender.send(Err(BedrockRenderError::Validation(
                            "planned tile index is out of range".to_string(),
                        )));
                        drop(send_result);
                        return;
                    };
                    let tile_result = renderer
                        .render_tile_from_cached_regions_with_stats(
                            planned.job.clone(),
                            &options,
                            regions,
                        )
                        .map(|(tile, stats)| (planned, tile, stats));
                    if sender.send(tile_result).is_err() {
                        return;
                    }
                }
            });
        }
        drop(sender);
        let mut stats = TileComposeStats::default();
        for message in receiver {
            let (planned, tile, tile_stats) = message?;
            stats.add(tile_stats);
            sink(planned, tile)?;
        }
        Ok::<TileComposeStats, BedrockRenderError>(stats)
    })
}

fn empty_region_payload(
    mode: RenderMode,
    width: u32,
    height: u32,
    missing_color: RgbaColor,
) -> Result<RegionBakePayload> {
    Ok(match mode {
        RenderMode::SurfaceBlocks => RegionBakePayload::Surface(SurfacePlane {
            colors: RgbaPlane::new(width, height, missing_color)?,
            heights: HeightPlane::new(width, height)?,
            relief_heights: HeightPlane::new(width, height)?,
            water_depths: DepthPlane::new(width, height)?,
        }),
        RenderMode::HeightMap => RegionBakePayload::HeightMap {
            colors: RgbaPlane::new(width, height, missing_color)?,
            heights: HeightPlane::new(width, height)?,
        },
        RenderMode::Biome { .. }
        | RenderMode::RawBiomeLayer { .. }
        | RenderMode::LayerBlocks { .. }
        | RenderMode::CaveSlice { .. } => {
            RegionBakePayload::Colors(RgbaPlane::new(width, height, missing_color)?)
        }
    })
}

fn copy_chunk_bake_to_region(
    bake: &ChunkBake,
    payload: &mut RegionBakePayload,
    region_x: u32,
    region_z: u32,
) {
    for local_z in 0..16_u32 {
        for local_x in 0..16_u32 {
            let color = chunk_bake_color(bake, local_x, local_z);
            let height = chunk_bake_height(bake, local_x, local_z);
            let dst_x = region_x + local_x;
            let dst_z = region_z + local_z;
            match payload {
                RegionBakePayload::Colors(colors) => {
                    if let Some(color) = color {
                        colors.set_color(dst_x, dst_z, color);
                    }
                }
                RegionBakePayload::Surface(surface) => {
                    if let Some(color) = color {
                        surface.colors.set_color(dst_x, dst_z, color);
                    }
                    if let Some(height) = height {
                        surface.heights.set_height(dst_x, dst_z, height);
                    }
                    if let Some(relief_height) = chunk_bake_relief_height(bake, local_x, local_z) {
                        surface
                            .relief_heights
                            .set_height(dst_x, dst_z, relief_height);
                    }
                    let depth = chunk_bake_water_depth(bake, local_x, local_z);
                    if depth > 0 {
                        surface.water_depths.set_depth(dst_x, dst_z, depth);
                    }
                }
                RegionBakePayload::HeightMap { colors, heights } => {
                    if let Some(color) = color {
                        colors.set_color(dst_x, dst_z, color);
                    }
                    if let Some(height) = height {
                        heights.set_height(dst_x, dst_z, height);
                    }
                }
            }
        }
    }
}

fn chunk_bake_color(bake: &ChunkBake, local_x: u32, local_z: u32) -> Option<RgbaColor> {
    match &bake.payload {
        ChunkBakePayload::Colors(colors) | ChunkBakePayload::HeightMap { colors, .. } => {
            colors.color_at(local_x, local_z)
        }
        ChunkBakePayload::Surface(surface) => surface.colors.color_at(local_x, local_z),
    }
}

fn chunk_bake_height(bake: &ChunkBake, local_x: u32, local_z: u32) -> Option<i16> {
    match &bake.payload {
        ChunkBakePayload::Surface(surface) => surface.heights.height_at(local_x, local_z),
        ChunkBakePayload::HeightMap { heights, .. } => heights.height_at(local_x, local_z),
        ChunkBakePayload::Colors(_) => None,
    }
}

fn chunk_bake_relief_height(bake: &ChunkBake, local_x: u32, local_z: u32) -> Option<i16> {
    match &bake.payload {
        ChunkBakePayload::Surface(surface) => surface.relief_heights.height_at(local_x, local_z),
        ChunkBakePayload::HeightMap { heights, .. } => heights.height_at(local_x, local_z),
        ChunkBakePayload::Colors(_) => None,
    }
}

fn chunk_bake_water_depth(bake: &ChunkBake, local_x: u32, local_z: u32) -> u8 {
    match &bake.payload {
        ChunkBakePayload::Surface(surface) => surface.water_depths.depth_at(local_x, local_z),
        ChunkBakePayload::Colors(_) | ChunkBakePayload::HeightMap { .. } => 0,
    }
}

fn prefetch_parallel<T, R, F>(tasks: Vec<T>, options: &RenderOptions, worker: F) -> Result<Vec<R>>
where
    T: Copy + Send + Sync,
    R: Send,
    F: Fn(T) -> Result<R> + Copy + Send + Sync,
{
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = options
        .threading
        .resolve_for_profile_checked(options.execution_profile, tasks.len())?;
    if worker_count == 1 {
        let mut results = Vec::with_capacity(tasks.len());
        for task in tasks {
            check_cancelled(options)?;
            results.push(worker(task)?);
        }
        return Ok(results);
    }

    let total_tasks = tasks.len();
    let next_task = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel::<Result<R>>();
    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_task = Arc::clone(&next_task);
            let sender = sender.clone();
            let options = options.clone();
            let tasks = &tasks;
            scope.spawn(move || {
                loop {
                    if check_cancelled(&options).is_err() {
                        let send_result = sender.send(Err(BedrockRenderError::Cancelled));
                        drop(send_result);
                        return;
                    }
                    let index = next_task.fetch_add(1, Ordering::Relaxed);
                    if index >= tasks.len() {
                        return;
                    }
                    let task = tasks[index];
                    if sender.send(worker(task)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(sender);
        let mut results = Vec::with_capacity(total_tasks);
        for result in receiver {
            results.push(result?);
            if results.len() == total_tasks {
                break;
            }
        }
        Ok(results)
    })
}

fn block_y_to_subchunk_y(y: i32) -> Result<i8> {
    i8::try_from(y.div_euclid(16)).map_err(|_| {
        BedrockRenderError::Validation(format!(
            "block y={y} cannot be represented as a Bedrock subchunk index"
        ))
    })
}

#[allow(dead_code)]
fn block_name_at(subchunk: &SubChunk, block_pos: BlockPos) -> Option<&str> {
    let (local_x, y, local_z) = block_pos.in_chunk_offset();
    let local_y = u8::try_from(y - i32::from(subchunk.y) * 16).ok()?;
    if local_y >= 16 {
        return None;
    }
    subchunk
        .block_state_at(local_x, local_y, local_z)
        .map(|state| state.name.as_str())
}

fn biome_storage_bucket_y(y: i32) -> i32 {
    y.div_euclid(16) * 16
}

fn local_biome_y(storage: &bedrock_world::ParsedBiomeStorage, y: i32) -> Result<u8> {
    let local_y = if let Some(start_y) = storage.y {
        y - start_y
    } else {
        0
    };
    u8::try_from(local_y).map_err(|_| {
        BedrockRenderError::Validation(format!("biome y={y} is outside biome storage bounds"))
    })
}

fn non_empty_biome_id(id: Option<u32>) -> Option<u32> {
    id.filter(|id| *id != u32::MAX)
}

fn encode_image(
    rgba: &[u8],
    width: u32,
    height: u32,
    format: ImageFormat,
) -> Result<Option<Vec<u8>>> {
    match format {
        ImageFormat::Rgba => Ok(None),
        ImageFormat::WebP => encode_webp(rgba, width, height).map(Some),
        ImageFormat::Png => encode_png(rgba, width, height).map(Some),
    }
}

#[cfg(feature = "webp")]
fn encode_webp(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    WebPEncoder::new_lossless(&mut output)
        .write_image(rgba, width, height, ExtendedColorType::Rgba8)
        .map_err(|error| BedrockRenderError::image("failed to encode WebP tile", error))?;
    Ok(output)
}

#[cfg(not(feature = "webp"))]
fn encode_webp(_rgba: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>> {
    Err(BedrockRenderError::UnsupportedMode(
        "webp feature is disabled".to_string(),
    ))
}

#[cfg(feature = "png")]
fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    PngEncoder::new(&mut output)
        .write_image(rgba, width, height, ExtendedColorType::Rgba8)
        .map_err(|error| BedrockRenderError::image("failed to encode PNG tile", error))?;
    Ok(output)
}

#[cfg(not(feature = "png"))]
fn encode_png(_rgba: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>> {
    Err(BedrockRenderError::UnsupportedMode(
        "png feature is disabled".to_string(),
    ))
}

fn check_cancelled(options: &RenderOptions) -> Result<()> {
    if options
        .cancel
        .as_ref()
        .is_some_and(RenderCancelFlag::is_cancelled)
    {
        return Err(BedrockRenderError::Cancelled);
    }
    Ok(())
}

fn emit_progress(options: &RenderOptions, completed_tiles: usize, total_tiles: usize) {
    if let Some(progress) = &options.progress {
        progress.emit(RenderProgress {
            completed_tiles,
            total_tiles,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bedrock_world::{
        ChunkKey, ChunkRecordTag, MemoryStorage, NbtTag, OpenOptions, WorldStorage,
        block_storage_index,
    };
    use indexmap::IndexMap;

    fn cardinal_heights(west: i16, east: i16, north: i16, south: i16) -> TerrainHeightNeighborhood {
        fn average(first: i16, second: i16) -> i16 {
            i16_from_i32(i32::midpoint(i32::from(first), i32::from(second)))
        }

        TerrainHeightNeighborhood {
            center: 64,
            north_west: average(west, north),
            north,
            north_east: average(east, north),
            west,
            east,
            south_west: average(west, south),
            south,
            south_east: average(east, south),
        }
    }

    fn uniform_neighbor_heights(center: i16, neighbor: i16) -> TerrainHeightNeighborhood {
        TerrainHeightNeighborhood {
            center,
            north_west: neighbor,
            north: neighbor,
            north_east: neighbor,
            west: neighbor,
            east: neighbor,
            south_west: neighbor,
            south: neighbor,
            south_east: neighbor,
        }
    }

    #[test]
    fn render_threading_validates_fixed_range_and_auto_is_not_capped_to_eight() {
        let expected_auto = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(10_000);
        assert_eq!(
            RenderThreadingOptions::Auto
                .resolve_checked(10_000)
                .expect("auto threads"),
            expected_auto
        );
        assert_eq!(
            RenderThreadingOptions::Fixed(MAX_RENDER_THREADS)
                .resolve_checked(10_000)
                .expect("max fixed threads"),
            MAX_RENDER_THREADS
        );
        assert!(
            RenderThreadingOptions::Fixed(0)
                .resolve_checked(10)
                .is_err()
        );
        assert!(
            RenderThreadingOptions::Fixed(MAX_RENDER_THREADS + 1)
                .resolve_checked(10)
                .is_err()
        );
        let expected_interactive = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .div_euclid(2)
            .clamp(1, 6)
            .min(10_000);
        assert_eq!(
            RenderThreadingOptions::Auto
                .resolve_for_profile_checked(RenderExecutionProfile::Interactive, 10_000)
                .expect("interactive threads"),
            expected_interactive
        );
    }

    #[test]
    fn render_memory_budget_auto_has_stable_profile_caps() {
        let export_budget = RenderMemoryBudget::Auto
            .resolve_bytes(RenderExecutionProfile::Export)
            .expect("export budget");
        let interactive_budget = RenderMemoryBudget::Auto
            .resolve_bytes(RenderExecutionProfile::Interactive)
            .expect("interactive budget");
        assert!(export_budget >= MIN_AUTO_MEMORY_BUDGET_BYTES);
        assert!(export_budget <= MAX_AUTO_MEMORY_BUDGET_BYTES);
        assert!(interactive_budget <= DEFAULT_INTERACTIVE_MEMORY_BUDGET_BYTES);
        assert_eq!(
            RenderMemoryBudget::FixedBytes(64 * 1024 * 1024)
                .resolve_bytes(RenderExecutionProfile::Export),
            Some(64 * 1024 * 1024)
        );
        assert_eq!(
            RenderMemoryBudget::Disabled.resolve_bytes(RenderExecutionProfile::Export),
            None
        );
    }

    #[test]
    fn render_layout_supports_pixels_per_block() {
        let layout = RenderLayout {
            chunks_per_tile: 16,
            blocks_per_pixel: 1,
            pixels_per_block: 2,
        };
        assert_eq!(layout.tile_size(), Some(512));
        let job = RenderJob::chunk_tile(
            TileCoord {
                x: 0,
                z: 0,
                dimension: Dimension::Overworld,
            },
            RenderMode::HeightMap,
            layout,
        )
        .expect("chunk tile");
        assert_eq!(job.tile_size, 512);
        assert_eq!(job.scale, 1);
        assert_eq!(job.pixels_per_block, 2);
    }

    #[test]
    fn render_layout_rejects_invalid_pixel_sizes() {
        assert_eq!(
            RenderLayout {
                chunks_per_tile: 16,
                blocks_per_pixel: 3,
                pixels_per_block: 2,
            }
            .tile_size(),
            None
        );
        assert_eq!(
            RenderLayout {
                chunks_per_tile: 256,
                blocks_per_pixel: 1,
                pixels_per_block: 2,
            }
            .tile_size(),
            None
        );
    }

    #[test]
    fn web_tile_pipeline_bakes_shared_chunks_once() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let layout = ChunkTileLayout::default();
        let region = ChunkRegion::new(Dimension::Overworld, 0, 0, 15, 15);
        let planned =
            MapRenderer::plan_region_tiles(region, RenderMode::HeightMap, layout).expect("plan");
        let planned = vec![planned[0].clone(), planned[0].clone()];
        let emitted_tiles = AtomicUsize::new(0);
        let result = renderer
            .render_web_tiles_blocking(
                &planned,
                RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Fixed(4),
                    memory_budget: RenderMemoryBudget::Disabled,
                    ..RenderOptions::default()
                },
                |_planned, tile| {
                    assert_eq!(tile.rgba.len(), 256 * 256 * 4);
                    emitted_tiles.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                },
            )
            .expect("render web tiles");
        assert_eq!(emitted_tiles.load(Ordering::Relaxed), 2);
        assert_eq!(result.stats.planned_tiles, 2);
        assert_eq!(result.stats.planned_regions, 1);
        assert_eq!(result.stats.baked_regions, 1);
        assert_eq!(result.stats.unique_chunks, 256);
        assert_eq!(result.stats.baked_chunks, 256);
    }

    #[test]
    fn tile_coordinate_handles_negative_world_blocks() {
        let job = RenderJob {
            coord: TileCoord {
                x: -1,
                z: 2,
                dimension: Dimension::Overworld,
            },
            tile_size: 256,
            scale: 2,
            ..RenderJob::new(
                TileCoord {
                    x: 0,
                    z: 0,
                    dimension: Dimension::Overworld,
                },
                RenderMode::Biome { y: 64 },
            )
        };
        assert_eq!(
            tile_pixel_to_block(&job, 0, 0).expect("coord"),
            (-512, 1024)
        );
        assert_eq!(
            tile_pixel_to_block(&job, 1, 1).expect("coord"),
            (-510, 1026)
        );
    }

    #[test]
    fn render_rgba_tile_has_expected_size() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let job = RenderJob {
            tile_size: 4,
            ..RenderJob::new(
                TileCoord {
                    x: 0,
                    z: 0,
                    dimension: Dimension::Overworld,
                },
                RenderMode::Biome { y: 64 },
            )
        };
        let tile = renderer
            .render_tile_with_options_blocking(
                job,
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    ..RenderOptions::default()
                },
            )
            .expect("render");
        assert_eq!(tile.width, 4);
        assert_eq!(tile.height, 4);
        assert_eq!(tile.rgba.len(), 4 * 4 * 4);
        assert!(tile.encoded.is_none());
    }

    #[test]
    fn cancelled_batch_returns_error() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let cancel = RenderCancelFlag::new();
        cancel.cancel();
        let result = renderer.render_tiles_blocking(
            vec![RenderJob::new(
                TileCoord {
                    x: 0,
                    z: 0,
                    dimension: Dimension::Overworld,
                },
                RenderMode::Biome { y: 64 },
            )],
            RenderOptions {
                format: ImageFormat::Rgba,
                cancel: Some(cancel),
                ..RenderOptions::default()
            },
        );
        assert!(matches!(result, Err(BedrockRenderError::Cancelled)));
    }

    #[test]
    fn chunk_region_plans_web_map_tiles() {
        let layout = ChunkTileLayout::default();
        let region = ChunkRegion::new(Dimension::Overworld, -1, -1, 15, 15);
        let tiles = MapRenderer::plan_region_tiles(region, RenderMode::HeightMap, layout)
            .expect("plan region");

        assert_eq!(tiles.len(), 4);
        assert_eq!(
            tiles[0].relative_path(TilePathScheme::WebMap, "webp"),
            std::path::PathBuf::from("overworld")
                .join("heightmap")
                .join("16c-1bpp-1ppb")
                .join("-1")
                .join("-1.webp")
        );
        assert_eq!(tiles[0].job.tile_size, 256);
        assert_eq!(tiles[0].job.scale, 1);
    }

    #[test]
    fn two_pixels_per_block_repeats_block_color() {
        let job = RenderJob {
            coord: TileCoord {
                x: -1,
                z: -1,
                dimension: Dimension::Overworld,
            },
            tile_size: 512,
            scale: 1,
            pixels_per_block: 2,
            mode: RenderMode::HeightMap,
        };
        assert_eq!(
            tile_pixel_to_block(&job, 0, 0).expect("origin"),
            (-256, -256)
        );
        assert_eq!(
            tile_pixel_to_block(&job, 1, 1).expect("duplicated first pixel"),
            (-256, -256)
        );
        assert_eq!(
            tile_pixel_to_block(&job, 2, 2).expect("next block"),
            (-255, -255)
        );
    }

    #[test]
    fn soft_lighting_keeps_flat_terrain_stable() {
        let base = RgbaColor::new(100, 120, 140, 255);
        let shaded = terrain_lit_color(
            base,
            cardinal_heights(64, 64, 64, 64),
            0,
            TerrainLightingOptions::soft(),
        );
        assert_eq!(shaded, base);
    }

    #[test]
    fn strong_lighting_changes_slope_contrast() {
        let base = RgbaColor::new(100, 100, 100, 255);
        let soft = terrain_lit_color(
            base,
            cardinal_heights(64, 72, 64, 64),
            0,
            TerrainLightingOptions::soft(),
        );
        let strong = terrain_lit_color(
            base,
            cardinal_heights(64, 72, 64, 64),
            0,
            TerrainLightingOptions::strong(),
        );
        let soft_delta = i16::from(soft.red).abs_diff(i16::from(base.red));
        let strong_delta = i16::from(strong.red).abs_diff(i16::from(base.red));
        assert!(soft_delta > 0);
        assert!(strong_delta > 0);
        assert_ne!(strong, soft);
        assert_eq!(strong.alpha, 255);
    }

    #[test]
    fn land_max_shadow_limits_darkest_terrain_shadow() {
        let base = RgbaColor::new(100, 100, 100, 255);
        let mut uncapped = TerrainLightingOptions::strong();
        uncapped.max_shadow = 70.0;
        let mut capped = uncapped;
        capped.max_shadow = 12.0;

        let dark = terrain_lit_color(base, cardinal_heights(72, 64, 64, 64), 0, uncapped);
        let limited = terrain_lit_color(base, cardinal_heights(72, 64, 64, 64), 0, capped);

        assert!(limited.red > dark.red);
        assert!(limited.red >= 88);
    }

    #[test]
    fn land_slope_softness_reduces_extreme_shadow_overlap() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let mut harsh = TerrainLightingOptions::strong();
        harsh.land_slope_softness = 1_000.0;
        harsh.max_shadow = 70.0;
        harsh.edge_relief_strength = 0.0;
        let mut softened = harsh;
        softened.land_slope_softness = 4.0;

        let steep = cardinal_heights(112, 32, 64, 64);
        let harsh_color = terrain_lit_color(base, steep, 0, harsh);
        let softened_color = terrain_lit_color(base, steep, 0, softened);

        assert!(softened_color.red > harsh_color.red);
    }

    #[test]
    fn land_slope_softness_keeps_small_slopes_visible() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let shaded = terrain_lit_color(
            base,
            cardinal_heights(64, 68, 64, 64),
            0,
            TerrainLightingOptions::soft(),
        );

        assert_ne!(shaded, base);
    }

    #[test]
    fn land_slope_softness_does_not_reduce_underwater_relief() {
        let base = RgbaColor::new(100, 100, 100, 255);
        let mut flat_land = TerrainLightingOptions::strong();
        flat_land.land_slope_softness = 0.0;
        let mut normal_land = flat_land;
        normal_land.land_slope_softness = 8.0;
        let slope = cardinal_heights(96, 32, 64, 64);

        let flat_land_water = terrain_lit_color(base, slope, 1, flat_land);
        let normal_land_water = terrain_lit_color(base, slope, 1, normal_land);

        assert_eq!(flat_land_water, normal_land_water);
    }

    #[test]
    fn edge_relief_does_not_change_flat_terrain() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let shaded = terrain_lit_color(
            base,
            uniform_neighbor_heights(64, 64),
            0,
            TerrainLightingOptions::strong(),
        );

        assert_eq!(shaded, base);
    }

    #[test]
    fn edge_relief_adds_contact_shadow_to_deep_pit() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let mut without_edge = TerrainLightingOptions::strong();
        without_edge.edge_relief_strength = 0.0;
        let mut with_edge = without_edge;
        with_edge.edge_relief_strength = 1.0;
        with_edge.edge_relief_max_shadow = 18.0;

        let pit = uniform_neighbor_heights(48, 72);
        let flat_pit = terrain_lit_color(base, pit, 0, without_edge);
        let edged_pit = terrain_lit_color(base, pit, 0, with_edge);

        assert!(edged_pit.red < flat_pit.red);
    }

    #[test]
    fn edge_relief_threshold_ignores_small_height_noise() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let mut lighting = TerrainLightingOptions::strong();
        lighting.edge_relief_strength = 1.0;
        lighting.edge_relief_threshold = 3.0;

        let shaded = terrain_lit_color(base, uniform_neighbor_heights(64, 66), 0, lighting);

        assert_eq!(shaded, base);
    }

    #[test]
    fn edge_relief_max_shadow_limits_pit_edge_darkness() {
        let base = RgbaColor::new(120, 120, 120, 255);
        let mut limited = TerrainLightingOptions::strong();
        limited.edge_relief_strength = 1.0;
        limited.edge_relief_max_shadow = 4.0;
        let mut strong = limited;
        strong.edge_relief_max_shadow = 18.0;

        let pit = uniform_neighbor_heights(48, 72);
        let limited_color = terrain_lit_color(base, pit, 0, limited);
        let strong_color = terrain_lit_color(base, pit, 0, strong);

        assert!(limited_color.red > strong_color.red);
        assert!(limited_color.red >= 115);
    }

    #[test]
    fn land_shadow_limit_does_not_reduce_underwater_relief() {
        let base = RgbaColor::new(100, 100, 100, 255);
        let mut capped = TerrainLightingOptions::strong();
        capped.max_shadow = 1.0;
        let normal = TerrainLightingOptions::strong();

        let capped_water = terrain_lit_color(base, cardinal_heights(72, 64, 64, 64), 1, capped);
        let normal_water = terrain_lit_color(base, cardinal_heights(72, 64, 64, 64), 1, normal);

        assert_eq!(capped_water, normal_water);
    }

    #[test]
    fn terrain_lighting_preserves_transparency() {
        let base = RgbaColor::new(100, 100, 100, 0);
        let shaded = terrain_lit_color(
            base,
            cardinal_heights(64, 72, 64, 64),
            0,
            TerrainLightingOptions::strong(),
        );
        assert_eq!(shaded, base);
    }

    #[test]
    fn terrain_lighting_scope_matches_surface_and_heightmap() {
        let surface = SurfaceRenderOptions::default();
        assert!(lighting_enabled_for(RenderMode::SurfaceBlocks, surface));
        assert!(lighting_enabled_for(RenderMode::HeightMap, surface));
        assert!(!lighting_enabled_for(RenderMode::Biome { y: 64 }, surface));
        assert!(!lighting_enabled_for(
            RenderMode::LayerBlocks { y: 64 },
            surface
        ));
        assert!(!lighting_enabled_for(
            RenderMode::CaveSlice { y: 32 },
            surface
        ));
    }

    #[test]
    fn fixed_y_layer_uses_xz_plane_not_vertical_slice() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(&ChunkKey::subchunk(pos, 4).encode(), &test_subchunk_bytes())
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let palette = RenderPalette::default()
            .with_block_color("minecraft:x_axis", RgbaColor::new(10, 0, 0, 255))
            .with_block_color("minecraft:z_axis", RgbaColor::new(0, 10, 0, 255));
        let renderer = MapRenderer::new(world, palette);
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 2,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::LayerBlocks { y: 64 },
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render layer");

        assert_eq!(&tile.rgba[0..4], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[4..8], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[8..12], &[0, 10, 0, 255]);
    }

    #[test]
    fn pixels_per_block_renders_repeated_source_blocks() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(&ChunkKey::subchunk(pos, 4).encode(), &test_subchunk_bytes())
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let palette = RenderPalette::default()
            .with_block_color("minecraft:x_axis", RgbaColor::new(10, 0, 0, 255))
            .with_block_color("minecraft:z_axis", RgbaColor::new(0, 10, 0, 255));
        let renderer = MapRenderer::new(world, palette);
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 4,
                    pixels_per_block: 2,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::LayerBlocks { y: 64 },
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render layer");

        assert_eq!(&tile.rgba[0..4], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[4..8], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[16..20], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[20..24], &[10, 0, 0, 255]);
        assert_eq!(&tile.rgba[32..36], &[0, 10, 0, 255]);
    }

    #[test]
    fn heightmap_and_cave_slice_render_rgba() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        for mode in [RenderMode::HeightMap, RenderMode::CaveSlice { y: 32 }] {
            let tile = renderer
                .render_tile_with_options_blocking(
                    RenderJob {
                        tile_size: 2,
                        ..RenderJob::new(
                            TileCoord {
                                x: 0,
                                z: 0,
                                dimension: Dimension::Overworld,
                            },
                            mode,
                        )
                    },
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render diagnostic mode");
            assert_eq!(tile.rgba.len(), 16);
        }
    }

    #[test]
    fn missing_fixed_y_layer_renders_transparent_pixels() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let diagnostics = Arc::new(Mutex::new(RenderDiagnostics::default()));
        let diagnostics_sink = RenderDiagnosticsSink::new({
            let diagnostics = Arc::clone(&diagnostics);
            move |value| {
                diagnostics.lock().expect("diagnostics lock").add(value);
            }
        });
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::LayerBlocks { y: 64 },
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    diagnostics: Some(diagnostics_sink),
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render missing layer");

        assert_eq!(&tile.rgba[0..4], &[0, 0, 0, 0]);
        let diagnostics = diagnostics.lock().expect("diagnostics lock");
        assert_eq!(diagnostics.transparent_pixels, 256);
        assert_eq!(diagnostics.purple_error_pixels, 0);
    }

    #[test]
    fn missing_biome_chunk_renders_transparent_not_unknown_biome() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::Biome { y: 64 },
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render missing biome chunk");

        assert_eq!(&tile.rgba[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn biome_layer_falls_back_to_top_non_empty_biome() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data3D).encode(),
                &test_data3d_single_biome_bytes(4),
            )
            .expect("put Data3D");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let expected = RenderPalette::default().biome_color(4).to_array();
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::Biome { y: 64 },
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render biome fallback");

        assert_eq!(&tile.rgba[0..4], &expected);
    }

    #[test]
    fn surface_blocks_uses_highest_non_air_block() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data2D).encode(),
                &test_data2d_bytes(65, 4),
            )
            .expect("put data2d");
        storage
            .put(
                &ChunkKey::subchunk(pos, 4).encode(),
                &test_surface_subchunk_bytes([
                    ("minecraft:air", 0_u16),
                    ("minecraft:stone", 1),
                    ("minecraft:grass_block", 2),
                ]),
            )
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let palette = RenderPalette::default()
            .with_block_color("minecraft:stone", RgbaColor::new(10, 10, 10, 255))
            .with_block_color("minecraft:grass_block", RgbaColor::new(20, 200, 20, 255));
        let renderer = MapRenderer::new(world, palette);
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::SurfaceBlocks,
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    surface: SurfaceRenderOptions {
                        biome_tint: false,
                        ..SurfaceRenderOptions::default()
                    },
                    ..RenderOptions::default()
                },
            )
            .expect("render surface");

        assert_eq!(&tile.rgba[0..4], &[20, 200, 20, 255]);
    }

    #[test]
    fn surface_blocks_scan_above_heightmap_for_tree_canopy() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data2D).encode(),
                &test_data2d_bytes(64, 4),
            )
            .expect("put data2d");
        storage
            .put(
                &ChunkKey::subchunk(pos, 4).encode(),
                &test_surface_subchunk_bytes([
                    ("minecraft:air", 0_u16),
                    ("minecraft:stone", 1),
                    ("minecraft:oak_leaves", 2),
                ]),
            )
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let palette = RenderPalette::default()
            .with_block_color("minecraft:stone", RgbaColor::new(10, 10, 10, 255))
            .with_block_color("minecraft:oak_leaves", RgbaColor::new(20, 120, 20, 255));
        let renderer = MapRenderer::new(world, palette);
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::SurfaceBlocks,
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    surface: SurfaceRenderOptions {
                        biome_tint: false,
                        height_shading: false,
                        ..SurfaceRenderOptions::default()
                    },
                    ..RenderOptions::default()
                },
            )
            .expect("render surface");

        assert_eq!(&tile.rgba[0..4], &[20, 120, 20, 255]);
    }

    #[test]
    fn surface_water_blends_underlying_block() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data2D).encode(),
                &test_data2d_bytes(65, 7),
            )
            .expect("put data2d");
        storage
            .put(
                &ChunkKey::subchunk(pos, 4).encode(),
                &test_surface_subchunk_bytes([
                    ("minecraft:air", 0_u16),
                    ("minecraft:sand", 1),
                    ("minecraft:water", 2),
                ]),
            )
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::SurfaceBlocks,
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render water surface");

        assert_eq!(tile.rgba[3], 255);
        assert_ne!(&tile.rgba[0..3], &[43, 92, 210]);
        assert_ne!(&tile.rgba[0..3], &[218, 210, 158]);
    }

    #[test]
    fn transparent_water_records_seabed_relief_height_and_depth() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data2D).encode(),
                &test_data2d_bytes(65, 7),
            )
            .expect("put data2d");
        storage
            .put(
                &ChunkKey::subchunk(pos, 4).encode(),
                &test_surface_subchunk_bytes([
                    ("minecraft:air", 0_u16),
                    ("minecraft:sand", 1),
                    ("minecraft:water", 2),
                ]),
            )
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let bake = renderer
            .bake_chunk_blocking(pos, BakeOptions::default())
            .expect("bake surface chunk");
        let ChunkBakePayload::Surface(surface) = bake.payload else {
            panic!("expected surface payload");
        };

        assert_eq!(surface.heights.height_at(0, 0), Some(65));
        assert_eq!(surface.relief_heights.height_at(0, 0), Some(64));
        assert_eq!(surface.water_depths.depth_at(0, 0), 1);
    }

    #[test]
    fn underwater_depth_fade_reduces_deep_water_relief() {
        let base = RgbaColor::new(100, 100, 100, 255);
        let shallow = terrain_lit_color(
            base,
            cardinal_heights(64, 72, 64, 64),
            1,
            TerrainLightingOptions::strong(),
        );
        let deep = terrain_lit_color(
            base,
            cardinal_heights(64, 72, 64, 64),
            16,
            TerrainLightingOptions::strong(),
        );
        let shallow_delta = i16::from(shallow.red).abs_diff(i16::from(base.red));
        let deep_delta = i16::from(deep.red).abs_diff(i16::from(base.red));
        assert!(shallow_delta > deep_delta);
    }

    #[test]
    fn surface_diagnostics_count_unknown_blocks() {
        let pos = ChunkPos {
            x: 0,
            z: 0,
            dimension: Dimension::Overworld,
        };
        let storage = Arc::new(MemoryStorage::new());
        storage
            .put(
                &ChunkKey::new(pos, ChunkRecordTag::Data2D).encode(),
                &test_data2d_bytes(64, 4),
            )
            .expect("put data2d");
        storage
            .put(
                &ChunkKey::subchunk(pos, 4).encode(),
                &test_surface_subchunk_bytes([
                    ("minecraft:air", 0_u16),
                    ("minecraft:custom_unknown", 1),
                    ("minecraft:custom_unknown", 1),
                ]),
            )
            .expect("put subchunk");
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            storage,
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let diagnostics = Arc::new(Mutex::new(RenderDiagnostics::default()));
        let diagnostics_sink = RenderDiagnosticsSink::new({
            let diagnostics = Arc::clone(&diagnostics);
            move |value| {
                diagnostics.lock().expect("diagnostics lock").add(value);
            }
        });
        renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::SurfaceBlocks,
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    threading: RenderThreadingOptions::Single,
                    diagnostics: Some(diagnostics_sink),
                    ..RenderOptions::default()
                },
            )
            .expect("render unknown surface");

        let diagnostics = diagnostics.lock().expect("diagnostics lock");
        assert_eq!(diagnostics.unknown_blocks, 256);
        assert_eq!(
            diagnostics
                .unknown_blocks_by_name
                .get("minecraft:custom_unknown")
                .copied(),
            Some(256)
        );
        assert_eq!(diagnostics.purple_error_pixels, 256);
    }

    #[test]
    fn missing_heightmap_renders_transparent_pixels() {
        let world = Arc::new(BedrockWorld::from_storage(
            "memory",
            Arc::new(MemoryStorage::new()),
            OpenOptions::default(),
        ));
        let renderer = MapRenderer::new(world, RenderPalette::default());
        let diagnostics = Arc::new(Mutex::new(RenderDiagnostics::default()));
        let diagnostics_sink = RenderDiagnosticsSink::new({
            let diagnostics = Arc::clone(&diagnostics);
            move |value| {
                diagnostics.lock().expect("diagnostics lock").add(value);
            }
        });
        let tile = renderer
            .render_tile_with_options_blocking(
                RenderJob {
                    tile_size: 1,
                    ..RenderJob::new(
                        TileCoord {
                            x: 0,
                            z: 0,
                            dimension: Dimension::Overworld,
                        },
                        RenderMode::SurfaceBlocks,
                    )
                },
                &RenderOptions {
                    format: ImageFormat::Rgba,
                    diagnostics: Some(diagnostics_sink),
                    threading: RenderThreadingOptions::Single,
                    ..RenderOptions::default()
                },
            )
            .expect("render missing heightmap");

        assert_eq!(&tile.rgba[0..4], &[0, 0, 0, 0]);
        let diagnostics = diagnostics.lock().expect("diagnostics lock");
        assert_eq!(diagnostics.transparent_pixels, 256);
        assert_eq!(diagnostics.missing_chunks, 256);
        assert_eq!(diagnostics.missing_heightmaps, 0);
    }

    #[test]
    fn tile_cache_key_changes_with_world_signature_and_layout() {
        let base = TileCacheKey {
            world_id: "world".to_string(),
            world_signature: "a".to_string(),
            renderer_version: RENDERER_CACHE_VERSION,
            palette_version: DEFAULT_PALETTE_VERSION,
            dimension: Dimension::Overworld,
            mode: "surface".to_string(),
            chunks_per_tile: 16,
            blocks_per_pixel: 1,
            pixels_per_block: 1,
            tile_x: 0,
            tile_z: 0,
            extension: "webp".to_string(),
        };
        let cache = TileCache::new("cache", 4);
        let signature_path = cache.path_for_key(&base);
        let mut changed_signature = base.clone();
        changed_signature.world_signature = "b".to_string();
        let mut changed_layout = base.clone();
        changed_layout.blocks_per_pixel = 4;
        let mut changed_pixel_scale = base.clone();
        changed_pixel_scale.pixels_per_block = 2;

        assert_ne!(signature_path, cache.path_for_key(&changed_signature));
        assert_ne!(signature_path, cache.path_for_key(&changed_layout));
        assert_ne!(signature_path, cache.path_for_key(&changed_pixel_scale));
    }

    fn test_subchunk_bytes() -> Vec<u8> {
        let palette = ["minecraft:x_axis", "minecraft:z_axis"];
        let mut bytes = vec![8, 1, 1 << 1];
        let mut words = vec![0_u32; 128];
        for local_z in 0..16_u8 {
            for local_x in 0..16_u8 {
                let value = u32::from(local_z > 0);
                let block_index = block_storage_index(local_x, 0, local_z);
                let word_index = block_index / 32;
                let bit_offset = block_index % 32;
                words[word_index] |= value << bit_offset;
            }
        }
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let palette_len = i32::try_from(palette.len()).expect("test palette length fits i32");
        bytes.extend_from_slice(&palette_len.to_le_bytes());
        for name in palette {
            let tag = NbtTag::Compound(IndexMap::from([
                ("name".to_string(), NbtTag::String(name.to_string())),
                ("states".to_string(), NbtTag::Compound(IndexMap::new())),
                ("version".to_string(), NbtTag::Int(1)),
            ]));
            bytes.extend_from_slice(&bedrock_world::nbt::serialize_root_nbt(&tag).expect("nbt"));
        }
        bytes
    }

    fn test_surface_subchunk_bytes<const N: usize>(palette_entries: [(&str, u16); N]) -> Vec<u8> {
        let bits_per_value = if palette_entries.len() <= 2 {
            1_u8
        } else {
            2_u8
        };
        let values_per_word = usize::from(32 / bits_per_value);
        let word_count = 4096_usize.div_ceil(values_per_word);
        let mut bytes = vec![8, 1, bits_per_value << 1];
        let mut words = vec![0_u32; word_count];
        for local_z in 0..16_u8 {
            for local_x in 0..16_u8 {
                for (local_y, value) in [(0_u8, 1_u16), (1, 2)] {
                    let block_index = block_storage_index(local_x, local_y, local_z);
                    let word_index = block_index / values_per_word;
                    let bit_offset = (block_index % values_per_word) * usize::from(bits_per_value);
                    words[word_index] |= u32::from(value) << bit_offset;
                }
            }
        }
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let palette_len =
            i32::try_from(palette_entries.len()).expect("test palette length fits i32");
        bytes.extend_from_slice(&palette_len.to_le_bytes());
        for (name, _) in palette_entries {
            let tag = NbtTag::Compound(IndexMap::from([
                ("name".to_string(), NbtTag::String(name.to_string())),
                ("states".to_string(), NbtTag::Compound(IndexMap::new())),
                ("version".to_string(), NbtTag::Int(1)),
            ]));
            bytes.extend_from_slice(&bedrock_world::nbt::serialize_root_nbt(&tag).expect("nbt"));
        }
        bytes
    }

    fn test_data2d_bytes(height: i16, biome: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(768);
        for _ in 0..256 {
            bytes.extend_from_slice(&height.to_le_bytes());
        }
        bytes.extend(std::iter::repeat_n(biome, 256));
        bytes
    }

    fn test_data3d_single_biome_bytes(biome: i32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(517);
        for _ in 0..256 {
            bytes.extend_from_slice(&64_i16.to_le_bytes());
        }
        bytes.push(0);
        bytes.extend_from_slice(&biome.to_le_bytes());
        bytes
    }
}
