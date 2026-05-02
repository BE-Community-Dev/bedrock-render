use super::pipeline::{GpuTileComposeInput, GpuTileComposeOutput};
use std::borrow::Cow;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

const WORKGROUP_SIZE: u32 = 64;
const SHADER: &str = r"
const MISSING_HEIGHT: i32 = -32768;

struct Params {
    width: u32,
    height: u32,
    pixel_count: u32,
    lighting_enabled: u32,
    normal_strength: f32,
    shadow_strength: f32,
    highlight_strength: f32,
    ambient_occlusion: f32,
    max_shadow: f32,
    land_slope_softness: f32,
    edge_relief_strength: f32,
    edge_relief_threshold: f32,
    edge_relief_max_shadow: f32,
    edge_relief_highlight: f32,
    underwater_relief_enabled: u32,
    _pad0: u32,
    underwater_relief_strength: f32,
    underwater_depth_fade: f32,
    underwater_min_light: f32,
    light_x: f32,
    light_y: f32,
    light_z: f32,
    flat_dot: f32,
    pixels_per_block: u32,
    blocks_per_pixel: u32,
    block_boundary_enabled: u32,
    _pad1: u32,
    block_boundary_strength: f32,
    block_boundary_flat_strength: f32,
    block_boundary_height_threshold: f32,
    block_boundary_max_shadow: f32,
    block_boundary_highlight_strength: f32,
    block_boundary_softness: f32,
    block_boundary_line_width_pixels: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> colors: array<u32>;
@group(0) @binding(1) var<storage, read> heights: array<i32>;
@group(0) @binding(2) var<storage, read> water_depths: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_pixels: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn channel(value: u32, shift: u32) -> u32 {
    return (value >> shift) & 0xffu;
}

fn pack_color(red: u32, green: u32, blue: u32, alpha: u32) -> u32 {
    return red | (green << 8u) | (blue << 16u) | (alpha << 24u);
}

fn shade_channel(channel_value: u32, factor: i32) -> u32 {
    let channel_i = i32(channel_value);
    var value: i32;
    if (factor >= 0) {
        value = channel_i + ((255 - channel_i) * factor / 100);
    } else {
        value = channel_i * (100 + factor) / 100;
    }
    return u32(clamp(value, 0, 255));
}

fn compress_land_slope(delta: f32, softness: f32) -> f32 {
    let safe_softness = max(softness, 0.0);
    if (safe_softness <= 0.00000011920928955078125) {
        return 0.0;
    }
    return delta / (1.0 + abs(delta) / safe_softness);
}

fn underwater_depth_factor(water_depth: u32) -> f32 {
    if (water_depth == 0u) {
        return 1.0;
    }
    let fade = max(params.underwater_depth_fade, 1.0);
    return clamp(1.0 - (f32(water_depth - 1u) / fade), 0.0, 1.0);
}

fn edge_relief(base: u32, threshold: f32) -> vec2<f32> {
    let center = f32(heights[base]);
    var higher_neighbor_delta = 0.0;
    var lower_neighbor_delta = 0.0;
    for (var index = 1u; index < 9u; index = index + 1u) {
        let delta = f32(heights[base + index]) - center;
        higher_neighbor_delta = max(higher_neighbor_delta, delta);
        lower_neighbor_delta = max(lower_neighbor_delta, -delta);
    }
    let max_delta = max(higher_neighbor_delta, lower_neighbor_delta);
    if (max_delta <= max(threshold, 0.0)) {
        return vec2<f32>(0.0, 0.0);
    }
    let amount = clamp((max_delta - max(threshold, 0.0)) / 16.0, 0.0, 1.0);
    let pit_edge = higher_neighbor_delta >= lower_neighbor_delta;
    if (pit_edge) {
        return vec2<f32>(amount, amount * 0.25);
    }
    return vec2<f32>(amount * 0.45, amount);
}

fn block_boundary_relief(base: u32, threshold: f32, softness: f32) -> vec2<f32> {
    let center = f32(heights[base]);
    var higher_neighbor_delta = 0.0;
    var lower_neighbor_delta = 0.0;
    for (var index = 1u; index < 9u; index = index + 1u) {
        let delta = f32(heights[base + index]) - center;
        higher_neighbor_delta = max(higher_neighbor_delta, delta);
        lower_neighbor_delta = max(lower_neighbor_delta, -delta);
    }
    let max_delta = max(higher_neighbor_delta, lower_neighbor_delta);
    let safe_threshold = max(threshold, 0.0);
    if (max_delta <= safe_threshold) {
        return vec2<f32>(0.0, 0.0);
    }
    let delta = max_delta - safe_threshold;
    let amount = clamp(delta / (delta + max(softness, 0.001)), 0.0, 1.0);
    let pit_edge = higher_neighbor_delta >= lower_neighbor_delta;
    if (pit_edge) {
        return vec2<f32>(amount, amount * 0.25);
    }
    return vec2<f32>(amount * 0.45, amount);
}

fn block_boundary_line_factor(pixel_index: u32) -> f32 {
    if (params.blocks_per_pixel != 1u) {
        return -1.0;
    }
    if (params.pixels_per_block <= 1u) {
        return 1.0;
    }
    let pixels_per_block = f32(params.pixels_per_block);
    let pixel_x = pixel_index % params.width;
    let pixel_z = pixel_index / params.width;
    let local_x = f32(pixel_x % params.pixels_per_block) + 0.5;
    let local_z = f32(pixel_z % params.pixels_per_block) + 0.5;
    let edge_distance = min(min(local_x, pixels_per_block - local_x), min(local_z, pixels_per_block - local_z));
    let width = clamp(params.block_boundary_line_width_pixels, 0.25, max(pixels_per_block * 0.5, 0.25));
    let softness = clamp(params.block_boundary_softness, 0.25, 2.0);
    return clamp((width + softness - edge_distance) / softness, 0.0, 1.0);
}

fn terrain_lit_color(color: u32, pixel_index: u32) -> u32 {
    if (channel(color, 24u) == 0u || params.lighting_enabled == 0u) {
        return color;
    }
    let base = pixel_index * 9u;
    if (heights[base] == MISSING_HEIGHT) {
        return color;
    }
    let water_depth = water_depths[pixel_index];
    var normal_strength = max(params.normal_strength, 0.0);
    var shadow_strength = params.shadow_strength;
    var highlight_strength = params.highlight_strength;
    var ambient_occlusion = params.ambient_occlusion;
    var edge_relief_strength = max(params.edge_relief_strength, 0.0);
    if (water_depth > 0u) {
        if (params.underwater_relief_enabled == 0u) {
            return color;
        }
        let fade = underwater_depth_factor(water_depth);
        let minimum_light = clamp(params.underwater_min_light, 0.0, 1.0);
        let light_factor = max(fade, minimum_light);
        normal_strength = normal_strength * max(params.underwater_relief_strength, 0.0) * fade;
        shadow_strength = shadow_strength * light_factor;
        highlight_strength = highlight_strength * light_factor;
        ambient_occlusion = ambient_occlusion * fade;
        edge_relief_strength = edge_relief_strength * fade;
    }
    let max_shadow = select(max(params.max_shadow, 0.0), 70.0, water_depth > 0u);
    if (normal_strength == 0.0) {
        return color;
    }

    var dx = (
        f32(heights[base + 3u]) + 2.0 * f32(heights[base + 5u]) + f32(heights[base + 8u])
        - f32(heights[base + 1u]) - 2.0 * f32(heights[base + 4u]) - f32(heights[base + 6u])
    ) / 8.0;
    var dz = (
        f32(heights[base + 6u]) + 2.0 * f32(heights[base + 7u]) + f32(heights[base + 8u])
        - f32(heights[base + 1u]) - 2.0 * f32(heights[base + 2u]) - f32(heights[base + 3u])
    ) / 8.0;
    if (water_depth == 0u) {
        dx = compress_land_slope(dx, params.land_slope_softness);
        dz = compress_land_slope(dz, params.land_slope_softness);
    }
    dx = dx * normal_strength;
    dz = dz * normal_strength;
    let normal_length = max(sqrt(dx * dx + dz * dz + 4.0), 0.00000011920928955078125);
    let normal_x = -dx / normal_length;
    let normal_y = 2.0 / normal_length;
    let normal_z = -dz / normal_length;
    let dot_light = normal_x * params.light_x + normal_y * params.light_y + normal_z * params.light_z;
    let relief = min(abs(dx) + abs(dz), 24.0) / 24.0;
    let relative_light = dot_light - params.flat_dot;
    var factor: f32;
    if (relative_light >= 0.0) {
        factor = relative_light * highlight_strength * 100.0;
    } else {
        factor = relative_light * shadow_strength * 100.0;
    }
    factor = factor - relief * ambient_occlusion * 100.0;
    if (edge_relief_strength > 0.0) {
        let edge = edge_relief(base, params.edge_relief_threshold);
        let edge_shadow = edge.x * edge_relief_strength * max(params.edge_relief_max_shadow, 0.0);
        var edge_highlight = 0.0;
        if (relative_light >= 0.0) {
            edge_highlight = edge.y * edge_relief_strength * max(params.edge_relief_highlight, 0.0) * 100.0;
        }
        factor = factor + edge_highlight - edge_shadow;
    }
    if (params.block_boundary_enabled != 0u && params.block_boundary_strength > 0.0) {
        let line_factor = block_boundary_line_factor(pixel_index);
        if (line_factor >= 0.0) {
            let boundary = block_boundary_relief(
                base,
                params.block_boundary_height_threshold,
                params.block_boundary_softness
            );
            let water_factor = select(1.0, 0.45, water_depth > 0u);
            var flat_shadow = 0.0;
            if (params.pixels_per_block > 1u) {
                flat_shadow = line_factor * max(params.block_boundary_flat_strength, 0.0);
            }
            let edge_shadow = boundary.x * line_factor * max(params.block_boundary_strength, 0.0);
            let boundary_shadow = (flat_shadow + edge_shadow) * max(params.block_boundary_max_shadow, 0.0) * water_factor;
            let boundary_highlight = boundary.y
                * line_factor
                * max(params.block_boundary_strength, 0.0)
                * max(params.block_boundary_highlight_strength, 0.0)
                * 100.0
                * water_factor;
            factor = factor + boundary_highlight - boundary_shadow;
        }
    }
    let factor_i = i32(clamp(round(factor), -max_shadow, 55.0));
    return pack_color(
        shade_channel(channel(color, 0u), factor_i),
        shade_channel(channel(color, 8u), factor_i),
        shade_channel(channel(color, 16u), factor_i),
        channel(color, 24u),
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_index = global_id.x;
    if (pixel_index >= params.pixel_count) {
        return;
    }
    output_pixels[pixel_index] = terrain_lit_color(colors[pixel_index], pixel_index);
}
";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    pixel_count: u32,
    lighting_enabled: u32,
    normal_strength: f32,
    shadow_strength: f32,
    highlight_strength: f32,
    ambient_occlusion: f32,
    max_shadow: f32,
    land_slope_softness: f32,
    edge_relief_strength: f32,
    edge_relief_threshold: f32,
    edge_relief_max_shadow: f32,
    edge_relief_highlight: f32,
    underwater_relief_enabled: u32,
    _pad0: u32,
    underwater_relief_strength: f32,
    underwater_depth_fade: f32,
    underwater_min_light: f32,
    light_x: f32,
    light_y: f32,
    light_z: f32,
    flat_dot: f32,
    pixels_per_block: u32,
    blocks_per_pixel: u32,
    block_boundary_enabled: u32,
    _pad1: u32,
    block_boundary_strength: f32,
    block_boundary_flat_strength: f32,
    block_boundary_height_threshold: f32,
    block_boundary_max_shadow: f32,
    block_boundary_highlight_strength: f32,
    block_boundary_softness: f32,
    block_boundary_line_width_pixels: f32,
    _pad2: f32,
}

struct GpuRenderer {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    in_flight: Mutex<usize>,
    in_flight_available: Condvar,
    buffer_pool: Mutex<GpuBufferPool>,
}

pub(super) struct GpuComposeExecutor {
    renderer: Arc<GpuRenderer>,
    max_in_flight: usize,
    batch_size: usize,
    readback_workers: usize,
    submit_workers: usize,
    batch_pixels: usize,
}

struct GpuPermit {
    renderer: Arc<GpuRenderer>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuBufferKind {
    Storage,
    Staging,
}

struct PooledGpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
}

#[derive(Default)]
struct GpuBufferPool {
    storage: Vec<PooledGpuBuffer>,
    staging: Vec<PooledGpuBuffer>,
    storage_bytes: u64,
    staging_bytes: u64,
    storage_budget: u64,
    staging_budget: u64,
}

struct GpuBufferLease {
    buffer: Option<wgpu::Buffer>,
    size: u64,
    usage: wgpu::BufferUsages,
    kind: GpuBufferKind,
    renderer: Arc<GpuRenderer>,
}

static GPU_RENDERER: OnceLock<Result<Arc<GpuRenderer>, String>> = OnceLock::new();

pub(super) fn feature_enabled() -> bool {
    true
}

pub(super) fn compose_tile(
    input: &GpuTileComposeInput<'_>,
) -> Result<GpuTileComposeOutput, String> {
    validate_input(input)?;
    GpuComposeExecutor::new(input)?.compose_tile(input)
}

impl GpuComposeExecutor {
    fn new(input: &GpuTileComposeInput<'_>) -> Result<Self, String> {
        let renderer = renderer()?;
        renderer.configure_pool(input.buffer_pool_bytes, input.staging_pool_bytes)?;
        Ok(Self {
            renderer,
            max_in_flight: input.max_in_flight.max(1),
            batch_size: input.batch_size.max(1),
            readback_workers: input.readback_workers.max(1),
            submit_workers: input.submit_workers.max(1),
            batch_pixels: input.batch_pixels.max(1),
        })
    }

    fn compose_tile(
        &self,
        input: &GpuTileComposeInput<'_>,
    ) -> Result<GpuTileComposeOutput, String> {
        let renderer = &self.renderer;
        let batches = self.batch_count_for_tiles(1);
        let queue_wait_start = Instant::now();
        let _permit = renderer.acquire_permit(self.max_in_flight)?;
        let queue_wait_ms = queue_wait_start.elapsed().as_millis();
        let upload_start = Instant::now();
        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let (color_buffer, color_reused) = renderer.pooled_buffer_with_data(
            "bedrock-render gpu colors",
            storage_usage,
            input.colors,
        )?;
        let (height_buffer, height_reused) = renderer.pooled_buffer_with_data(
            "bedrock-render gpu heights",
            storage_usage,
            input.heights,
        )?;
        let (water_buffer, water_reused) = renderer.pooled_buffer_with_data(
            "bedrock-render gpu water depths",
            storage_usage,
            input.water_depths,
        )?;
        let params = params_for_input(input)?;
        let params_buffer = renderer.uniform_buffer("bedrock-render gpu params", &[params])?;
        let output_size = buffer_size_for_len(input.colors.len(), std::mem::size_of::<u32>())?;
        let (output_buffer, output_reused) = renderer.pooled_empty_buffer(
            "bedrock-render gpu output",
            GpuBufferKind::Storage,
            output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let (staging_buffer, staging_reused) = renderer.pooled_empty_buffer(
            "bedrock-render gpu readback",
            GpuBufferKind::Staging,
            output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )?;
        let bind_group_layout = renderer.pipeline.get_bind_group_layout(0);
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bedrock-render gpu compose bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: color_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: height_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: water_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let upload_ms = upload_start.elapsed().as_millis();

        let dispatch_start = Instant::now();
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bedrock-render gpu compose encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bedrock-render gpu compose pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&renderer.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = params.pixel_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        renderer.queue.submit([encoder.finish()]);
        let dispatch_ms = dispatch_start.elapsed().as_millis();

        let readback_start = Instant::now();
        let rgba = read_buffer(&renderer.device, &staging_buffer, output_size)?;
        let readback_ms = readback_start.elapsed().as_millis();
        let buffer_reuses = usize::from(color_reused)
            + usize::from(height_reused)
            + usize::from(water_reused)
            + usize::from(output_reused);
        let staging_reuses = usize::from(staging_reused);
        Ok(GpuTileComposeOutput {
            rgba,
            upload_ms,
            dispatch_ms,
            readback_ms,
            queue_wait_ms,
            batches,
            batch_tiles: 1,
            max_in_flight: self.max_in_flight,
            worker_threads: self.readback_workers,
            submit_workers: self.submit_workers,
            buffer_reuses,
            buffer_allocations: 4usize.saturating_sub(buffer_reuses),
            buffer_evictions: 0,
            staging_reuses,
            staging_allocations: 1usize.saturating_sub(staging_reuses),
            staging_evictions: 0,
            adapter_name: renderer.adapter_name.clone(),
        })
    }

    fn batch_count_for_tiles(&self, tile_count: usize) -> usize {
        tile_count
            .div_ceil(self.batch_size.max(1))
            .max(tile_count.div_ceil(self.batch_pixels.max(1)))
            .max(1)
    }
}

fn renderer() -> Result<Arc<GpuRenderer>, String> {
    GPU_RENDERER
        .get_or_init(|| GpuRenderer::new().map(Arc::new))
        .clone()
}

impl GpuRenderer {
    fn new() -> Result<Self, String> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|error| format!("no supported gpu adapter found: {error}"))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bedrock-render gpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| format!("failed to create gpu device: {error}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bedrock-render gpu compose shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bedrock-render gpu compose pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            _instance: instance,
            device,
            queue,
            pipeline,
            adapter_name: format!(
                "{} ({:?}, {:?})",
                adapter_info.name, adapter_info.backend, adapter_info.device_type
            ),
            in_flight: Mutex::new(0),
            in_flight_available: Condvar::new(),
            buffer_pool: Mutex::new(GpuBufferPool::default()),
        })
    }

    fn configure_pool(&self, storage_budget: usize, staging_budget: usize) -> Result<(), String> {
        let mut pool = self
            .buffer_pool
            .lock()
            .map_err(|_| "gpu buffer pool lock was poisoned".to_string())?;
        pool.storage_budget = u64::try_from(storage_budget)
            .map_err(|_| "gpu storage buffer pool budget exceeds u64".to_string())?;
        pool.staging_budget = u64::try_from(staging_budget)
            .map_err(|_| "gpu staging buffer pool budget exceeds u64".to_string())?;
        pool.evict_to_budgets();
        Ok(())
    }

    fn acquire_permit(self: &Arc<Self>, max_in_flight: usize) -> Result<GpuPermit, String> {
        let mut active = self
            .in_flight
            .lock()
            .map_err(|_| "gpu in-flight lock was poisoned".to_string())?;
        while *active >= max_in_flight {
            active = self
                .in_flight_available
                .wait(active)
                .map_err(|_| "gpu in-flight lock was poisoned".to_string())?;
        }
        *active = active.saturating_add(1);
        Ok(GpuPermit {
            renderer: Arc::clone(self),
        })
    }

    fn pooled_buffer_with_data<T: bytemuck::Pod>(
        self: &Arc<Self>,
        label: &'static str,
        usage: wgpu::BufferUsages,
        values: &[T],
    ) -> Result<(GpuBufferLease, bool), String> {
        let bytes = bytemuck::cast_slice(values);
        let size = buffer_size_for_bytes(bytes)?;
        let (buffer, reused) =
            self.pooled_empty_buffer(label, GpuBufferKind::Storage, size, usage)?;
        self.queue.write_buffer(&buffer, 0, bytes);
        Ok((buffer, reused))
    }

    fn pooled_empty_buffer(
        self: &Arc<Self>,
        label: &'static str,
        kind: GpuBufferKind,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<(GpuBufferLease, bool), String> {
        if let Some(pooled) = self.take_pooled_buffer(kind, size, usage)? {
            return Ok((
                GpuBufferLease {
                    buffer: Some(pooled.buffer),
                    size: pooled.size,
                    usage,
                    kind,
                    renderer: Arc::clone(self),
                },
                true,
            ));
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok((
            GpuBufferLease {
                buffer: Some(buffer),
                size,
                usage,
                kind,
                renderer: Arc::clone(self),
            },
            false,
        ))
    }

    fn take_pooled_buffer(
        &self,
        kind: GpuBufferKind,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<Option<PooledGpuBuffer>, String> {
        let mut pool = self
            .buffer_pool
            .lock()
            .map_err(|_| "gpu buffer pool lock was poisoned".to_string())?;
        Ok(pool.take(kind, size, usage))
    }

    fn uniform_buffer<T: bytemuck::Pod>(
        &self,
        label: &'static str,
        values: &[T],
    ) -> Result<wgpu::Buffer, String> {
        let bytes = bytemuck::cast_slice(values);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size_for_bytes(bytes)?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        Ok(buffer)
    }
}

impl GpuBufferPool {
    fn take(
        &mut self,
        kind: GpuBufferKind,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<PooledGpuBuffer> {
        let (buffers, bytes) = self.pool_parts_mut(kind);
        let index = buffers
            .iter()
            .position(|buffer| buffer.size >= size && buffer.usage == usage)?;
        let buffer = buffers.swap_remove(index);
        *bytes = bytes.saturating_sub(buffer.size);
        Some(buffer)
    }

    fn put(&mut self, kind: GpuBufferKind, buffer: PooledGpuBuffer) {
        let budget = self.budget(kind);
        if budget == 0 || buffer.size > budget {
            return;
        }
        let (buffers, bytes) = self.pool_parts_mut(kind);
        *bytes = bytes.saturating_add(buffer.size);
        buffers.push(buffer);
        self.evict(kind);
    }

    fn evict_to_budgets(&mut self) {
        self.evict(GpuBufferKind::Storage);
        self.evict(GpuBufferKind::Staging);
    }

    fn evict(&mut self, kind: GpuBufferKind) {
        let budget = self.budget(kind);
        let (buffers, bytes) = self.pool_parts_mut(kind);
        while *bytes > budget {
            let Some(buffer) = (!buffers.is_empty()).then(|| buffers.remove(0)) else {
                *bytes = 0;
                break;
            };
            *bytes = bytes.saturating_sub(buffer.size);
        }
    }

    fn budget(&self, kind: GpuBufferKind) -> u64 {
        match kind {
            GpuBufferKind::Storage => self.storage_budget,
            GpuBufferKind::Staging => self.staging_budget,
        }
    }

    fn pool_parts_mut(&mut self, kind: GpuBufferKind) -> (&mut Vec<PooledGpuBuffer>, &mut u64) {
        match kind {
            GpuBufferKind::Storage => (&mut self.storage, &mut self.storage_bytes),
            GpuBufferKind::Staging => (&mut self.staging, &mut self.staging_bytes),
        }
    }
}

impl Deref for GpuBufferLease {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("gpu buffer lease missing buffer")
    }
}

impl Drop for GpuBufferLease {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if let Ok(mut pool) = self.renderer.buffer_pool.lock() {
            pool.put(
                self.kind,
                PooledGpuBuffer {
                    buffer,
                    size: self.size,
                    usage: self.usage,
                },
            );
        }
    }
}

impl Drop for GpuPermit {
    fn drop(&mut self) {
        if let Ok(mut active) = self.renderer.in_flight.lock() {
            *active = active.saturating_sub(1);
            self.renderer.in_flight_available.notify_one();
        }
    }
}

fn validate_input(input: &GpuTileComposeInput<'_>) -> Result<(), String> {
    let pixel_count = usize::try_from(input.width)
        .ok()
        .and_then(|width| {
            usize::try_from(input.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| "gpu tile dimensions overflow".to_string())?;
    if input.colors.len() != pixel_count {
        return Err("gpu color buffer length does not match tile dimensions".to_string());
    }
    if input.water_depths.len() != pixel_count {
        return Err("gpu water-depth buffer length does not match tile dimensions".to_string());
    }
    if input.heights.len() != pixel_count.saturating_mul(9) {
        return Err(
            "gpu height-neighborhood buffer length does not match tile dimensions".to_string(),
        );
    }
    if input.pixels_per_block == 0 {
        return Err("gpu pixels_per_block must be greater than zero".to_string());
    }
    if input.blocks_per_pixel == 0 {
        return Err("gpu blocks_per_pixel must be greater than zero".to_string());
    }
    Ok(())
}

fn params_for_input(input: &GpuTileComposeInput<'_>) -> Result<GpuParams, String> {
    let pixel_count = input
        .colors
        .len()
        .try_into()
        .map_err(|_| "gpu tile pixel count exceeds u32".to_string())?;
    let lighting = input.lighting;
    let azimuth = lighting.light_azimuth_degrees.to_radians();
    let elevation = lighting
        .light_elevation_degrees
        .to_radians()
        .clamp(0.01, 1.55);
    let light_horizontal = elevation.cos();
    let light_x = azimuth.sin() * light_horizontal;
    let light_y = elevation.sin();
    let light_z = -azimuth.cos() * light_horizontal;
    let block_boundaries = input.block_boundaries;
    Ok(GpuParams {
        width: input.width,
        height: input.height,
        pixel_count,
        lighting_enabled: u32::from(input.lighting_enabled),
        normal_strength: lighting.normal_strength,
        shadow_strength: lighting.shadow_strength,
        highlight_strength: lighting.highlight_strength,
        ambient_occlusion: lighting.ambient_occlusion,
        max_shadow: lighting.max_shadow,
        land_slope_softness: lighting.land_slope_softness,
        edge_relief_strength: lighting.edge_relief_strength,
        edge_relief_threshold: lighting.edge_relief_threshold,
        edge_relief_max_shadow: lighting.edge_relief_max_shadow,
        edge_relief_highlight: lighting.edge_relief_highlight,
        underwater_relief_enabled: u32::from(lighting.underwater_relief_enabled),
        _pad0: 0,
        underwater_relief_strength: lighting.underwater_relief_strength,
        underwater_depth_fade: lighting.underwater_depth_fade,
        underwater_min_light: lighting.underwater_min_light,
        light_x,
        light_y,
        light_z,
        flat_dot: light_y,
        pixels_per_block: input.pixels_per_block,
        blocks_per_pixel: input.blocks_per_pixel,
        block_boundary_enabled: u32::from(block_boundaries.enabled),
        _pad1: 0,
        block_boundary_strength: block_boundaries.strength,
        block_boundary_flat_strength: block_boundaries.flat_strength,
        block_boundary_height_threshold: block_boundaries.height_threshold,
        block_boundary_max_shadow: block_boundaries.max_shadow,
        block_boundary_highlight_strength: block_boundaries.highlight_strength,
        block_boundary_softness: block_boundaries.softness,
        block_boundary_line_width_pixels: block_boundaries.line_width_pixels,
        _pad2: 0.0,
    })
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let send_result = sender.send(result.map_err(|error| error.to_string()));
        drop(send_result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("gpu device poll failed: {error}"))?;
    receiver
        .recv()
        .map_err(|error| format!("gpu readback channel failed: {error}"))?
        .map_err(|error| format!("gpu readback failed: {error}"))?;
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(bytes)
}

fn buffer_size_for_bytes(bytes: &[u8]) -> Result<u64, String> {
    u64::try_from(bytes.len().max(4)).map_err(|_| "gpu buffer size exceeds u64".to_string())
}

fn buffer_size_for_len(len: usize, element_size: usize) -> Result<u64, String> {
    let bytes = len
        .checked_mul(element_size)
        .ok_or_else(|| "gpu buffer size overflow".to_string())?;
    u64::try_from(bytes.max(4)).map_err(|_| "gpu buffer size exceeds u64".to_string())
}
