use bedrock_render::{
    BakeOptions, ChunkRegion, ImageFormat, MapRenderer, RenderExecutionProfile, RenderJob,
    RenderLayout, RenderMemoryBudget, RenderMode, RenderOptions, RenderPalette,
    RenderThreadingOptions, TileCoord,
};
use bedrock_world::{BedrockLevelDbStorage, BedrockWorld, ChunkPos, Dimension, OpenOptions};
use criterion::{Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

fn fixture_world_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bedrock-world")
        .join("tests")
        .join("fixtures")
        .join("sample-bedrock-world")
}

fn renderer() -> Option<MapRenderer> {
    let world_path = fixture_world_path();
    if !world_path.join("db").join("CURRENT").exists() {
        return None;
    }
    let storage = Arc::new(BedrockLevelDbStorage::open(world_path.join("db")).ok()?);
    let world = Arc::new(BedrockWorld::from_storage(
        world_path,
        storage,
        OpenOptions::default(),
    ));
    Some(MapRenderer::new(world, RenderPalette::default()))
}

// Keep the related Criterion cases in one harness so shared fixture setup is measured consistently.
#[allow(clippy::too_many_lines)]
fn render_benches(c: &mut Criterion) {
    let Some(renderer) = renderer() else {
        return;
    };
    let coord = TileCoord {
        x: 0,
        z: 0,
        dimension: Dimension::Overworld,
    };
    c.bench_function("bedrock_render/biome_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::Biome { y: 64 }),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render biome tile");
        });
    });
    c.bench_function("bedrock_render/biome_tile_256_webp", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_blocking(RenderJob::new(coord, RenderMode::Biome { y: 64 }))
                .expect("render biome tile");
        });
    });
    c.bench_function("bedrock_render/fixed_y_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::LayerBlocks { y: 64 }),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render layer tile");
        });
    });
    c.bench_function("bedrock_render/raw_biome_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::RawBiomeLayer { y: 64 }),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render raw biome tile");
        });
    });
    c.bench_function("bedrock_render/surface_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::SurfaceBlocks),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render surface tile");
        });
    });
    c.bench_function("bedrock_render/bake_chunk_surface", |bench| {
        bench.iter(|| {
            renderer
                .bake_chunk_blocking(
                    ChunkPos {
                        x: 0,
                        z: 0,
                        dimension: Dimension::Overworld,
                    },
                    BakeOptions {
                        mode: RenderMode::SurfaceBlocks,
                        ..BakeOptions::default()
                    },
                )
                .expect("bake surface chunk");
        });
    });
    c.bench_function("bedrock_render/render_tile_surface_from_bake", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_from_bake_blocking(
                    RenderJob::new(coord, RenderMode::SurfaceBlocks),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        threading: RenderThreadingOptions::Auto,
                        ..RenderOptions::default()
                    },
                )
                .expect("render surface from bake");
        });
    });
    c.bench_function("bedrock_render/heightmap_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::HeightMap),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render heightmap tile");
        });
    });
    c.bench_function("bedrock_render/cave_slice_tile_256_rgba", |bench| {
        bench.iter(|| {
            renderer
                .render_tile_with_options_blocking(
                    RenderJob::new(coord, RenderMode::CaveSlice { y: 32 }),
                    &RenderOptions {
                        format: ImageFormat::Rgba,
                        ..RenderOptions::default()
                    },
                )
                .expect("render cave slice tile");
        });
    });
    let jobs = (0..4)
        .map(|x| {
            RenderJob::new(
                TileCoord {
                    x,
                    z: 0,
                    dimension: Dimension::Overworld,
                },
                RenderMode::Biome { y: 64 },
            )
        })
        .collect::<Vec<_>>();
    c.bench_function("bedrock_render/tile_batch_auto_threads", |bench| {
        bench.iter(|| {
            renderer
                .render_tiles_blocking(
                    jobs.clone(),
                    RenderOptions {
                        format: ImageFormat::Rgba,
                        threading: RenderThreadingOptions::Auto,
                        ..RenderOptions::default()
                    },
                )
                .expect("render batch");
        });
    });
    c.bench_function("bedrock_render/tile_batch_single_thread", |bench| {
        bench.iter(|| {
            renderer
                .render_tiles_blocking(
                    jobs.clone(),
                    RenderOptions {
                        format: ImageFormat::Rgba,
                        threading: RenderThreadingOptions::Single,
                        ..RenderOptions::default()
                    },
                )
                .expect("render batch");
        });
    });

    if std::env::var_os("BEDROCK_RENDER_FULL_BENCH").is_none() {
        return;
    }

    let web_region = ChunkRegion::new(Dimension::Overworld, 0, 0, 31, 31);
    let web_tiles = MapRenderer::plan_region_tiles(
        web_region,
        RenderMode::SurfaceBlocks,
        RenderLayout::default(),
    )
    .expect("plan web tiles");
    for threads in [1_usize, 8, 16, 32] {
        c.bench_function(
            &format!("bedrock_render/web_export_surface_threads_{threads}"),
            |bench| {
                bench.iter(|| {
                    renderer
                        .render_web_tiles_blocking(
                            &web_tiles,
                            RenderOptions {
                                format: ImageFormat::Rgba,
                                threading: if threads == 1 {
                                    RenderThreadingOptions::Single
                                } else {
                                    RenderThreadingOptions::Fixed(threads)
                                },
                                execution_profile: RenderExecutionProfile::Export,
                                memory_budget: RenderMemoryBudget::Disabled,
                                ..RenderOptions::default()
                            },
                            |_planned, _tile| Ok(()),
                        )
                        .expect("web export surface");
                });
            },
        );
    }
    c.bench_function("bedrock_render/global_chunk_bake_queue", |bench| {
        bench.iter(|| {
            renderer
                .render_web_tiles_blocking(
                    &web_tiles,
                    RenderOptions {
                        format: ImageFormat::Rgba,
                        threading: RenderThreadingOptions::Auto,
                        execution_profile: RenderExecutionProfile::Export,
                        memory_budget: RenderMemoryBudget::Disabled,
                        ..RenderOptions::default()
                    },
                    |_planned, _tile| Ok(()),
                )
                .expect("global bake queue");
        });
    });
    c.bench_function("bedrock_render/tile_compose_encode_pipeline", |bench| {
        bench.iter(|| {
            renderer
                .render_web_tiles_blocking(
                    &web_tiles,
                    RenderOptions {
                        format: ImageFormat::WebP,
                        threading: RenderThreadingOptions::Auto,
                        execution_profile: RenderExecutionProfile::Export,
                        memory_budget: RenderMemoryBudget::Disabled,
                        ..RenderOptions::default()
                    },
                    |_planned, _tile| Ok(()),
                )
                .expect("compose encode pipeline");
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    targets = render_benches
}
criterion_main!(benches);
