use bedrock_render::{
    ImageFormat, MapRenderer, RenderJob, RenderMode, RenderOptions, RenderPalette, TileCoord,
};
use bedrock_world::{BedrockLevelDbStorage, BedrockWorld, Dimension, OpenOptions};
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_world_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bedrock-world")
        .join("tests")
        .join("fixtures")
        .join("sample-bedrock-world")
}

#[test]
fn renders_fixture_biome_tile_as_rgba() {
    let world_path = fixture_world_path();
    if !world_path.join("db").join("CURRENT").exists() {
        return;
    }
    let storage = Arc::new(BedrockLevelDbStorage::open(world_path.join("db")).expect("open db"));
    let world = Arc::new(BedrockWorld::from_storage(
        world_path,
        storage,
        OpenOptions::default(),
    ));
    let renderer = MapRenderer::new(world, RenderPalette::default());
    let tile = renderer
        .render_tile_with_options_blocking(
            RenderJob {
                tile_size: 16,
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
                ..RenderOptions::default()
            },
        )
        .expect("render fixture biome tile");
    assert_eq!(tile.rgba.len(), 16 * 16 * 4);
    assert!(tile.rgba.chunks_exact(4).any(|pixel| pixel[3] != 0));
}
