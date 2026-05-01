use bedrock_render::{BedrockRenderError, RenderPalette};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const BLOCK_JSON: &str = "data/colors/bedrock-block-color.json";
const BIOME_JSON: &str = "data/colors/bedrock-biome-color.json";
const CACHE_FILE: &str = "data/colors/bedrock-colors.brpal";
const PALETTE_VERSION: &str = "1.21.x";

type Result<T> = bedrock_render::Result<T>;

#[derive(Debug, Clone, Copy)]
enum Command {
    Audit,
    Normalize,
    RebuildCache,
}

#[derive(Debug)]
struct Config {
    command: Command,
    check: bool,
    block_json: PathBuf,
    biome_json: PathBuf,
    cache_file: PathBuf,
}

fn main() -> Result<()> {
    let config = Config::parse()?;
    match config.command {
        Command::Audit => audit(&config),
        Command::Normalize => normalize(&config),
        Command::RebuildCache => rebuild_cache(&config),
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let Some(command) = args.next() else {
            print_usage();
            return Err(validation("missing command"));
        };
        let command = match command.as_str() {
            "audit" => Command::Audit,
            "normalize" => Command::Normalize,
            "rebuild-cache" => Command::RebuildCache,
            "--help" | "-h" => {
                print_usage();
                return Ok(Self {
                    command: Command::Audit,
                    check: true,
                    block_json: PathBuf::from(BLOCK_JSON),
                    biome_json: PathBuf::from(BIOME_JSON),
                    cache_file: PathBuf::from(CACHE_FILE),
                });
            }
            other => {
                print_usage();
                return Err(validation(format!("unknown command: {other}")));
            }
        };

        let mut check = false;
        let mut block_json = PathBuf::from(BLOCK_JSON);
        let mut biome_json = PathBuf::from(BIOME_JSON);
        let mut cache_file = PathBuf::from(CACHE_FILE);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--check" => check = true,
                "--block-json" => {
                    block_json = PathBuf::from(next_arg(&mut args, "--block-json")?);
                }
                "--biome-json" => {
                    biome_json = PathBuf::from(next_arg(&mut args, "--biome-json")?);
                }
                "--cache" => {
                    cache_file = PathBuf::from(next_arg(&mut args, "--cache")?);
                }
                other => return Err(validation(format!("unknown argument: {other}"))),
            }
        }

        Ok(Self {
            command,
            check,
            block_json,
            biome_json,
            cache_file,
        })
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  cargo run --example palette_tool -- audit [--check]");
    eprintln!("  cargo run --example palette_tool -- normalize [--check]");
    eprintln!("  cargo run --example palette_tool -- rebuild-cache [--check]");
    eprintln!("Optional paths: --block-json <path> --biome-json <path> --cache <path>");
}

fn next_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| validation(format!("missing value for {name}")))
}

fn audit(config: &Config) -> Result<()> {
    let block = read_json(&config.block_json)?;
    let biome = read_json(&config.biome_json)?;
    let normalized_block = normalize_block_document(&block)?;
    let normalized_biome = normalize_biome_document(&biome)?;
    ensure_source_metadata(&block, "block")?;
    ensure_source_metadata(&biome, "biome")?;
    ensure_cache_matches_sources(&config.cache_file)?;
    if config.check {
        ensure_canonical(&config.block_json, &block, &normalized_block)?;
        ensure_canonical(&config.biome_json, &biome, &normalized_biome)?;
    }
    println!("palette audit ok");
    Ok(())
}

fn normalize(config: &Config) -> Result<()> {
    let block = read_json(&config.block_json)?;
    let biome = read_json(&config.biome_json)?;
    let normalized_block = normalize_block_document(&block)?;
    let normalized_biome = normalize_biome_document(&biome)?;
    if config.check {
        ensure_canonical(&config.block_json, &block, &normalized_block)?;
        ensure_canonical(&config.biome_json, &biome, &normalized_biome)?;
        println!("palette JSON is normalized");
        return Ok(());
    }
    write_json(&config.block_json, &normalized_block)?;
    write_json(&config.biome_json, &normalized_biome)?;
    println!("palette JSON normalized");
    Ok(())
}

fn rebuild_cache(config: &Config) -> Result<()> {
    if config.block_json != Path::new(BLOCK_JSON) || config.biome_json != Path::new(BIOME_JSON) {
        return Err(validation(
            "rebuild-cache uses the embedded built-in JSON sources; custom JSON paths are audit-only",
        ));
    }
    let bytes = RenderPalette::from_builtin_json_sources()?.to_binary_vec()?;
    if config.check {
        let existing = read_bytes(&config.cache_file)?;
        if existing != bytes {
            return Err(validation(format!(
                "{} is not in sync with built-in palette JSON sources",
                config.cache_file.display()
            )));
        }
        println!("palette cache is in sync");
        return Ok(());
    }
    fs::write(&config.cache_file, bytes).map_err(|error| {
        BedrockRenderError::io(
            format!("failed to write {}", config.cache_file.display()),
            error,
        )
    })?;
    println!("palette cache rebuilt: {}", config.cache_file.display());
    Ok(())
}

fn ensure_cache_matches_sources(cache_file: &Path) -> Result<()> {
    let bytes = RenderPalette::from_builtin_json_sources()?.to_binary_vec()?;
    let existing = read_bytes(cache_file)?;
    if existing != bytes {
        return Err(validation(format!(
            "{} is not in sync with built-in palette JSON sources",
            cache_file.display()
        )));
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path).map_err(|error| {
        BedrockRenderError::io(format!("failed to read {}", path.display()), error)
    })?;
    serde_json::from_str(&content)
        .map_err(|error| validation(format!("invalid JSON in {}: {error}", path.display())))
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|error| {
        BedrockRenderError::io(format!("failed to read {}", path.display()), error)
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, canonical_json(value)).map_err(|error| {
        BedrockRenderError::io(format!("failed to write {}", path.display()), error)
    })
}

fn ensure_canonical(path: &Path, original: &Value, normalized: &Value) -> Result<()> {
    if canonical_json(original) != canonical_json(normalized) {
        return Err(validation(format!(
            "{} is not normalized; run `cargo run --example palette_tool -- normalize`",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_source_metadata(value: &Value, label: &str) -> Result<()> {
    let Some(sources) = value.get("sources").and_then(Value::as_array) else {
        return Err(validation(format!(
            "{label} palette is missing sources metadata"
        )));
    };
    if sources.is_empty() {
        return Err(validation(format!(
            "{label} palette sources metadata is empty"
        )));
    }
    for source in sources {
        let Some(source) = source.as_object() else {
            return Err(validation(format!(
                "{label} palette source entry must be an object"
            )));
        };
        for key in ["id", "kind", "license", "retrieved_at", "usage"] {
            if !source.get(key).is_some_and(Value::is_string) {
                return Err(validation(format!(
                    "{label} palette source entry is missing string field `{key}`"
                )));
            }
        }
    }
    Ok(())
}

fn normalize_block_document(value: &Value) -> Result<Value> {
    let blocks = value.get("blocks").unwrap_or(value);
    let Some(blocks) = blocks.as_object() else {
        return Err(validation(
            "block palette JSON must contain an object `blocks` map",
        ));
    };
    let mut normalized_blocks = Map::new();
    let mut block_entries = blocks.iter().collect::<Vec<_>>();
    block_entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (block_name, entry) in block_entries {
        let Some(textures) = entry.as_object() else {
            return Err(validation(format!(
                "block `{block_name}` entry must be an object"
            )));
        };
        let mut normalized_textures = Map::new();
        let mut texture_entries = textures.iter().collect::<Vec<_>>();
        texture_entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (texture_name, color) in texture_entries {
            normalized_textures.insert(
                texture_name.clone(),
                color_array(color, &format!("{block_name}.{texture_name}"))?,
            );
        }
        normalized_blocks.insert(block_name.clone(), Value::Object(normalized_textures));
    }

    Ok(json!({
        "schema_version": 1,
        "minecraft_bedrock_version": minecraft_version(value),
        "sources": block_sources(),
        "blocks": normalized_blocks,
    }))
}

fn normalize_biome_document(value: &Value) -> Result<Value> {
    let biomes = value.get("biomes").unwrap_or(value);
    let Some(biomes) = biomes.as_object() else {
        return Err(validation(
            "biome palette JSON must contain an object `biomes` map",
        ));
    };
    let defaults_value = value
        .get("defaults")
        .or_else(|| biomes.get("default"))
        .ok_or_else(|| validation("biome palette JSON must contain `defaults`"))?;
    let defaults = normalize_biome_defaults(defaults_value)?;
    let mut normalized_biomes = Map::new();
    let mut ids = BTreeSet::new();
    let mut entries = biomes
        .iter()
        .filter(|(name, _)| name.as_str() != "default")
        .collect::<Vec<_>>();
    entries.sort_by(|(left_name, left), (right_name, right)| {
        let left_id = left.get("id").and_then(Value::as_u64).unwrap_or(u64::MAX);
        let right_id = right.get("id").and_then(Value::as_u64).unwrap_or(u64::MAX);
        left_id
            .cmp(&right_id)
            .then_with(|| left_name.cmp(right_name))
    });
    for (name, entry) in entries {
        let Some(map) = entry.as_object() else {
            return Err(validation(format!(
                "biome `{name}` entry must be an object"
            )));
        };
        let id = map
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| validation(format!("biome `{name}` is missing integer id")))?;
        let id = u32::try_from(id)
            .map_err(|_| validation(format!("biome `{name}` id is outside u32 range")))?;
        if !ids.insert(id) {
            return Err(validation(format!("duplicate biome id {id} at `{name}`")));
        }
        let mut biome = Map::new();
        biome.insert("id".to_string(), Value::from(id));
        biome.insert(
            "rgb".to_string(),
            color_array(
                map.get("rgb")
                    .ok_or_else(|| validation(format!("biome `{name}` is missing rgb")))?,
                &format!("{name}.rgb"),
            )?,
        );
        for key in ["grass", "leaves", "water"] {
            if let Some(color) = map.get(key) {
                biome.insert(
                    key.to_string(),
                    color_array(color, &format!("{name}.{key}"))?,
                );
            }
        }
        normalized_biomes.insert(name.clone(), Value::Object(biome));
    }

    Ok(json!({
        "schema_version": 1,
        "minecraft_bedrock_version": minecraft_version(value),
        "sources": biome_sources(),
        "defaults": defaults,
        "biomes": normalized_biomes,
    }))
}

fn normalize_biome_defaults(value: &Value) -> Result<Value> {
    let Some(map) = value.as_object() else {
        return Err(validation("biome defaults must be an object"));
    };
    let mut defaults = Map::new();
    for key in ["rgb", "grass", "leaves", "water"] {
        let Some(color) = map.get(key) else {
            return Err(validation(format!("biome defaults are missing `{key}`")));
        };
        defaults.insert(
            key.to_string(),
            color_array(color, &format!("defaults.{key}"))?,
        );
    }
    Ok(Value::Object(defaults))
}

fn color_array(value: &Value, label: &str) -> Result<Value> {
    let Some(channels) = value.as_array() else {
        return Err(validation(format!("{label} color must be an array")));
    };
    if !(channels.len() == 3 || channels.len() == 4) {
        return Err(validation(format!(
            "{label} color must have 3 or 4 channels"
        )));
    }
    let mut output = Vec::with_capacity(4);
    for channel in channels {
        let Some(channel) = channel.as_u64() else {
            return Err(validation(format!(
                "{label} color channel must be an integer"
            )));
        };
        let channel = u8::try_from(channel)
            .map_err(|_| validation(format!("{label} color channel is outside 0..=255")))?;
        output.push(Value::from(channel));
    }
    if output.len() == 3 {
        output.push(Value::from(255_u8));
    }
    Ok(Value::Array(output))
}

fn canonical_json(value: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(value).expect("palette JSON should serialize")
    )
}

fn minecraft_version(value: &Value) -> String {
    value
        .get("minecraft_bedrock_version")
        .and_then(Value::as_str)
        .unwrap_or(PALETTE_VERSION)
        .to_string()
}

fn block_sources() -> Value {
    json!([
        {
            "id": "legacy-current",
            "kind": "legacy-current",
            "description": "Project-maintained colors carried forward from the pre-schema bedrock-render palette; used where public references do not provide a directly reusable color table.",
            "license": "MIT OR Apache-2.0 as part of this repository",
            "retrieved_at": "2026-05-01",
            "usage": "color-values"
        },
        {
            "id": "microsoft-learn-map-color",
            "kind": "public-reference",
            "name": "Microsoft Learn minecraft:map_color component reference",
            "url": "https://learn.microsoft.com/minecraft/creator/reference/content/blockreference/examples/blockcomponents/minecraftblock_map_color?view=minecraft-bedrock-experimental",
            "license": "Microsoft Learn documentation terms",
            "retrieved_at": "2026-05-01",
            "usage": "semantics-reference"
        },
        {
            "id": "microsoft-minecraft-samples",
            "kind": "public-reference",
            "name": "Microsoft Minecraft samples repository",
            "url": "https://github.com/microsoft/minecraft-samples",
            "license": "Repository license and notices apply",
            "retrieved_at": "2026-05-01",
            "usage": "schema-and-pack-reference"
        }
    ])
}

fn biome_sources() -> Value {
    json!([
        {
            "id": "legacy-current",
            "kind": "legacy-current",
            "description": "Project-maintained colors carried forward from the pre-schema bedrock-render palette; used where public references do not provide a directly reusable color table.",
            "license": "MIT OR Apache-2.0 as part of this repository",
            "retrieved_at": "2026-05-01",
            "usage": "color-values"
        },
        {
            "id": "mojang-bedrock-samples-client-biomes",
            "kind": "public-reference",
            "name": "Mojang Bedrock samples client biome reference",
            "url": "https://mojang.github.io/bedrock-samples/Client%20Biomes.html",
            "license": "Documentation/sample terms apply",
            "retrieved_at": "2026-05-01",
            "usage": "biome-schema-reference"
        },
        {
            "id": "microsoft-minecraft-samples",
            "kind": "public-reference",
            "name": "Microsoft Minecraft samples repository",
            "url": "https://github.com/microsoft/minecraft-samples",
            "license": "Repository license and notices apply",
            "retrieved_at": "2026-05-01",
            "usage": "schema-and-pack-reference"
        }
    ])
}

fn validation(message: impl Into<String>) -> BedrockRenderError {
    BedrockRenderError::Validation(message.into())
}
