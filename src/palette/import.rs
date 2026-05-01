use crate::{BedrockRenderError, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;

const PALETTE_CACHE_MAGIC: &[u8; 8] = b"BRPAL01\0";
const BUILTIN_BLOCK_COLOR_JSON: &str = include_str!("../../data/colors/bedrock-block-color.json");
const BUILTIN_BIOME_COLOR_JSON: &str = include_str!("../../data/colors/bedrock-biome-color.json");
const BUILTIN_PALETTE_CACHE_BYTES: &[u8] = include_bytes!("../../data/colors/bedrock-colors.brpal");

/// An 8-bit RGBA color used by palettes and decoded render planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbaColor {
    /// Red channel.
    pub red: u8,
    /// Green channel.
    pub green: u8,
    /// Blue channel.
    pub blue: u8,
    /// Alpha channel.
    pub alpha: u8,
}

impl RgbaColor {
    /// Creates a new color from red, green, blue, and alpha channels.
    #[must_use]
    pub const fn new(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Returns the color as `[red, green, blue, alpha]`.
    #[must_use]
    pub const fn to_array(self) -> [u8; 4] {
        [self.red, self.green, self.blue, self.alpha]
    }
}

/// Import counters returned after merging palette data.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PaletteImportReport {
    /// Number of block colors imported or overwritten.
    pub block_colors: usize,
    /// Number of biome colors imported or overwritten.
    pub biome_colors: usize,
    /// Number of entries skipped because they were incomplete or unsupported.
    pub skipped_entries: usize,
}

/// Color palette used by block, biome, surface, height, and cave render modes.
#[derive(Debug, Clone)]
pub struct RenderPalette {
    biome_colors: HashMap<u32, RgbaColor>,
    biome_grass_colors: HashMap<u32, RgbaColor>,
    biome_foliage_colors: HashMap<u32, RgbaColor>,
    biome_water_colors: HashMap<u32, RgbaColor>,
    block_colors: HashMap<String, RgbaColor>,
    unknown_biome_color: RgbaColor,
    unknown_block_color: RgbaColor,
    missing_chunk_color: RgbaColor,
    void_color: RgbaColor,
    air_color: RgbaColor,
    default_grass_color: RgbaColor,
    default_foliage_color: RgbaColor,
    default_water_color: RgbaColor,
    cave_air_color: RgbaColor,
    cave_solid_color: RgbaColor,
    cave_water_color: RgbaColor,
    cave_lava_color: RgbaColor,
    min_height_color: RgbaColor,
    max_height_color: RgbaColor,
}

impl Default for RenderPalette {
    fn default() -> Self {
        let mut palette = Self {
            biome_colors: HashMap::new(),
            biome_grass_colors: HashMap::new(),
            biome_foliage_colors: HashMap::new(),
            biome_water_colors: HashMap::new(),
            block_colors: HashMap::new(),
            unknown_biome_color: RgbaColor::new(255, 0, 255, 180),
            unknown_block_color: RgbaColor::new(255, 0, 255, 255),
            missing_chunk_color: RgbaColor::new(0, 0, 0, 0),
            void_color: RgbaColor::new(0, 0, 0, 0),
            air_color: RgbaColor::new(0, 0, 0, 0),
            default_grass_color: RgbaColor::new(142, 185, 113, 255),
            default_foliage_color: RgbaColor::new(113, 167, 77, 255),
            default_water_color: RgbaColor::new(63, 118, 228, 255),
            cave_air_color: RgbaColor::new(12, 12, 14, 255),
            cave_solid_color: RgbaColor::new(116, 116, 116, 255),
            cave_water_color: RgbaColor::new(38, 82, 180, 255),
            cave_lava_color: RgbaColor::new(255, 92, 12, 255),
            min_height_color: RgbaColor::new(36, 52, 100, 255),
            max_height_color: RgbaColor::new(242, 244, 232, 255),
        };
        palette.insert_default_biomes();
        palette.insert_default_blocks();
        if let Err(error) = palette.merge_builtin_cache() {
            panic!("embedded bedrock-render palette cache is invalid: {error}");
        }
        if let Err(error) = palette.merge_json_str(BUILTIN_BIOME_COLOR_JSON) {
            panic!("embedded bedrock-render biome palette is invalid: {error}");
        }
        palette
    }
}

impl RenderPalette {
    /// Creates the default embedded palette.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the embedded palette from the auditable JSON sources only.
    ///
    /// This bypasses the embedded `BRPAL01` cache and is intended for cache
    /// rebuild tools, source-data audits, and tests that need to prove the JSON
    /// sources and binary cache describe the same palette.
    ///
    /// # Errors
    ///
    /// Returns an error if either embedded JSON source is invalid.
    pub fn from_builtin_json_sources() -> Result<Self> {
        let mut palette = Self {
            biome_colors: HashMap::new(),
            biome_grass_colors: HashMap::new(),
            biome_foliage_colors: HashMap::new(),
            biome_water_colors: HashMap::new(),
            block_colors: HashMap::new(),
            unknown_biome_color: RgbaColor::new(255, 0, 255, 180),
            unknown_block_color: RgbaColor::new(255, 0, 255, 255),
            missing_chunk_color: RgbaColor::new(0, 0, 0, 0),
            void_color: RgbaColor::new(0, 0, 0, 0),
            air_color: RgbaColor::new(0, 0, 0, 0),
            default_grass_color: RgbaColor::new(142, 185, 113, 255),
            default_foliage_color: RgbaColor::new(113, 167, 77, 255),
            default_water_color: RgbaColor::new(63, 118, 228, 255),
            cave_air_color: RgbaColor::new(12, 12, 14, 255),
            cave_solid_color: RgbaColor::new(116, 116, 116, 255),
            cave_water_color: RgbaColor::new(38, 82, 180, 255),
            cave_lava_color: RgbaColor::new(255, 92, 12, 255),
            min_height_color: RgbaColor::new(36, 52, 100, 255),
            max_height_color: RgbaColor::new(242, 244, 232, 255),
        };
        palette.insert_default_biomes();
        palette.insert_default_blocks();
        palette.merge_json_str(BUILTIN_BLOCK_COLOR_JSON)?;
        palette.merge_json_str(BUILTIN_BIOME_COLOR_JSON)?;
        Ok(palette)
    }

    /// Returns the embedded block-color JSON source.
    #[must_use]
    pub fn builtin_block_color_json() -> &'static str {
        BUILTIN_BLOCK_COLOR_JSON
    }

    /// Returns the embedded biome-color JSON source.
    #[must_use]
    pub fn builtin_biome_color_json() -> &'static str {
        BUILTIN_BIOME_COLOR_JSON
    }

    /// Returns the embedded compact binary palette cache.
    #[must_use]
    pub fn builtin_palette_cache_bytes() -> &'static [u8] {
        BUILTIN_PALETTE_CACHE_BYTES
    }

    /// Merges the embedded binary palette cache into this palette.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded cache bytes fail validation.
    pub fn merge_builtin_cache(&mut self) -> Result<PaletteImportReport> {
        self.merge_binary_slice(BUILTIN_PALETTE_CACHE_BYTES)
    }

    /// Adds or replaces a biome color and returns the updated palette.
    #[must_use]
    pub fn with_biome_color(mut self, id: u32, color: RgbaColor) -> Self {
        self.biome_colors.insert(id, color);
        self
    }

    /// Adds or replaces a block color and returns the updated palette.
    #[must_use]
    pub fn with_block_color(mut self, name: impl Into<String>, color: RgbaColor) -> Self {
        self.block_colors.insert(name.into(), color);
        self
    }

    /// Changes the fallback color used for unknown biome IDs.
    #[must_use]
    pub fn with_unknown_biome_color(mut self, color: RgbaColor) -> Self {
        self.unknown_biome_color = color;
        self
    }

    /// Changes the fallback color used for unknown block names.
    #[must_use]
    pub fn with_unknown_block_color(mut self, color: RgbaColor) -> Self {
        self.unknown_block_color = color;
        self
    }

    /// Changes the low and high colors used by height-map rendering.
    #[must_use]
    pub fn with_height_gradient(mut self, min_color: RgbaColor, max_color: RgbaColor) -> Self {
        self.min_height_color = min_color;
        self.max_height_color = max_color;
        self
    }

    /// Changes cave diagnostic colors for air, solid, water, and lava.
    #[must_use]
    pub fn with_cave_colors(
        mut self,
        air: RgbaColor,
        solid: RgbaColor,
        water: RgbaColor,
        lava: RgbaColor,
    ) -> Self {
        self.cave_air_color = air;
        self.cave_solid_color = solid;
        self.cave_water_color = water;
        self.cave_lava_color = lava;
        self
    }

    /// Merges palette entries from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the JSON shape is invalid.
    pub fn merge_json_file(&mut self, path: impl AsRef<Path>) -> Result<PaletteImportReport> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|error| {
            BedrockRenderError::io(
                format!("failed to read palette JSON {}", path.display()),
                error,
            )
        })?;
        self.merge_json_str(&content)
    }

    /// Merges palette entries from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON cannot be parsed or contains invalid required fields.
    pub fn merge_json_str(&mut self, content: &str) -> Result<PaletteImportReport> {
        let value = serde_json::from_str::<Value>(content).map_err(|error| {
            BedrockRenderError::Validation(format!("invalid palette JSON: {error}"))
        })?;
        let mut report = PaletteImportReport::default();
        self.merge_json_value(&value, &mut report)?;
        Ok(report)
    }

    /// Merges palette entries from a compact binary cache file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the cache bytes are invalid.
    pub fn merge_binary_file(&mut self, path: impl AsRef<Path>) -> Result<PaletteImportReport> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            BedrockRenderError::io(
                format!("failed to read palette cache {}", path.display()),
                error,
            )
        })?;
        self.merge_binary_slice(&bytes)
    }

    /// Merges palette entries from compact `BRPAL01` cache bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache header, counts, names, or color records are invalid.
    pub fn merge_binary_slice(&mut self, bytes: &[u8]) -> Result<PaletteImportReport> {
        let mut cursor = Cursor::new(bytes);
        let mut magic = [0_u8; 8];
        cursor.read_exact(&mut magic).map_err(|error| {
            BedrockRenderError::Validation(format!("invalid palette cache header: {error}"))
        })?;
        if &magic != PALETTE_CACHE_MAGIC {
            return Err(BedrockRenderError::Validation(
                "invalid palette cache magic".to_string(),
            ));
        }

        let mut report = PaletteImportReport::default();
        let biome_count = read_u32(&mut cursor, "biome count")?;
        for _ in 0..biome_count {
            let id = read_u32(&mut cursor, "biome id")?;
            let color = read_rgba(&mut cursor, "biome color")?;
            self.biome_colors.insert(id, color);
            report.biome_colors += 1;
        }

        let block_count = read_u32(&mut cursor, "block count")?;
        for _ in 0..block_count {
            let name_len = usize::from(read_u16(&mut cursor, "block name length")?);
            let mut name_bytes = vec![0_u8; name_len];
            cursor.read_exact(&mut name_bytes).map_err(|error| {
                BedrockRenderError::Validation(format!(
                    "invalid block name in palette cache: {error}"
                ))
            })?;
            let name = String::from_utf8(name_bytes).map_err(|error| {
                BedrockRenderError::Validation(format!("invalid UTF-8 block name: {error}"))
            })?;
            let color = read_rgba(&mut cursor, "block color")?;
            self.block_colors.insert(name, color);
            report.block_colors += 1;
        }
        Ok(report)
    }

    /// Writes this palette as a compact binary cache file.
    ///
    /// # Errors
    ///
    /// Returns an error if the palette cannot be encoded or the file cannot be written.
    pub fn write_binary_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_binary_vec()?;
        fs::write(path, bytes).map_err(|error| {
            BedrockRenderError::io(
                format!("failed to write palette cache {}", path.display()),
                error,
            )
        })
    }

    /// Encodes this palette as compact `BRPAL01` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the palette contains too many entries or an oversized block name.
    pub fn to_binary_vec(&self) -> Result<Vec<u8>> {
        let biome_count = u32::try_from(self.biome_colors.len()).map_err(|_| {
            BedrockRenderError::Validation("too many biome colors for palette cache".to_string())
        })?;
        let block_count = u32::try_from(self.block_colors.len()).map_err(|_| {
            BedrockRenderError::Validation("too many block colors for palette cache".to_string())
        })?;
        let mut bytes = Vec::with_capacity(16 + self.block_colors.len() * 40);
        bytes.extend_from_slice(PALETTE_CACHE_MAGIC);
        bytes.extend_from_slice(&biome_count.to_le_bytes());
        let mut biome_colors = self.biome_colors.iter().collect::<Vec<_>>();
        biome_colors.sort_by_key(|(id, _)| **id);
        for (id, color) in biome_colors {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&color.to_array());
        }
        bytes.extend_from_slice(&block_count.to_le_bytes());
        let mut block_colors = self.block_colors.iter().collect::<Vec<_>>();
        block_colors.sort_by(|left, right| left.0.cmp(right.0));
        for (name, color) in block_colors {
            let name_len = u16::try_from(name.len()).map_err(|_| {
                BedrockRenderError::Validation(format!(
                    "block name too long for palette cache: {name}"
                ))
            })?;
            bytes.extend_from_slice(&name_len.to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&color.to_array());
        }
        Ok(bytes)
    }

    /// Returns the configured biome color or the unknown-biome fallback.
    #[must_use]
    pub fn biome_color(&self, id: u32) -> RgbaColor {
        self.biome_colors
            .get(&id)
            .copied()
            .unwrap_or(self.unknown_biome_color)
    }

    /// Returns a deterministic diagnostic color for known biome IDs.
    #[must_use]
    pub fn raw_biome_color(&self, id: u32) -> RgbaColor {
        if self.biome_colors.contains_key(&id) {
            return biome_hash_color(id);
        }
        self.unknown_biome_color
    }

    /// Returns whether this palette has an explicit color for the biome ID.
    #[must_use]
    pub fn has_biome_color(&self, id: u32) -> bool {
        self.biome_colors.contains_key(&id)
    }

    /// Returns the configured block color or the unknown-block fallback.
    #[must_use]
    pub fn block_color(&self, name: &str) -> RgbaColor {
        if is_air_block(name) {
            return self.air_color;
        }
        self.find_block_color(name)
            .unwrap_or(self.unknown_block_color)
    }

    /// Returns a surface block color with optional biome tinting applied.
    #[must_use]
    pub fn surface_block_color(
        &self,
        name: &str,
        biome_id: Option<u32>,
        biome_tint: bool,
    ) -> RgbaColor {
        let color = self.block_color(name);
        if !biome_tint {
            return with_alpha(color, 255);
        }
        if is_grass_tinted_block(name) {
            return multiply_with_biome_tint(
                color,
                self.biome_grass_tint(biome_id),
                self.default_grass_color,
            );
        }
        if is_foliage_tinted_block(name) {
            return multiply_with_biome_tint(
                color,
                self.biome_foliage_tint(biome_id),
                self.default_foliage_color,
            );
        }
        if is_water_block(name) {
            return multiply_with_biome_tint(
                color,
                self.biome_water_tint(biome_id),
                self.default_water_color,
            );
        }
        with_alpha(color, 255)
    }

    /// Blends a transparent water color over the block below it.
    #[must_use]
    pub fn transparent_water_color(
        &self,
        water_name: &str,
        under_name: Option<&str>,
        biome_id: Option<u32>,
        depth: u8,
        biome_tint: bool,
    ) -> RgbaColor {
        let water = self.surface_block_color(water_name, biome_id, biome_tint);
        let under = under_name.map_or(self.missing_chunk_color, |name| {
            self.surface_block_color(name, biome_id, biome_tint)
        });
        let water_alpha = u8_from_u16((u16::from(depth.min(7)) * 25).min(179));
        alpha_blend(with_alpha(water, water_alpha), under)
    }

    /// Applies simple relative height shading to a color.
    #[must_use]
    pub fn height_shaded_color(&self, color: RgbaColor, current: i16, reference: i16) -> RgbaColor {
        if color.alpha == 0 {
            return color;
        }
        let delta = i32::from(current) - i32::from(reference);
        if delta == 0 {
            return color;
        }
        let factor = (delta.clamp(-8, 8) * 5).clamp(-40, 40);
        shade_color(color, factor)
    }

    /// Applies normal-style shading from west/east/north/south height samples.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn height_normal_shaded_color(
        &self,
        color: RgbaColor,
        west: i16,
        east: i16,
        north: i16,
        south: i16,
    ) -> RgbaColor {
        if color.alpha == 0 {
            return color;
        }
        let dx = (f32::from(east) - f32::from(west)) * 1.2;
        let dz = (f32::from(south) - f32::from(north)) * 1.2;
        let normal_length = (dx.mul_add(dx, dz.mul_add(dz, 4.0)))
            .sqrt()
            .max(f32::EPSILON);
        let normal_x = -dx / normal_length;
        let normal_y = 2.0 / normal_length;
        let normal_z = -dz / normal_length;
        let azimuth = 315.0_f32.to_radians();
        let elevation = 45.0_f32.to_radians();
        let light_horizontal = elevation.cos();
        let light_x = azimuth.sin() * light_horizontal;
        let light_y = elevation.sin();
        let light_z = -azimuth.cos() * light_horizontal;
        let dot = normal_x.mul_add(light_x, normal_y.mul_add(light_y, normal_z * light_z));
        let relief = (dx.abs() + dz.abs()).min(24.0) / 24.0;
        let relative_light = dot - light_y;
        let mut factor = if relative_light >= 0.0 {
            relative_light * 35.0
        } else {
            relative_light * 55.0
        };
        factor -= relief * 8.0;
        let factor = factor.round().clamp(-70.0, 55.0) as i32;
        if factor == 0 {
            color
        } else {
            shade_color(color, factor)
        }
    }

    /// Returns whether a block name is recognized by exact or category color lookup.
    #[must_use]
    pub fn has_block_color(&self, name: &str) -> bool {
        is_air_block(name) || self.find_block_color(name).is_some()
    }

    /// Returns whether the name is treated as air.
    #[must_use]
    pub fn is_air_block(&self, name: &str) -> bool {
        is_air_block(name)
    }

    /// Returns whether the name is treated as water.
    #[must_use]
    pub fn is_water_block(&self, name: &str) -> bool {
        is_water_block(name)
    }

    #[must_use]
    fn biome_grass_tint(&self, biome_id: Option<u32>) -> Option<RgbaColor> {
        biome_id.and_then(|id| self.biome_grass_colors.get(&id).copied())
    }

    #[must_use]
    fn biome_foliage_tint(&self, biome_id: Option<u32>) -> Option<RgbaColor> {
        biome_id.and_then(|id| self.biome_foliage_colors.get(&id).copied())
    }

    #[must_use]
    fn biome_water_tint(&self, biome_id: Option<u32>) -> Option<RgbaColor> {
        biome_id.and_then(|id| self.biome_water_colors.get(&id).copied())
    }

    /// Returns the transparent color used for missing chunks.
    #[must_use]
    pub const fn missing_chunk_color(&self) -> RgbaColor {
        self.missing_chunk_color
    }

    /// Returns the transparent color used for void or empty space.
    #[must_use]
    pub const fn void_color(&self) -> RgbaColor {
        self.void_color
    }

    /// Returns the fallback color used for unknown biome IDs.
    #[must_use]
    pub const fn unknown_biome_color(&self) -> RgbaColor {
        self.unknown_biome_color
    }

    /// Returns the fallback color used for unknown block names.
    #[must_use]
    pub const fn unknown_block_color(&self) -> RgbaColor {
        self.unknown_block_color
    }

    /// Returns the height-gradient color for a height within a world range.
    #[must_use]
    pub fn height_color(&self, height: i16, min_height: i16, max_height: i16) -> RgbaColor {
        if min_height >= max_height {
            return self.max_height_color;
        }
        let numerator = i32::from(height.saturating_sub(min_height))
            .clamp(0, i32::from(max_height.saturating_sub(min_height)));
        let denominator = i32::from(max_height.saturating_sub(min_height)).max(1);
        lerp_color(
            self.min_height_color,
            self.max_height_color,
            numerator,
            denominator,
        )
    }

    /// Returns the cave diagnostic color for an optional block name.
    #[must_use]
    pub fn cave_color(&self, block_name: Option<&str>) -> RgbaColor {
        let Some(block_name) = block_name else {
            return self.cave_air_color;
        };
        if is_air_block(block_name) {
            return self.cave_air_color;
        }
        if block_name.contains("water") {
            return self.cave_water_color;
        }
        if block_name.contains("lava") {
            return self.cave_lava_color;
        }
        self.cave_solid_color
    }

    fn insert_default_biomes(&mut self) {
        for (id, color) in [
            (0, RgbaColor::new(141, 179, 96, 255)),
            (1, RgbaColor::new(250, 148, 24, 255)),
            (2, RgbaColor::new(250, 240, 192, 255)),
            (3, RgbaColor::new(96, 96, 96, 255)),
            (4, RgbaColor::new(5, 102, 33, 255)),
            (5, RgbaColor::new(11, 102, 89, 255)),
            (6, RgbaColor::new(7, 249, 178, 255)),
            (7, RgbaColor::new(0, 0, 255, 255)),
            (8, RgbaColor::new(255, 0, 0, 255)),
            (9, RgbaColor::new(128, 128, 255, 255)),
            (10, RgbaColor::new(160, 160, 255, 255)),
            (11, RgbaColor::new(255, 255, 255, 255)),
            (12, RgbaColor::new(160, 160, 160, 255)),
            (13, RgbaColor::new(255, 255, 160, 255)),
            (14, RgbaColor::new(0, 160, 0, 255)),
            (15, RgbaColor::new(255, 200, 128, 255)),
            (16, RgbaColor::new(255, 220, 160, 255)),
            (17, RgbaColor::new(48, 116, 68, 255)),
            (18, RgbaColor::new(27, 82, 54, 255)),
            (19, RgbaColor::new(89, 102, 81, 255)),
            (20, RgbaColor::new(69, 79, 62, 255)),
            (21, RgbaColor::new(80, 112, 80, 255)),
            (22, RgbaColor::new(129, 161, 129, 255)),
            (23, RgbaColor::new(91, 95, 69, 255)),
            (24, RgbaColor::new(30, 144, 255, 255)),
            (25, RgbaColor::new(98, 140, 120, 255)),
            (26, RgbaColor::new(112, 141, 129, 255)),
            (27, RgbaColor::new(160, 167, 140, 255)),
            (28, RgbaColor::new(119, 156, 98, 255)),
            (29, RgbaColor::new(180, 180, 180, 255)),
            (30, RgbaColor::new(160, 160, 180, 255)),
            (31, RgbaColor::new(40, 60, 40, 255)),
            (32, RgbaColor::new(50, 70, 50, 255)),
            (33, RgbaColor::new(189, 178, 95, 255)),
            (34, RgbaColor::new(167, 157, 100, 255)),
            (35, RgbaColor::new(120, 120, 120, 255)),
            (36, RgbaColor::new(80, 80, 80, 255)),
            (37, RgbaColor::new(90, 120, 80, 255)),
            (38, RgbaColor::new(130, 130, 100, 255)),
            (39, RgbaColor::new(85, 107, 47, 255)),
            (40, RgbaColor::new(255, 255, 255, 255)),
            (41, RgbaColor::new(0, 0, 172, 255)),
            (42, RgbaColor::new(45, 85, 180, 255)),
            (43, RgbaColor::new(32, 70, 150, 255)),
            (44, RgbaColor::new(32, 85, 150, 255)),
            (45, RgbaColor::new(28, 65, 130, 255)),
            (46, RgbaColor::new(90, 160, 190, 255)),
            (47, RgbaColor::new(70, 120, 160, 255)),
            (48, RgbaColor::new(80, 150, 70, 255)),
            (49, RgbaColor::new(55, 120, 55, 255)),
            (50, RgbaColor::new(189, 178, 95, 255)),
            (81, RgbaColor::new(70, 100, 35, 255)),
            (82, RgbaColor::new(80, 85, 38, 255)),
            (129, RgbaColor::new(220, 220, 220, 255)),
            (130, RgbaColor::new(255, 188, 64, 255)),
            (131, RgbaColor::new(80, 80, 80, 255)),
            (132, RgbaColor::new(34, 139, 34, 255)),
            (133, RgbaColor::new(20, 120, 95, 255)),
            (134, RgbaColor::new(35, 92, 70, 255)),
            (140, RgbaColor::new(200, 220, 255, 255)),
            (149, RgbaColor::new(30, 120, 30, 255)),
            (151, RgbaColor::new(105, 130, 105, 255)),
            (155, RgbaColor::new(120, 145, 90, 255)),
            (156, RgbaColor::new(175, 175, 175, 255)),
            (157, RgbaColor::new(110, 110, 130, 255)),
            (158, RgbaColor::new(45, 70, 45, 255)),
            (160, RgbaColor::new(177, 170, 90, 255)),
            (161, RgbaColor::new(96, 120, 70, 255)),
            (162, RgbaColor::new(178, 164, 90, 255)),
            (163, RgbaColor::new(160, 140, 80, 255)),
            (164, RgbaColor::new(140, 105, 65, 255)),
            (165, RgbaColor::new(175, 120, 80, 255)),
            (166, RgbaColor::new(150, 90, 70, 255)),
            (167, RgbaColor::new(155, 120, 100, 255)),
            (168, RgbaColor::new(60, 150, 80, 255)),
            (169, RgbaColor::new(40, 120, 65, 255)),
        ] {
            self.biome_colors.insert(id, color);
        }
    }

    fn merge_json_value(&mut self, value: &Value, report: &mut PaletteImportReport) -> Result<()> {
        match value {
            Value::Object(map) => {
                if let Some(defaults) = map.get("defaults") {
                    self.merge_biome_defaults(defaults);
                }
                if let Some(blocks) = map.get("blocks").or_else(|| map.get("block_colors")) {
                    self.merge_block_json_value(blocks, report)?;
                }
                if let Some(biomes) = map.get("biomes").or_else(|| map.get("biome_colors")) {
                    self.merge_biome_json_value(biomes, report)?;
                }
                if !map.contains_key("blocks")
                    && !map.contains_key("block_colors")
                    && !map.contains_key("biomes")
                    && !map.contains_key("biome_colors")
                {
                    if looks_like_biome_map(map) || looks_like_named_biome_map(map) {
                        self.merge_biome_json_value(value, report)?;
                    } else {
                        self.merge_block_json_value(value, report)?;
                    }
                }
                Ok(())
            }
            Value::Array(_) => self.merge_block_json_value(value, report),
            _ => Err(BedrockRenderError::Validation(
                "palette JSON must be an object or an array".to_string(),
            )),
        }
    }

    fn merge_block_json_value(
        &mut self,
        value: &Value,
        report: &mut PaletteImportReport,
    ) -> Result<()> {
        match value {
            Value::Object(map) => {
                for (name, entry) in map {
                    let Some(color) = parse_block_color_entry(name, entry) else {
                        report.skipped_entries += 1;
                        continue;
                    };
                    self.block_colors.insert(normalize_block_name(name), color);
                    report.block_colors += 1;
                }
                Ok(())
            }
            Value::Array(entries) => {
                for entry in entries {
                    let Some((name, color)) = parse_named_block_entry(entry) else {
                        report.skipped_entries += 1;
                        continue;
                    };
                    self.block_colors.insert(normalize_block_name(&name), color);
                    report.block_colors += 1;
                }
                Ok(())
            }
            _ => Err(BedrockRenderError::Validation(
                "block palette JSON must be an object or array".to_string(),
            )),
        }
    }

    fn merge_biome_json_value(
        &mut self,
        value: &Value,
        report: &mut PaletteImportReport,
    ) -> Result<()> {
        match value {
            Value::Object(map) => {
                for (id, entry) in map {
                    let parsed_id = id
                        .parse::<u32>()
                        .ok()
                        .or_else(|| biome_id_from_entry(entry));
                    let Some(id) = parsed_id else {
                        report.skipped_entries += 1;
                        continue;
                    };
                    let Some(color) = parse_color(entry).or_else(|| parse_nested_color(entry))
                    else {
                        report.skipped_entries += 1;
                        continue;
                    };
                    self.biome_colors.insert(id, color);
                    self.merge_biome_tints(id, entry);
                    report.biome_colors += 1;
                }
                Ok(())
            }
            Value::Array(entries) => {
                for entry in entries {
                    let Some((id, color)) = parse_named_biome_entry(entry) else {
                        report.skipped_entries += 1;
                        continue;
                    };
                    self.biome_colors.insert(id, color);
                    self.merge_biome_tints(id, entry);
                    report.biome_colors += 1;
                }
                Ok(())
            }
            _ => Err(BedrockRenderError::Validation(
                "biome palette JSON must be an object or array".to_string(),
            )),
        }
    }

    fn merge_biome_tints(&mut self, id: u32, entry: &Value) {
        let Value::Object(map) = entry else {
            return;
        };
        if let Some(color) = map.get("grass").and_then(parse_color) {
            self.biome_grass_colors.insert(id, with_alpha(color, 255));
        }
        if let Some(color) = map.get("leaves").and_then(parse_color) {
            self.biome_foliage_colors.insert(id, with_alpha(color, 255));
        }
        if let Some(color) = map.get("water").and_then(parse_color) {
            self.biome_water_colors.insert(id, with_alpha(color, 255));
        }
        if id == 1 {
            if let Some(color) = self.biome_grass_colors.get(&id).copied() {
                self.default_grass_color = color;
            }
            if let Some(color) = self.biome_foliage_colors.get(&id).copied() {
                self.default_foliage_color = color;
            }
            if let Some(color) = self.biome_water_colors.get(&id).copied() {
                self.default_water_color = color;
            }
        }
    }

    fn merge_biome_defaults(&mut self, value: &Value) {
        let Value::Object(map) = value else {
            return;
        };
        if let Some(color) = map.get("grass").and_then(parse_color) {
            self.default_grass_color = with_alpha(color, 255);
        }
        if let Some(color) = map
            .get("leaves")
            .or_else(|| map.get("foliage"))
            .and_then(parse_color)
        {
            self.default_foliage_color = with_alpha(color, 255);
        }
        if let Some(color) = map.get("water").and_then(parse_color) {
            self.default_water_color = with_alpha(color, 255);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn insert_default_blocks(&mut self) {
        for (name, color) in [
            ("minecraft:air", self.air_color),
            ("air", self.air_color),
            ("minecraft:grass", RgbaColor::new(88, 150, 62, 255)),
            ("minecraft:short_grass", RgbaColor::new(88, 150, 62, 255)),
            ("minecraft:tall_grass", RgbaColor::new(88, 150, 62, 255)),
            ("minecraft:grass_block", RgbaColor::new(102, 158, 74, 255)),
            ("minecraft:farmland", RgbaColor::new(108, 76, 48, 255)),
            ("minecraft:podzol", RgbaColor::new(119, 86, 51, 255)),
            ("minecraft:mycelium", RgbaColor::new(112, 96, 112, 255)),
            ("minecraft:moss_block", RgbaColor::new(86, 118, 38, 255)),
            ("minecraft:dirt", RgbaColor::new(134, 96, 67, 255)),
            ("minecraft:coarse_dirt", RgbaColor::new(119, 85, 61, 255)),
            ("minecraft:rooted_dirt", RgbaColor::new(113, 82, 57, 255)),
            ("minecraft:stone", RgbaColor::new(125, 125, 125, 255)),
            ("minecraft:deepslate", RgbaColor::new(80, 80, 82, 255)),
            ("minecraft:cobblestone", RgbaColor::new(109, 109, 109, 255)),
            ("minecraft:granite", RgbaColor::new(149, 103, 85, 255)),
            ("minecraft:diorite", RgbaColor::new(190, 190, 190, 255)),
            ("minecraft:andesite", RgbaColor::new(136, 136, 136, 255)),
            ("minecraft:tuff", RgbaColor::new(92, 92, 86, 255)),
            ("minecraft:calcite", RgbaColor::new(224, 222, 214, 255)),
            (
                "minecraft:dripstone_block",
                RgbaColor::new(138, 105, 83, 255),
            ),
            ("minecraft:sand", RgbaColor::new(218, 210, 158, 255)),
            ("minecraft:sandstone", RgbaColor::new(216, 203, 145, 255)),
            ("minecraft:red_sand", RgbaColor::new(190, 103, 33, 255)),
            ("minecraft:red_sandstone", RgbaColor::new(178, 88, 30, 255)),
            ("minecraft:gravel", RgbaColor::new(126, 122, 118, 255)),
            ("minecraft:clay", RgbaColor::new(160, 166, 176, 255)),
            ("minecraft:mud", RgbaColor::new(63, 54, 50, 255)),
            ("minecraft:water", RgbaColor::new(43, 92, 210, 190)),
            ("minecraft:flowing_water", RgbaColor::new(43, 92, 210, 190)),
            ("minecraft:ice", RgbaColor::new(160, 210, 255, 210)),
            ("minecraft:packed_ice", RgbaColor::new(130, 185, 242, 235)),
            ("minecraft:blue_ice", RgbaColor::new(102, 167, 240, 235)),
            ("minecraft:snow", RgbaColor::new(245, 250, 250, 255)),
            ("minecraft:snow_layer", RgbaColor::new(245, 250, 250, 230)),
            ("minecraft:oak_leaves", RgbaColor::new(60, 112, 42, 245)),
            ("minecraft:spruce_leaves", RgbaColor::new(46, 86, 46, 245)),
            ("minecraft:birch_leaves", RgbaColor::new(85, 124, 49, 245)),
            ("minecraft:jungle_leaves", RgbaColor::new(42, 108, 44, 245)),
            ("minecraft:acacia_leaves", RgbaColor::new(72, 112, 44, 245)),
            ("minecraft:dark_oak_leaves", RgbaColor::new(38, 82, 36, 245)),
            ("minecraft:mangrove_leaves", RgbaColor::new(44, 98, 44, 245)),
            (
                "minecraft:cherry_leaves",
                RgbaColor::new(244, 174, 188, 245),
            ),
            ("minecraft:leaves", RgbaColor::new(60, 112, 42, 245)),
            ("minecraft:leaves2", RgbaColor::new(60, 112, 42, 245)),
            ("minecraft:log", RgbaColor::new(102, 76, 45, 255)),
            ("minecraft:oak_log", RgbaColor::new(102, 76, 45, 255)),
            ("minecraft:birch_log", RgbaColor::new(197, 184, 135, 255)),
            ("minecraft:spruce_log", RgbaColor::new(76, 55, 34, 255)),
            ("minecraft:jungle_log", RgbaColor::new(105, 78, 43, 255)),
            ("minecraft:acacia_log", RgbaColor::new(138, 75, 42, 255)),
            ("minecraft:dark_oak_log", RgbaColor::new(58, 39, 23, 255)),
            ("minecraft:mangrove_log", RgbaColor::new(91, 48, 43, 255)),
            ("minecraft:cherry_log", RgbaColor::new(126, 78, 85, 255)),
            ("minecraft:planks", RgbaColor::new(157, 128, 79, 255)),
            ("minecraft:oak_planks", RgbaColor::new(157, 128, 79, 255)),
            ("minecraft:birch_planks", RgbaColor::new(196, 178, 116, 255)),
            ("minecraft:spruce_planks", RgbaColor::new(114, 84, 48, 255)),
            ("minecraft:jungle_planks", RgbaColor::new(154, 109, 77, 255)),
            ("minecraft:acacia_planks", RgbaColor::new(174, 92, 50, 255)),
            ("minecraft:dark_oak_planks", RgbaColor::new(75, 50, 28, 255)),
            (
                "minecraft:mangrove_planks",
                RgbaColor::new(116, 54, 48, 255),
            ),
            (
                "minecraft:cherry_planks",
                RgbaColor::new(227, 164, 174, 255),
            ),
            ("minecraft:cactus", RgbaColor::new(35, 116, 49, 255)),
            ("minecraft:sugar_cane", RgbaColor::new(96, 178, 64, 255)),
            ("minecraft:reeds", RgbaColor::new(96, 178, 64, 255)),
            ("minecraft:bamboo", RgbaColor::new(119, 150, 45, 255)),
            ("minecraft:wheat", RgbaColor::new(193, 174, 72, 255)),
            ("minecraft:carrots", RgbaColor::new(83, 142, 45, 255)),
            ("minecraft:potatoes", RgbaColor::new(90, 145, 48, 255)),
            ("minecraft:beetroot", RgbaColor::new(118, 72, 54, 255)),
            ("minecraft:pumpkin_stem", RgbaColor::new(96, 150, 54, 255)),
            ("minecraft:melon_stem", RgbaColor::new(96, 150, 54, 255)),
            ("minecraft:double_plant", RgbaColor::new(88, 150, 62, 255)),
            ("minecraft:large_fern", RgbaColor::new(76, 132, 55, 255)),
            ("minecraft:fern", RgbaColor::new(76, 132, 55, 255)),
            ("minecraft:bush", RgbaColor::new(58, 112, 46, 255)),
            ("minecraft:azalea", RgbaColor::new(66, 118, 52, 255)),
            (
                "minecraft:flowering_azalea",
                RgbaColor::new(92, 128, 64, 255),
            ),
            ("minecraft:mangrove_roots", RgbaColor::new(72, 55, 43, 255)),
            (
                "minecraft:muddy_mangrove_roots",
                RgbaColor::new(67, 52, 43, 255),
            ),
            ("minecraft:pink_petals", RgbaColor::new(238, 148, 184, 255)),
            ("minecraft:wildflowers", RgbaColor::new(214, 180, 82, 255)),
            ("minecraft:leaf_litter", RgbaColor::new(132, 88, 44, 255)),
            ("minecraft:yellow_flower", RgbaColor::new(230, 204, 56, 255)),
            ("minecraft:red_flower", RgbaColor::new(196, 44, 40, 255)),
            ("minecraft:poppy", RgbaColor::new(196, 44, 40, 255)),
            ("minecraft:dandelion", RgbaColor::new(230, 204, 56, 255)),
            ("minecraft:blue_orchid", RgbaColor::new(50, 150, 190, 255)),
            ("minecraft:allium", RgbaColor::new(154, 92, 190, 255)),
            ("minecraft:azure_bluet", RgbaColor::new(230, 230, 215, 255)),
            ("minecraft:oxeye_daisy", RgbaColor::new(232, 232, 220, 255)),
            ("minecraft:cornflower", RgbaColor::new(72, 105, 190, 255)),
            (
                "minecraft:lily_of_the_valley",
                RgbaColor::new(238, 238, 226, 255),
            ),
            ("minecraft:sunflower", RgbaColor::new(229, 184, 40, 255)),
            ("minecraft:lilac", RgbaColor::new(190, 130, 190, 255)),
            ("minecraft:rose_bush", RgbaColor::new(178, 40, 70, 255)),
            ("minecraft:peony", RgbaColor::new(220, 142, 180, 255)),
            (
                "minecraft:brown_mushroom",
                RgbaColor::new(145, 109, 83, 255),
            ),
            ("minecraft:red_mushroom", RgbaColor::new(182, 54, 45, 255)),
            ("minecraft:deadbush", RgbaColor::new(125, 91, 48, 255)),
            ("minecraft:sapling", RgbaColor::new(65, 125, 47, 255)),
            ("minecraft:seagrass", RgbaColor::new(44, 124, 80, 255)),
            ("minecraft:kelp", RgbaColor::new(48, 110, 72, 255)),
            ("minecraft:lily_pad", RgbaColor::new(44, 115, 52, 255)),
            ("minecraft:glow_lichen", RgbaColor::new(137, 157, 145, 210)),
            ("minecraft:torch", RgbaColor::new(245, 190, 78, 255)),
            ("minecraft:lantern", RgbaColor::new(198, 150, 82, 255)),
            ("minecraft:rail", RgbaColor::new(116, 108, 96, 255)),
            ("minecraft:iron_bars", RgbaColor::new(132, 132, 132, 210)),
            ("minecraft:piston", RgbaColor::new(112, 98, 74, 255)),
            ("minecraft:sticky_piston", RgbaColor::new(94, 116, 70, 255)),
            ("minecraft:hopper", RgbaColor::new(72, 78, 82, 255)),
            ("minecraft:lever", RgbaColor::new(118, 104, 78, 255)),
            ("minecraft:scaffolding", RgbaColor::new(190, 162, 89, 235)),
            ("minecraft:fire", RgbaColor::new(242, 98, 18, 210)),
            ("minecraft:lava", RgbaColor::new(255, 90, 0, 255)),
            ("minecraft:flowing_lava", RgbaColor::new(255, 90, 0, 255)),
            ("minecraft:bubble_column", RgbaColor::new(72, 128, 220, 120)),
            ("minecraft:netherrack", RgbaColor::new(110, 53, 51, 255)),
            ("minecraft:basalt", RgbaColor::new(74, 72, 76, 255)),
            ("minecraft:blackstone", RgbaColor::new(42, 36, 43, 255)),
            ("minecraft:soul_sand", RgbaColor::new(82, 64, 56, 255)),
            ("minecraft:soul_soil", RgbaColor::new(77, 60, 52, 255)),
            ("minecraft:warped_nylium", RgbaColor::new(43, 106, 103, 255)),
            ("minecraft:crimson_nylium", RgbaColor::new(117, 33, 45, 255)),
            ("minecraft:shroomlight", RgbaColor::new(240, 158, 88, 255)),
            ("minecraft:weeping_vines", RgbaColor::new(126, 35, 54, 235)),
            ("minecraft:end_stone", RgbaColor::new(220, 222, 158, 255)),
            ("minecraft:end_portal", RgbaColor::new(24, 18, 34, 180)),
            ("minecraft:obsidian", RgbaColor::new(26, 20, 39, 255)),
            ("minecraft:bedrock", RgbaColor::new(84, 84, 84, 255)),
            ("minecraft:terracotta", RgbaColor::new(152, 94, 67, 255)),
            (
                "minecraft:white_terracotta",
                RgbaColor::new(210, 178, 161, 255),
            ),
            (
                "minecraft:orange_terracotta",
                RgbaColor::new(162, 84, 38, 255),
            ),
            (
                "minecraft:yellow_terracotta",
                RgbaColor::new(186, 133, 36, 255),
            ),
            ("minecraft:red_terracotta", RgbaColor::new(143, 61, 47, 255)),
            (
                "minecraft:brown_terracotta",
                RgbaColor::new(92, 61, 45, 255),
            ),
            ("minecraft:gray_terracotta", RgbaColor::new(57, 42, 36, 255)),
            (
                "minecraft:light_gray_terracotta",
                RgbaColor::new(135, 107, 98, 255),
            ),
            ("minecraft:concrete", RgbaColor::new(128, 128, 128, 255)),
            (
                "minecraft:white_concrete",
                RgbaColor::new(207, 213, 214, 255),
            ),
            ("minecraft:orange_concrete", RgbaColor::new(224, 97, 0, 255)),
            (
                "minecraft:yellow_concrete",
                RgbaColor::new(241, 175, 21, 255),
            ),
            ("minecraft:green_concrete", RgbaColor::new(73, 91, 36, 255)),
            ("minecraft:blue_concrete", RgbaColor::new(45, 47, 143, 255)),
            ("minecraft:white_wool", RgbaColor::new(234, 236, 237, 255)),
            ("minecraft:orange_wool", RgbaColor::new(241, 118, 20, 255)),
            ("minecraft:yellow_wool", RgbaColor::new(248, 197, 39, 255)),
            ("minecraft:green_wool", RgbaColor::new(84, 109, 27, 255)),
            ("minecraft:blue_wool", RgbaColor::new(53, 57, 157, 255)),
        ] {
            self.block_colors.insert(name.to_string(), color);
        }
        self.insert_generated_wood_defaults();
        self.insert_generated_dye_defaults();
        self.insert_generated_ore_defaults();
        self.insert_generated_utility_defaults();
    }

    fn insert_generated_wood_defaults(&mut self) {
        for (wood, log, plank, leaves) in [
            (
                "oak",
                RgbaColor::new(102, 76, 45, 255),
                RgbaColor::new(157, 128, 79, 255),
                RgbaColor::new(60, 112, 42, 245),
            ),
            (
                "spruce",
                RgbaColor::new(76, 55, 34, 255),
                RgbaColor::new(114, 84, 48, 255),
                RgbaColor::new(46, 86, 46, 245),
            ),
            (
                "birch",
                RgbaColor::new(197, 184, 135, 255),
                RgbaColor::new(196, 178, 116, 255),
                RgbaColor::new(85, 124, 49, 245),
            ),
            (
                "jungle",
                RgbaColor::new(105, 78, 43, 255),
                RgbaColor::new(154, 109, 77, 255),
                RgbaColor::new(42, 108, 44, 245),
            ),
            (
                "acacia",
                RgbaColor::new(138, 75, 42, 255),
                RgbaColor::new(174, 92, 50, 255),
                RgbaColor::new(72, 112, 44, 245),
            ),
            (
                "dark_oak",
                RgbaColor::new(58, 39, 23, 255),
                RgbaColor::new(75, 50, 28, 255),
                RgbaColor::new(38, 82, 36, 245),
            ),
            (
                "mangrove",
                RgbaColor::new(91, 48, 43, 255),
                RgbaColor::new(116, 54, 48, 255),
                RgbaColor::new(44, 98, 44, 245),
            ),
            (
                "cherry",
                RgbaColor::new(126, 78, 85, 255),
                RgbaColor::new(227, 164, 174, 255),
                RgbaColor::new(244, 174, 188, 245),
            ),
            (
                "bamboo",
                RgbaColor::new(119, 150, 45, 255),
                RgbaColor::new(196, 174, 88, 255),
                RgbaColor::new(119, 150, 45, 255),
            ),
            (
                "crimson",
                RgbaColor::new(122, 43, 72, 255),
                RgbaColor::new(126, 49, 83, 255),
                RgbaColor::new(96, 31, 70, 245),
            ),
            (
                "warped",
                RgbaColor::new(42, 112, 108, 255),
                RgbaColor::new(44, 122, 116, 255),
                RgbaColor::new(32, 104, 96, 245),
            ),
        ] {
            for (suffix, color) in [
                ("log", log),
                ("wood", log),
                ("stripped_log", plank),
                ("stripped_wood", plank),
                ("planks", plank),
                ("stairs", plank),
                ("slab", plank),
                ("fence", plank),
                ("fence_gate", plank),
                ("door", plank),
                ("trapdoor", plank),
                ("button", plank),
                ("pressure_plate", plank),
                ("sign", plank),
                ("hanging_sign", plank),
                ("leaves", leaves),
                ("sapling", leaves),
            ] {
                self.block_colors
                    .insert(format!("minecraft:{wood}_{suffix}"), color);
            }
        }
    }

    fn insert_generated_dye_defaults(&mut self) {
        for (color_name, color) in [
            ("white", RgbaColor::new(232, 236, 236, 255)),
            ("light_gray", RgbaColor::new(142, 142, 134, 255)),
            ("gray", RgbaColor::new(62, 68, 71, 255)),
            ("black", RgbaColor::new(21, 21, 26, 255)),
            ("brown", RgbaColor::new(96, 59, 31, 255)),
            ("red", RgbaColor::new(150, 42, 38, 255)),
            ("orange", RgbaColor::new(224, 97, 0, 255)),
            ("yellow", RgbaColor::new(241, 175, 21, 255)),
            ("lime", RgbaColor::new(112, 185, 25, 255)),
            ("green", RgbaColor::new(73, 91, 36, 255)),
            ("cyan", RgbaColor::new(21, 137, 145, 255)),
            ("light_blue", RgbaColor::new(58, 175, 217, 255)),
            ("blue", RgbaColor::new(45, 47, 143, 255)),
            ("purple", RgbaColor::new(120, 50, 167, 255)),
            ("magenta", RgbaColor::new(190, 68, 201, 255)),
            ("pink", RgbaColor::new(237, 141, 172, 255)),
        ] {
            for suffix in [
                "wool",
                "carpet",
                "concrete",
                "concrete_powder",
                "terracotta",
                "stained_glass",
                "stained_glass_pane",
                "glazed_terracotta",
                "shulker_box",
                "candle",
                "bed",
            ] {
                self.block_colors
                    .insert(format!("minecraft:{color_name}_{suffix}"), color);
            }
        }
    }

    fn insert_generated_ore_defaults(&mut self) {
        for (name, color) in [
            ("coal", RgbaColor::new(52, 52, 52, 255)),
            ("iron", RgbaColor::new(190, 160, 130, 255)),
            ("copper", RgbaColor::new(190, 112, 72, 255)),
            ("gold", RgbaColor::new(232, 190, 62, 255)),
            ("redstone", RgbaColor::new(170, 42, 42, 255)),
            ("emerald", RgbaColor::new(58, 190, 95, 255)),
            ("lapis", RgbaColor::new(48, 88, 180, 255)),
            ("diamond", RgbaColor::new(92, 210, 218, 255)),
            ("quartz", RgbaColor::new(226, 218, 205, 255)),
            ("nether_gold", RgbaColor::new(198, 134, 45, 255)),
        ] {
            for suffix in ["ore", "block"] {
                self.block_colors
                    .insert(format!("minecraft:{name}_{suffix}"), color);
            }
            self.block_colors
                .insert(format!("minecraft:deepslate_{name}_ore"), darken(color, 32));
        }
    }

    fn insert_generated_utility_defaults(&mut self) {
        for (name, color) in [
            ("crafting_table", RgbaColor::new(142, 101, 58, 255)),
            ("furnace", RgbaColor::new(92, 92, 92, 255)),
            ("blast_furnace", RgbaColor::new(82, 82, 86, 255)),
            ("smoker", RgbaColor::new(92, 76, 60, 255)),
            ("chest", RgbaColor::new(154, 105, 42, 255)),
            ("trapped_chest", RgbaColor::new(154, 105, 42, 255)),
            ("ender_chest", RgbaColor::new(42, 68, 72, 255)),
            ("barrel", RgbaColor::new(116, 79, 45, 255)),
            ("bookshelf", RgbaColor::new(132, 88, 48, 255)),
            ("lectern", RgbaColor::new(130, 90, 52, 255)),
            ("anvil", RgbaColor::new(72, 72, 72, 255)),
            ("grindstone", RgbaColor::new(132, 124, 112, 255)),
            ("loom", RgbaColor::new(151, 127, 83, 255)),
            ("cartography_table", RgbaColor::new(128, 104, 72, 255)),
            ("smithing_table", RgbaColor::new(86, 66, 52, 255)),
            ("stonecutter", RgbaColor::new(108, 108, 108, 255)),
            ("composter", RgbaColor::new(94, 67, 38, 255)),
            ("beehive", RgbaColor::new(190, 145, 55, 255)),
            ("bee_nest", RgbaColor::new(185, 145, 70, 255)),
            ("bell", RgbaColor::new(220, 170, 62, 255)),
        ] {
            self.block_colors.insert(format!("minecraft:{name}"), color);
        }
    }

    fn find_block_color(&self, name: &str) -> Option<RgbaColor> {
        self.block_colors
            .get(name)
            .copied()
            .or_else(|| {
                name.strip_prefix("minecraft:")
                    .and_then(|short_name| self.block_colors.get(short_name).copied())
            })
            .or_else(|| category_block_color(name))
    }
}

fn is_air_block(name: &str) -> bool {
    matches!(
        name,
        "air"
            | "minecraft:air"
            | "minecraft:cave_air"
            | "minecraft:void_air"
            | "minecraft:structure_void"
            | "minecraft:light_block"
            | "minecraft:light"
    )
}

fn looks_like_biome_map(map: &serde_json::Map<String, Value>) -> bool {
    !map.is_empty() && map.keys().all(|key| key.parse::<u32>().is_ok())
}

fn looks_like_named_biome_map(map: &serde_json::Map<String, Value>) -> bool {
    !map.is_empty()
        && map.values().any(|value| {
            let Value::Object(entry) = value else {
                return false;
            };
            entry.contains_key("id") && entry.contains_key("rgb")
        })
}

fn normalize_block_name(name: &str) -> String {
    if name.contains(':') {
        name.to_string()
    } else {
        format!("minecraft:{name}")
    }
}

fn parse_named_block_entry(value: &Value) -> Option<(String, RgbaColor)> {
    let Value::Object(map) = value else {
        return None;
    };
    let name = map
        .get("name")
        .or_else(|| map.get("id"))
        .or_else(|| map.get("identifier"))
        .and_then(Value::as_str)?;
    let color = parse_block_color_entry(name, value)?;
    Some((name.to_string(), color))
}

fn parse_named_biome_entry(value: &Value) -> Option<(u32, RgbaColor)> {
    let id = biome_id_from_entry(value)?;
    let color = parse_color(value).or_else(|| parse_nested_color(value))?;
    Some((id, color))
}

fn biome_id_from_entry(value: &Value) -> Option<u32> {
    let Value::Object(map) = value else {
        return None;
    };
    map.get("id")
        .or_else(|| map.get("biome_id"))
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
}

fn parse_nested_color(value: &Value) -> Option<RgbaColor> {
    let Value::Object(map) = value else {
        return None;
    };
    for key in ["color", "map_color", "rgba", "rgb"] {
        if let Some(color) = map.get(key).and_then(parse_color) {
            return Some(color);
        }
    }
    average_child_colors(map)
}

fn parse_block_color_entry(block_name: &str, value: &Value) -> Option<RgbaColor> {
    parse_color(value).or_else(|| parse_nested_block_color(block_name, value))
}

fn parse_nested_block_color(block_name: &str, value: &Value) -> Option<RgbaColor> {
    let Value::Object(map) = value else {
        return None;
    };
    for key in ["color", "map_color", "rgba", "rgb"] {
        if let Some(color) = map.get(key).and_then(parse_color) {
            return Some(color);
        }
    }
    best_child_color_for_block(block_name, map).or_else(|| average_child_colors(map))
}

fn best_child_color_for_block(
    block_name: &str,
    map: &serde_json::Map<String, Value>,
) -> Option<RgbaColor> {
    let block_tokens = block_name_tokens(block_name);
    let mut best: Option<(i32, u8, RgbaColor)> = None;
    for (texture_name, value) in map {
        let Some(color) = parse_color(value) else {
            continue;
        };
        let score = texture_match_score(texture_name, &block_tokens);
        let candidate = (score, color.alpha, color);
        if best.is_none_or(|current| (candidate.0, candidate.1) > (current.0, current.1)) {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, color)| color)
}

fn block_name_tokens(block_name: &str) -> Vec<&str> {
    let short_name = block_name
        .strip_prefix("minecraft:")
        .unwrap_or(block_name)
        .trim();
    short_name
        .split(['_', ':'])
        .filter(|token| !token.is_empty() && !matches!(*token, "block" | "item"))
        .collect()
}

fn texture_match_score(texture_name: &str, block_tokens: &[&str]) -> i32 {
    let mut score = 0_i32;
    let texture_tokens = texture_name.split(['_', ':']).collect::<Vec<_>>();
    for token in block_tokens {
        if texture_tokens
            .iter()
            .any(|texture_token| texture_token == token)
        {
            score += 16;
        } else if texture_name.contains(token) {
            score += 8;
        }
    }
    if texture_tokens
        .iter()
        .any(|token| matches!(*token, "top" | "upper" | "opaque"))
    {
        score += 2;
    }
    if texture_tokens
        .iter()
        .any(|token| matches!(*token, "side" | "lower" | "bottom"))
    {
        score -= 1;
    }
    score
}

fn average_child_colors(map: &serde_json::Map<String, Value>) -> Option<RgbaColor> {
    let mut red = 0_u32;
    let mut green = 0_u32;
    let mut blue = 0_u32;
    let mut alpha = 0_u32;
    let mut count = 0_u32;
    for value in map.values() {
        let Some(color) = parse_color(value) else {
            continue;
        };
        red += u32::from(color.red);
        green += u32::from(color.green);
        blue += u32::from(color.blue);
        alpha += u32::from(color.alpha);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(RgbaColor::new(
        u8_from_u32(red / count),
        u8_from_u32(green / count),
        u8_from_u32(blue / count),
        u8_from_u32(alpha / count),
    ))
}

fn parse_color(value: &Value) -> Option<RgbaColor> {
    match value {
        Value::String(color) => parse_hex_color(color),
        Value::Array(channels) => parse_color_array(channels),
        Value::Object(map) => parse_color_object(map),
        Value::Number(number) => number.as_u64().and_then(parse_integer_color),
        _ => None,
    }
}

fn parse_color_array(channels: &[Value]) -> Option<RgbaColor> {
    if !(channels.len() == 3 || channels.len() == 4) {
        return None;
    }
    let red = parse_u8_channel(channels.first()?)?;
    let green = parse_u8_channel(channels.get(1)?)?;
    let blue = parse_u8_channel(channels.get(2)?)?;
    let alpha = channels.get(3).map_or(Some(255), parse_u8_channel)?;
    Some(RgbaColor::new(red, green, blue, alpha))
}

fn parse_color_object(map: &serde_json::Map<String, Value>) -> Option<RgbaColor> {
    let red = map
        .get("red")
        .or_else(|| map.get("r"))
        .and_then(parse_u8_channel)?;
    let green = map
        .get("green")
        .or_else(|| map.get("g"))
        .and_then(parse_u8_channel)?;
    let blue = map
        .get("blue")
        .or_else(|| map.get("b"))
        .and_then(parse_u8_channel)?;
    let alpha = map
        .get("alpha")
        .or_else(|| map.get("a"))
        .map_or(Some(255), parse_u8_channel)?;
    Some(RgbaColor::new(red, green, blue, alpha))
}

fn parse_u8_channel(value: &Value) -> Option<u8> {
    value
        .as_u64()
        .and_then(|channel| u8::try_from(channel).ok())
}

fn parse_hex_color(value: &str) -> Option<RgbaColor> {
    let value = value
        .trim()
        .trim_start_matches('#')
        .trim_start_matches("0x");
    match value.len() {
        6 => {
            let color = u32::from_str_radix(value, 16).ok()?;
            Some(RgbaColor::new(
                u8_from_u32((color >> 16) & 0xff),
                u8_from_u32((color >> 8) & 0xff),
                u8_from_u32(color & 0xff),
                255,
            ))
        }
        8 => {
            let color = u32::from_str_radix(value, 16).ok()?;
            Some(RgbaColor::new(
                u8_from_u32((color >> 24) & 0xff),
                u8_from_u32((color >> 16) & 0xff),
                u8_from_u32((color >> 8) & 0xff),
                u8_from_u32(color & 0xff),
            ))
        }
        _ => None,
    }
}

fn parse_integer_color(value: u64) -> Option<RgbaColor> {
    let color = u32::try_from(value).ok()?;
    Some(RgbaColor::new(
        u8_from_u32((color >> 16) & 0xff),
        u8_from_u32((color >> 8) & 0xff),
        u8_from_u32(color & 0xff),
        255,
    ))
}

fn read_u16(cursor: &mut Cursor<&[u8]>, label: &str) -> Result<u16> {
    let mut bytes = [0_u8; 2];
    cursor
        .read_exact(&mut bytes)
        .map_err(|error| BedrockRenderError::Validation(format!("invalid {label}: {error}")))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(cursor: &mut Cursor<&[u8]>, label: &str) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|error| BedrockRenderError::Validation(format!("invalid {label}: {error}")))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_rgba(cursor: &mut Cursor<&[u8]>, label: &str) -> Result<RgbaColor> {
    let mut bytes = [0_u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|error| BedrockRenderError::Validation(format!("invalid {label}: {error}")))?;
    Ok(RgbaColor::new(bytes[0], bytes[1], bytes[2], bytes[3]))
}

fn is_water_block(name: &str) -> bool {
    matches!(
        name,
        "water" | "flowing_water" | "minecraft:water" | "minecraft:flowing_water"
    )
}

fn is_grass_tinted_block(name: &str) -> bool {
    name.contains("grass_block")
        || name.ends_with(":grass")
        || name.ends_with(":short_grass")
        || name.ends_with(":tall_grass")
        || name.contains("fern")
        || name.contains("vine")
}

fn is_foliage_tinted_block(name: &str) -> bool {
    name.contains("leaves")
        || name.contains("leaf")
        || name.contains("leave")
        || name.contains("foliage")
}

#[allow(clippy::too_many_lines)]
fn category_block_color(name: &str) -> Option<RgbaColor> {
    let name = name.strip_prefix("minecraft:").unwrap_or(name);
    if name.contains("coral") {
        return Some(RgbaColor::new(210, 88, 110, 255));
    }
    if name.contains("copper") {
        return Some(RgbaColor::new(179, 109, 77, 255));
    }
    if name.contains("resin") {
        return Some(RgbaColor::new(226, 112, 32, 255));
    }
    if name.contains("amethyst") {
        return Some(RgbaColor::new(154, 112, 210, 255));
    }
    if name.contains("prismarine") {
        return Some(RgbaColor::new(86, 154, 146, 255));
    }
    if name.contains("basalt") || name.contains("blackstone") {
        return Some(RgbaColor::new(42, 38, 45, 255));
    }
    if name.contains("netherrack") || name.contains("nylium") || name.contains("wart") {
        return Some(RgbaColor::new(112, 42, 44, 255));
    }
    if name.contains("end_stone") || name.contains("end_brick") {
        return Some(RgbaColor::new(218, 222, 158, 255));
    }
    if name.contains("ore") {
        return Some(RgbaColor::new(118, 118, 118, 255));
    }
    if name.contains("stone")
        || name.contains("slate")
        || name.contains("brick")
        || name.contains("polished")
        || name.contains("smooth")
        || name.contains("chiseled")
        || name.contains("tile")
    {
        return Some(RgbaColor::new(122, 122, 122, 255));
    }
    if name.contains("leaves") || name.contains("leaf") {
        return Some(RgbaColor::new(58, 105, 42, 245));
    }
    if name.contains("log")
        || name.contains("stem")
        || name.contains("wood")
        || name.contains("hyphae")
        || name.contains("bamboo")
    {
        return Some(RgbaColor::new(99, 70, 42, 255));
    }
    if name.contains("planks") {
        return Some(RgbaColor::new(154, 118, 70, 255));
    }
    if name.contains("stairs")
        || name.contains("slab")
        || name.contains("fence")
        || name.contains("door")
        || name.contains("trapdoor")
        || name.contains("button")
        || name.contains("pressure_plate")
        || name.contains("sign")
        || name.contains("hanging_sign")
        || name.contains("ladder")
        || name.contains("chest")
        || name.contains("barrel")
        || name.contains("bookshelf")
    {
        return Some(RgbaColor::new(145, 105, 62, 255));
    }
    if name.contains("flower")
        || name.contains("tulip")
        || name.contains("daisy")
        || name.contains("orchid")
        || name.contains("allium")
        || name.contains("cornflower")
        || name.contains("peony")
        || name.contains("lilac")
        || name.contains("rose")
        || name.contains("petals")
        || name.contains("spore_blossom")
    {
        return Some(RgbaColor::new(210, 120, 80, 255));
    }
    if name.contains("grass")
        || name.contains("fern")
        || name.contains("bush")
        || name.contains("sapling")
        || name.contains("crop")
        || name.contains("wheat")
        || name.contains("carrots")
        || name.contains("potatoes")
        || name.contains("beetroot")
        || name.contains("melon_stem")
        || name.contains("pumpkin_stem")
        || name.contains("seagrass")
        || name.contains("kelp")
        || name.contains("lily_pad")
        || name.contains("moss")
        || name.contains("roots")
        || name.contains("azalea")
        || name.contains("propagule")
        || name.contains("cactus")
        || name.contains("sugar_cane")
    {
        return Some(RgbaColor::new(77, 140, 56, 255));
    }
    if name.contains("hay") || name.contains("sponge") || name.contains("honey") {
        return Some(RgbaColor::new(204, 164, 62, 255));
    }
    if name.contains("mushroom") {
        return Some(RgbaColor::new(150, 96, 76, 255));
    }
    if name.contains("torch")
        || name.contains("lantern")
        || name.contains("rail")
        || name.contains("redstone")
        || name.contains("repeater")
        || name.contains("comparator")
    {
        return Some(RgbaColor::new(168, 136, 84, 255));
    }
    if name.contains("sand") || name.contains("beach") {
        return Some(RgbaColor::new(214, 199, 140, 255));
    }
    if name.contains("mud") || name.contains("dirt") || name.contains("path") {
        return Some(RgbaColor::new(124, 90, 62, 255));
    }
    if name.contains("terracotta") {
        return Some(RgbaColor::new(145, 88, 63, 255));
    }
    if name.contains("concrete") {
        return Some(RgbaColor::new(136, 136, 136, 255));
    }
    if name.contains("wool") || name.contains("carpet") {
        return Some(RgbaColor::new(190, 190, 190, 255));
    }
    if name.contains("glass") {
        return Some(RgbaColor::new(180, 220, 235, 128));
    }
    if name.contains("snow") {
        return Some(RgbaColor::new(245, 250, 250, 255));
    }
    if name.contains("ice") || name.contains("frosted") {
        return Some(RgbaColor::new(145, 200, 245, 220));
    }
    if name.contains("water") {
        return Some(RgbaColor::new(43, 92, 210, 190));
    }
    if name.contains("lava") {
        return Some(RgbaColor::new(255, 90, 0, 255));
    }
    if name.contains("obsidian") {
        return Some(RgbaColor::new(25, 20, 36, 255));
    }
    if name.contains("bedrock") {
        return Some(RgbaColor::new(82, 82, 82, 255));
    }
    None
}

fn multiply_with_biome_tint(
    base: RgbaColor,
    tint: Option<RgbaColor>,
    default_tint: RgbaColor,
) -> RgbaColor {
    let tint = tint.unwrap_or(default_tint);
    RgbaColor::new(
        multiply_channel(base.red, tint.red),
        multiply_channel(base.green, tint.green),
        multiply_channel(base.blue, tint.blue),
        255,
    )
}

fn multiply_channel(base: u8, tint: u8) -> u8 {
    let value = (u16::from(base) * u16::from(tint)) / 255;
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn with_alpha(color: RgbaColor, alpha: u8) -> RgbaColor {
    RgbaColor::new(color.red, color.green, color.blue, alpha)
}

fn alpha_blend(foreground: RgbaColor, background: RgbaColor) -> RgbaColor {
    let alpha = u16::from(foreground.alpha);
    let inverse = 255_u16.saturating_sub(alpha);
    RgbaColor::new(
        u8_from_u16(
            ((u16::from(foreground.red) * alpha) + (u16::from(background.red) * inverse)) / 255,
        ),
        u8_from_u16(
            ((u16::from(foreground.green) * alpha) + (u16::from(background.green) * inverse)) / 255,
        ),
        u8_from_u16(
            ((u16::from(foreground.blue) * alpha) + (u16::from(background.blue) * inverse)) / 255,
        ),
        255,
    )
}

fn shade_color(color: RgbaColor, factor: i32) -> RgbaColor {
    RgbaColor::new(
        shade_channel(color.red, factor),
        shade_channel(color.green, factor),
        shade_channel(color.blue, factor),
        color.alpha,
    )
}

fn shade_channel(channel: u8, factor: i32) -> u8 {
    if factor >= 0 {
        let value = i32::from(channel) + ((255 - i32::from(channel)) * factor / 100);
        u8_from_i32(value.clamp(0, 255))
    } else {
        let value = i32::from(channel) * (100 + factor) / 100;
        u8_from_i32(value.clamp(0, 255))
    }
}

fn darken(color: RgbaColor, amount: u8) -> RgbaColor {
    RgbaColor::new(
        color.red.saturating_sub(amount),
        color.green.saturating_sub(amount),
        color.blue.saturating_sub(amount),
        color.alpha,
    )
}

fn lerp_color(start: RgbaColor, end: RgbaColor, numerator: i32, denominator: i32) -> RgbaColor {
    RgbaColor::new(
        lerp_channel(start.red, end.red, numerator, denominator),
        lerp_channel(start.green, end.green, numerator, denominator),
        lerp_channel(start.blue, end.blue, numerator, denominator),
        lerp_channel(start.alpha, end.alpha, numerator, denominator),
    )
}

fn lerp_channel(start: u8, end: u8, numerator: i32, denominator: i32) -> u8 {
    let value =
        i32::from(start) + (i32::from(end) - i32::from(start)) * numerator / denominator.max(1);
    u8_from_i32(value.clamp(0, 255))
}

fn biome_hash_color(id: u32) -> RgbaColor {
    let hash = id.wrapping_mul(0x045d_9f3b);
    RgbaColor::new(
        64 + u8_from_u32(hash & 0x7f),
        64 + u8_from_u32((hash >> 8) & 0x7f),
        64 + u8_from_u32((hash >> 16) & 0x7f),
        255,
    )
}

fn u8_from_u16(value: u16) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn u8_from_u32(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn u8_from_i32(value: i32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_palette_imports_blocks_and_biomes() {
        let mut palette = RenderPalette::default();
        let report = palette
            .merge_json_str(
                r##"{
                    "blocks": {
                        "minecraft:test_block": "#112233",
                        "test_alpha": [10, 20, 30, 40]
                    },
                    "biomes": {
                        "900": {"color": "#445566ff"}
                    }
                }"##,
            )
            .expect("valid palette json should import");

        assert_eq!(report.block_colors, 2);
        assert_eq!(report.biome_colors, 1);
        assert_eq!(
            palette.block_color("minecraft:test_block"),
            RgbaColor::new(17, 34, 51, 255)
        );
        assert_eq!(
            palette.block_color("minecraft:test_alpha"),
            RgbaColor::new(10, 20, 30, 40)
        );
        assert_eq!(palette.biome_color(900), RgbaColor::new(68, 85, 102, 255));
    }

    #[test]
    fn bedrock_level_style_palette_json_imports() {
        let mut palette = RenderPalette::default();
        let report = palette
            .merge_json_str(
                r#"{
                    "minecraft:sample_multi_texture": {
                        "texture_a": [10, 20, 30, 255],
                        "texture_b": [30, 40, 50, 255]
                    },
                    "plains": {
                        "id": 1,
                        "rgb": [141, 179, 96],
                        "grass": [142, 185, 113],
                        "leaves": [113, 167, 77],
                        "water": [63, 118, 228]
                    }
                }"#,
            )
            .expect("bedrock-level style json should import");

        assert_eq!(report.block_colors, 0);
        assert_eq!(report.biome_colors, 1);
        assert_eq!(palette.biome_color(1), RgbaColor::new(141, 179, 96, 255));

        let mut block_palette = RenderPalette::default();
        let report = block_palette
            .merge_json_str(
                r#"{
                    "minecraft:sample_multi_texture": {
                        "texture_a": [10, 20, 30, 255],
                        "texture_b": [30, 40, 50, 255]
                    }
                }"#,
            )
            .expect("bedrock-level style block json should import");
        assert_eq!(report.block_colors, 1);
        assert_eq!(
            block_palette.block_color("minecraft:sample_multi_texture"),
            RgbaColor::new(10, 20, 30, 255)
        );

        let mut biome_palette = RenderPalette::default();
        let report = biome_palette
            .merge_json_str(
                r#"{
                    "plains": {
                        "id": 901,
                        "rgb": [141, 179, 96],
                        "grass": [142, 185, 113]
                    }
                }"#,
            )
            .expect("named biome json should import");
        assert_eq!(report.biome_colors, 1);
        assert_eq!(
            biome_palette.biome_color(901),
            RgbaColor::new(141, 179, 96, 255)
        );
    }

    #[test]
    fn block_palette_prefers_texture_matching_block_name() {
        let mut palette = RenderPalette::default();
        let report = palette
            .merge_json_str(
                r#"{
                    "minecraft:acacia_door": {
                        "door_wood_lower": [138, 108, 62, 255],
                        "door_spruce_lower": [105, 80, 48, 255],
                        "door_acacia_lower": [162, 91, 57, 183],
                        "door_dark_oak_lower": [73, 49, 23, 255]
                    },
                    "minecraft:oak_leaves": {
                        "leaves_oak": [144, 143, 144, 171],
                        "leaves_oak_opaque": [131, 129, 131, 255]
                    }
                }"#,
            )
            .expect("multi-texture block palette should import");

        assert_eq!(report.block_colors, 2);
        assert_eq!(
            palette.block_color("minecraft:acacia_door"),
            RgbaColor::new(162, 91, 57, 183)
        );
        assert_eq!(
            palette.block_color("minecraft:oak_leaves"),
            RgbaColor::new(131, 129, 131, 255)
        );
    }

    #[test]
    fn biome_palette_imports_surface_tint_colors() {
        let mut palette = RenderPalette::default();
        palette
            .merge_json_str(
                r#"{
                    "custom": {
                        "id": 902,
                        "rgb": [1, 2, 3],
                        "grass": [80, 200, 40],
                        "leaves": [40, 160, 30],
                        "water": [20, 60, 180]
                    }
                }"#,
            )
            .expect("biome tint json should import");

        let grass = palette.surface_block_color("minecraft:grass_block", Some(902), true);
        let leaves = palette.surface_block_color("minecraft:oak_leaves", Some(902), true);
        let water = palette.surface_block_color("minecraft:water", Some(902), true);

        assert_eq!(grass.alpha, 255);
        assert_eq!(leaves.alpha, 255);
        assert_eq!(water.alpha, 255);
        assert_ne!(
            grass,
            palette.surface_block_color("minecraft:grass_block", None, true)
        );
        assert_ne!(
            leaves,
            palette.surface_block_color("minecraft:oak_leaves", None, true)
        );
        assert_ne!(
            water,
            palette.surface_block_color("minecraft:water", None, true)
        );
    }

    #[test]
    fn binary_palette_cache_roundtrips_without_json() {
        let palette = RenderPalette::default()
            .with_block_color("minecraft:cache_block", RgbaColor::new(1, 2, 3, 4))
            .with_biome_color(901, RgbaColor::new(5, 6, 7, 8));
        let bytes = palette
            .to_binary_vec()
            .expect("palette cache should encode");
        let mut imported = RenderPalette::default();
        let report = imported
            .merge_binary_slice(&bytes)
            .expect("palette cache should decode");

        assert!(report.block_colors > 0);
        assert!(report.biome_colors > 0);
        assert_eq!(
            imported.block_color("minecraft:cache_block"),
            RgbaColor::new(1, 2, 3, 4)
        );
        assert_eq!(imported.biome_color(901), RgbaColor::new(5, 6, 7, 8));
    }

    #[test]
    fn builtin_palette_cache_is_loaded_by_default() {
        let palette = RenderPalette::default();
        assert!(palette.block_colors.len() >= 1_200);
        assert!(palette.biome_colors.len() >= 88);
        assert!(palette.has_block_color("minecraft:farmland"));
        assert!(palette.has_block_color("minecraft:glow_lichen"));
        assert!(palette.has_block_color("minecraft:weeping_vines"));
        assert_eq!(palette.block_color("minecraft:unknown_test").alpha, 255);
    }

    #[test]
    fn builtin_json_sources_rebuild_embedded_palette_cache() {
        let palette = RenderPalette::from_builtin_json_sources()
            .expect("built-in JSON palette sources should import");
        let bytes = palette
            .to_binary_vec()
            .expect("built-in JSON palette should encode");
        assert_eq!(bytes, RenderPalette::builtin_palette_cache_bytes());
    }

    #[test]
    fn builtin_biome_schema_keeps_defaults_out_of_biome_ids() {
        let value = serde_json::from_str::<Value>(RenderPalette::builtin_biome_color_json())
            .expect("built-in biome palette JSON should parse");
        let biomes = value
            .get("biomes")
            .and_then(Value::as_object)
            .expect("built-in biome palette should use wrapper schema");
        assert!(value.get("defaults").is_some());
        assert!(!biomes.contains_key("default"));
        let mut ids = std::collections::BTreeSet::new();
        for (name, entry) in biomes {
            let id = entry
                .get("id")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("{name} should have an integer id"));
            assert!(ids.insert(id), "duplicate biome id {id} at {name}");
        }
    }

    #[test]
    fn builtin_palette_sources_are_embedded() {
        assert!(RenderPalette::builtin_block_color_json().contains("grass"));
        assert!(RenderPalette::builtin_biome_color_json().contains("ocean"));
        assert!(RenderPalette::builtin_palette_cache_bytes().starts_with(PALETTE_CACHE_MAGIC));
    }

    #[test]
    fn common_plant_blocks_are_known_colors() {
        let palette = RenderPalette::default();
        for name in [
            "minecraft:short_grass",
            "minecraft:tall_grass",
            "minecraft:double_plant",
            "minecraft:pink_petals",
            "minecraft:carrots",
            "minecraft:oak_trapdoor",
            "minecraft:cyan_concrete",
            "minecraft:deepslate_diamond_ore",
            "minecraft:crafting_table",
            "minecraft:farmland",
            "minecraft:glow_lichen",
            "minecraft:bubble_column",
            "minecraft:piston",
            "minecraft:reeds",
            "minecraft:fire",
            "minecraft:scaffolding",
            "minecraft:shroomlight",
            "minecraft:weeping_vines",
            "minecraft:end_portal",
            "minecraft:iron_bars",
            "minecraft:hopper",
            "minecraft:lever",
        ] {
            assert!(
                palette.has_block_color(name),
                "{name} should not be unknown"
            );
        }
    }

    #[test]
    fn height_shading_preserves_alpha_and_changes_brightness() {
        let palette = RenderPalette::default();
        let base = RgbaColor::new(100, 120, 140, 200);
        let brighter = palette.height_shaded_color(base, 80, 70);
        let darker = palette.height_shaded_color(base, 60, 70);

        assert_eq!(brighter.alpha, 200);
        assert_eq!(darker.alpha, 200);
        assert!(brighter.red > base.red);
        assert!(darker.red < base.red);
    }
}
