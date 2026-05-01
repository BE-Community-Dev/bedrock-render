# Palette Sources

`bedrock-render` treats the JSON palette files as auditable source data and
`data/colors/bedrock-colors.brpal` as the runtime cache generated from those
sources.

## Files

- `data/colors/bedrock-block-color.json` stores block texture candidates keyed
  by block name. Multi-texture entries are intentionally preserved so the loader
  can choose the best map color without losing source detail.
- `data/colors/bedrock-biome-color.json` stores biome base colors and optional
  grass, foliage, and water tint overrides. Shared fallback tint values live in
  `defaults`, not as a synthetic biome id.
- `data/colors/bedrock-colors.brpal` is the compact `BRPAL01` cache generated
  from the default block and biome sources plus built-in compatibility defaults.

## Source Policy

Every palette JSON document contains:

- `schema_version`
- `minecraft_bedrock_version`
- `sources`
- either `blocks` or `defaults` plus `biomes`

Current color values are marked as `legacy-current` where public references do
not provide a directly reusable color table. Public references are used for
schema, semantics, and compatibility checks unless their license clearly allows
copying concrete color values.

Referenced public material:

- Microsoft Learn `minecraft:map_color` component reference:
  <https://learn.microsoft.com/minecraft/creator/reference/content/blockreference/examples/blockcomponents/minecraftblock_map_color?view=minecraft-bedrock-experimental>
- Mojang Bedrock Samples client biome reference:
  <https://mojang.github.io/bedrock-samples/Client%20Biomes.html>
- Microsoft Minecraft samples repository:
  <https://github.com/microsoft/minecraft-samples>

## Maintenance Commands

Audit source shape, metadata, normalization, and cache sync:

```powershell
cargo run --example palette_tool -- audit --check
```

Check whether the JSON files already match the canonical schema/ordering:

```powershell
cargo run --example palette_tool -- normalize --check
```

Rewrite the JSON files into canonical form:

```powershell
cargo run --example palette_tool -- normalize
```

Check whether the embedded cache matches the JSON sources:

```powershell
cargo run --example palette_tool -- rebuild-cache --check
```

Regenerate the embedded cache after intentional palette changes:

```powershell
cargo run --example palette_tool -- rebuild-cache
```

If a color change affects rendered output, update preview images and bump
`DEFAULT_PALETTE_VERSION` so tile caches are invalidated.
