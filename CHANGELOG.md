# Changelog

All notable changes to `bedrock-render` are tracked here.

## 0.1.0 - 2026-05-01

### Added

- Initial public crate-ready tile renderer for Minecraft Bedrock worlds.
- Surface, height map, biome, raw biome, fixed layer, and cave slice render modes.
- Embedded Bedrock block and biome color palette sources plus a compact `BRPAL01`
  palette cache for fast startup.
- Palette source schema metadata, source-policy documentation, JSON-only palette
  rebuild support, and a `palette_tool` example for audit/normalize/cache checks.
- Region planning, deterministic tile paths, render diagnostics, cancellation,
  progress callbacks, bounded threading, memory-budgeted region baking, and
  optional GPU terrain-light compose.
- PNG/WebP/RGBA output support, preview and static web-map examples, fixture tests,
  Criterion benches, and English/Simplified Chinese documentation.

### Notes

- The crate depends on `bedrock-world` by pinned Git revision until the Bedrock
  crate family is ready for crates.io publishing.
- Real Bedrock worlds and generated render outputs are intentionally excluded
  from version control.
