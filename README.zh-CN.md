# bedrock-render

[English](README.md) | [简体中文](README.zh-CN.md)

`bedrock-render` 是 Minecraft Bedrock 世界瓦片渲染库。它依赖
`bedrock-world` 查询 Bedrock LevelDB、NBT、chunk、subchunk、高度图和 biome 数据。
渲染、调色板、瓦片规划、图片编码、取消、线程、预览和 benchmark 独立在这个 crate
中维护，避免把图片依赖塞进核心世界解析库。

本仓库按独立 checkout 使用整理。当前 MSRV 是 Rust 1.88，默认特性包含
`async`、`webp` 和 `gpu`。CPU-only 用户可以使用 `--no-default-features`，再按需启用
图片格式特性。

## 设计

- 所有渲染输出都是 X/Z 平面图。`Y` 只表示采样层或高度选择参数。
- 瓦片由 `RenderLayout` 控制。默认布局是每张图 `16x16` 个 chunk，每像素一个方块，
  输出 `256x256` 图片。更大的 `blocks_per_pixel` 会保持同样世界范围，但用更低细节
  更快渲染大地图预览。
- `ChunkRegion` 支持任意矩形 chunk 区域。
- Web-map 路径稳定且能安全失效缓存：
  `<world>/<signature>/r<renderer>-p<palette>/<dimension>/<mode>/<layout>/<tile_x>/<tile_z>.<ext>`。
- 批量 tile 使用有界 worker。`Auto` 会根据执行 profile 解析：`Export` 使用当前机器逻辑
  CPU 数，`Interactive` 使用更保守的前台安全线程数；显式线程数支持 `1..=512`。
- Web-map 导出使用全局 chunk bake 队列、按内存预算分 wave、并行 tile compose/encode
  和有界 MPMC writer queue，避免 CPU worker 被 `fs::write` 串行阻塞。
- `RenderMemoryBudget::Auto` 会给 chunk bake 和导出 wave 设置有界缓存预算；
  离线工具也可以使用 `FixedBytes` 或 `Disabled`。
- 长任务支持显式取消和进度回调。

## 渲染模式

- `RenderMode::Biome { y }`：指定 Y 层的 biome 色彩图。
- `RenderMode::RawBiomeLayer { y }`：调试用 biome id 色彩图。
- `RenderMode::LayerBlocks { y }`：指定世界 Y 层的方块平面图。
- `RenderMode::SurfaceBlocks`：主俯视地形图。每个 X/Z 列从 Bedrock height map
  开始，向下寻找最高可渲染方块，应用 biome tint，并把透明水体和下方实体方块混合。
- `RenderMode::HeightMap`：基于 Bedrock Data2D/Data3D 的高度渐变图。
- `RenderMode::CaveSlice { y }`：指定 Y 层洞穴诊断图，区分空气、实体方块、水和熔岩。

`SurfaceBlocks` 对齐 BedrockMap 的地形 bake 核心流程：先使用
`chunk.get_height(x,z)`，再在该列向下寻找可渲染方块，同时烘焙 terrain/biome/height
数据。默认地形预览会叠加轻量高度法线明暗，用于表现坡度和地形起伏；完整高度分析仍通过
独立 `HeightMap` 输出。缺失 chunk 或缺失 height 记录被视为没有地形，
不会误渲染成灰色地图块。
固定 Y 层调试继续使用 `LayerBlocks { y }`。
未知方块会渲染为不透明紫色诊断像素；缺失 chunk、缺失 height map 和空地形会渲染为
透明像素，并在 `RenderDiagnostics` 中单独计数。

## API 示例

```rust
use std::sync::Arc;
use bedrock_render::{
    ChunkRegion, ImageFormat, MapRenderer, RenderExecutionProfile, RenderLayout,
    RenderMemoryBudget, RenderMode, RenderOptions, RenderPalette, RenderThreadingOptions,
};

let renderer = MapRenderer::new(Arc::new(world), RenderPalette::default());
let region = ChunkRegion::new(dimension, -32, -32, 31, 31);
let layout = RenderLayout {
    chunks_per_tile: 16,
    blocks_per_pixel: 1,
    pixels_per_block: 1,
};

let tiles = renderer.render_region_tiles_blocking(
    region,
    RenderMode::HeightMap,
    layout,
    RenderOptions {
        format: ImageFormat::WebP,
        threading: RenderThreadingOptions::Auto,
        execution_profile: RenderExecutionProfile::Export,
        memory_budget: RenderMemoryBudget::Auto,
        ..RenderOptions::default()
    },
)?;
```

## 预览工具

预览 example 会生成 6 张 atlas PNG 和 web-map tile 目录：

```text
cargo run --example render_preview --features png
```

可选参数：

```text
render_preview <world_path> <output_dir> <center_tile_x> <center_tile_z> \
  <viewport_tiles> <layer_y> <cave_y> <chunks_per_tile>
```

预览输出结构：

```text
<output_dir>/
  biome-viewport.png
  raw-biome-viewport.png
  layer-y64-viewport.png
  surface-viewport.png
  heightmap-viewport.png
  cave-y32-viewport.png
  web-tiles/sample/signature/r2-p1/overworld/heightmap/16c-1bpp/21/12.png
```

## Web 地图导出

`render_web_map` 会导出 WebP 瓦片和一个自包含的静态 HTML 查看器。它用于验证渲染器，
也可以生成可分享的 web-map 产物，不依赖 GPUI 或 CDN。未传 `--region` 时，会自动
发现并渲染每个选中维度的已加载 chunk bounds。

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map \
  --dimensions overworld,nether,end \
  --mode surface,heightmap,biome,layer \
  --chunks-per-tile 16 \
  --chunks-per-region 32 \
  --blocks-per-pixel 4 \
  --threads auto \
  --profile export \
  --memory-budget auto \
  --pipeline-depth 256 \
  --tile-batch-size auto \
  --writer-threads 2 \
  --write-queue-capacity 256 \
  --stats \
  --force
```

输出结构：

```text
target/bedrock-web-map/
  viewer.html
  map-layout.json
  map-data.js
  tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp
```

`map-layout.json` 是自动生成的地图布局数据，供工具或其它前端读取；`map-data.js`
只内置最小布局常量，并动态提供 `tileBounds()`、`tileId()`、`tilePath()`。瓦片位置由固定
目录规则 `tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp` 推导，HTML 查看器
直接通过普通脚本加载它。这样从 `file://` 打开 `viewer.html` 时不需要 `fetch()` JSON，
也不会触发浏览器 CORS。导出会在打开和扫描世界前先写出 `viewer.html`、`map-layout.json`
和 `map-data.js`；后续发现维度 bounds、完成每个模式渲染时，`map-layout.json`/
`map-data.js` 会逐步刷新布局和统计数据。查看器会定时重新加载 `map-data.js` 并重试
当前视口瓦片，所以导出仍在运行时也能逐步看到新写入的图片。
它支持维度切换、模式切换、拖拽平移、滚轮缩放、瓦片坐标和加载失败提示。
大地图导出前建议使用 `--region` 限制范围，或提高 `--blocks-per-pixel` 降低输出体积。
8 核 16 线程机器推荐默认 `--profile export --threads auto`；UI 或交互式预览使用
`--profile interactive`。只有离线导出且磁盘能跟上时，才建议手动提高到更大的固定线程数，
最大 512。`--stats` 会输出 planned tiles、unique chunks、baked chunks、bake/encode/write
耗时、cache 命中和峰值缓存字节，用于判断 CPU 占用不足到底来自 I/O、编码还是 chunk bake。
缺失 chunk、缺失 subchunk、固定 Y 层未加载区域都会透明；不透明紫色表示已经读取到
真实方块名，但当前 palette 没有对应颜色映射。

## 外部调色板

默认 palette 已内置到 crate 和编译后的二进制中。它包含两份可审计 JSON 源文件，
以及一个预构建的 `BRPAL01` 二进制缓存：

```text
data/colors/bedrock-block-color.json
data/colors/bedrock-biome-color.json
data/colors/bedrock-colors.brpal
```

`RenderPalette::default()` 会优先加载内置二进制缓存，所以普通渲染不需要解析 JSON，
也不需要外部 palette 文件。JSON 仍可以通过
`RenderPalette::builtin_block_color_json()` 和
`RenderPalette::builtin_biome_color_json()` 获取，用于审计和重建工具。
`RenderPalette::from_builtin_json_sources()` 可以只从可审计 JSON 源重建同一份 palette，
不读取二进制缓存。内置数据由本项目
维护用于 Bedrock 渲染；loader 同时理解公开 bedrock-level 颜色 JSON 的结构，便于在
项目许可策略允许时重建或对比 palette。

如果项目拥有自己的合法颜色数据，`RenderPalette` 也支持额外用户 JSON 覆盖，并支持一个
紧凑的二进制缓存格式：

```text
--palette-json target/bedrock-block-color.json
--palette-json target/bedrock-biome-color.json
--palette-cache target/colors.brpal
--rebuild-palette-cache
```

JSON 是导入/覆盖格式。内置 `BRPAL01` 二进制缓存是应用热路径推荐格式：它是顺序读取的
biome id 和 block-name RGBA 表，避免启动渲染时反复解析 JSON。JSON loader 支持带
`schema_version` / `sources` / `blocks` / `defaults` / `biomes` 的组合对象、
`{"minecraft:stone":"#7d7d7d"}` 这样的对象映射，
包含 `name` / `id` 和 `color` 字段的数组，也支持 bedrock-level 风格的颜色 JSON
结构作为用户本地参考数据。

Palette 源文件维护命令：

```text
cargo run --example palette_tool -- audit --check
cargo run --example palette_tool -- normalize --check
cargo run --example palette_tool -- rebuild-cache --check
```

来源策略和公开参考资料见 [docs/PALETTE_SOURCES.md](docs/PALETTE_SOURCES.md)。

最近一次本地参考 palette smoke 运行：

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map-ref-palette \
  --region 0,0,15,15 \
  --mode surface,heightmap,biome,layer \
  --y 64 \
  --palette-json target/bedrock-block-color.json \
  --palette-json target/bedrock-biome-color.json \
  --palette-cache target/bedrock-colors.brpal \
  --rebuild-palette-cache \
  --force

loaded palette JSON target\bedrock-block-color.json: block_colors=1207 biome_colors=0 skipped=0
loaded palette JSON target\bedrock-biome-color.json: block_colors=0 biome_colors=88 skipped=0
wrote palette cache: target\bedrock-colors.brpal
overworld surface tiles=1 missing=0 transparent=0 unknown=0
overworld layer-y64 tiles=1 missing=0 transparent=12544 unknown=0
```

最近一次本地 debug 预览结果：

```text
cargo run --example render_preview --features png
生成 6 张 atlas 图片和 54 个 web-map PNG tile。
视口：3x3 tile，每 tile 16 个 chunk，每 tile 256x256 px。
surface diagnostics: missing_chunks=0 missing_heightmaps=0 unknown_blocks=0 fallback_pixels=0
```

最近一次本地 WebP web-map smoke 运行：

```text
cargo run --example render_web_map -- \
  --world ../bedrock-world/tests/fixtures/sample-bedrock-world \
  --out target/bedrock-web-map \
  --region 0,0,15,15 \
  --mode surface,heightmap,biome,layer \
  --threads auto \
  --tile-batch-size auto \
  --writer-threads 2 \
  --write-queue-capacity 64 \
  --force

已生成 viewer.html、map-layout.json、map-data.js，以及 overworld/nether/end 的 WebP 瓦片。
LayerBlocks 的透明像素表示固定 Y 层未加载区域，这是预期行为。
```

## 渲染效果示例

这些图片由预览工具基于内置 sample fixture 和当前默认调色板生成。它们是文档示例，
不是完整 BedrockMap 等价输出。

### `Biome { y }`

在指定 Y 层采样的 X/Z 平面 biome 色彩图。

![Biome map](docs/images/biome-viewport.png)

### `RawBiomeLayer { y }`

调试用 biome id 色彩图。未知或稀疏 id 会被有意显示出来。

![Raw biome map](docs/images/raw-biome-viewport.png)

### `LayerBlocks { y }`

指定世界 Y 层的 X/Z 方块平面图。这个模式不是侧面剖切图。

![Fixed Y layer map](docs/images/layer-y64-viewport.png)

### `SurfaceBlocks`

主俯视地形图。每个 X/Z 列使用 chunk height map，向下寻找最高可渲染方块，
应用 biome tint，并把透明水体与下方实体方块混合。默认叠加轻量高度法线明暗，避免地形
看起来完全平面；高度数值分析请使用 `HeightMap`。

![Surface block map](docs/images/surface-viewport.png)

### `HeightMap`

基于 Bedrock Data2D/Data3D 高度记录生成的高度渐变图。

![Height map](docs/images/heightmap-viewport.png)

### `CaveSlice { y }`

指定 Y 层洞穴诊断图，用于区分空气、实体方块、水和熔岩。

![Cave slice map](docs/images/cave-y32-viewport.png)

## 性能模型

- `Biome`、`RawBiomeLayer`、`LayerBlocks`、`HeightMap`、`CaveSlice` 只预取当前 tile
  需要的 chunk record。
- 高度数据通过 `bedrock-world` 每 chunk 读取一次，并在 tile 上下文中缓存成紧凑的
  `16x16` 高度数组。
- `SurfaceBlocks` 不再对缺失 height 的列从世界顶部向下扫描。缺失 chunk/height 会进入
  diagnostics，并按无地形处理。
- Subchunk 访问统一走 `SubChunk::block_state_at(local_x, local_y, local_z)`，避免
  renderer 内复制 palette index 公式。
- Web 导出使用 region-first bake 管线。`--chunks-per-region` 控制烘焙缓存单元；
  大地图默认推荐 `32`，小范围验证或交互预览需要更低首屏延迟时可用 `16`。
- Web 导出不会把完整地图的所有 region bake 常驻内存；renderer 会按
  `--memory-budget` 分 wave 烘焙、合成、写盘。`--pipeline-depth` 和
  `--write-queue-capacity` 只限制已编码瓦片等待写盘的队列深度。
- `SurfaceBlocks` 使用 chunk bake：先把每个 chunk 规约成 `16x16` 地形图，再把
  baked chunk 像素拼成 region plane 和 WebP tile。
- BMCBL 查看器使用 `RenderLayout` 自动比例。小地图默认全细节，大地图可用 2/4/8
  blocks per pixel 快速预览，但不改变单个 tile 覆盖的世界范围。
- 缓存 key 包含世界路径 hash、世界文件签名、renderer 版本、palette 版本、维度、模式、
  Y 层和布局。UI 会拒绝全透明的陈旧缓存瓦片并重新生成。
- `ImageFormat::Rgba` 是 UI 纹理最低延迟路径；WebP/PNG 适合缓存、导出和预览。
- 静态 web-map example 会把 WebP 瓦片写到
  `tiles/<dimension>/<mode>/<layout>/<tile_z>/<tile_x>.webp`；瓦片之间不会有缝隙，
  缺失世界数据用透明像素表示。
- BMCBL 的 GPUI 地图查看窗口会以世界 `level.dat` 的重生点为初始中心，支持维度/
  自定义维度切换，并把 WebP web-map tile 写入系统临时缓存目录。

## 测试和 Benchmark

```text
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps --all-features
cargo rustdoc --lib --all-features -- -D missing_docs
cargo bench --bench render --all-features
```

默认 benchmark 套件覆盖瓦片渲染、chunk bake 和小批量渲染。完整 web-map 导出 benchmark
需要显式开启：

```text
$env:BEDROCK_RENDER_FULL_BENCH='1'
cargo bench --bench render --all-features
Remove-Item Env:\BEDROCK_RENDER_FULL_BENCH
```

更多细节见 [docs/API.md](docs/API.md)、[docs/TESTING.md](docs/TESTING.md) 和
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

## 当前限制

- GPUI 查看器是启动器内的预览窗口，不是完整 BedrockMap 兼容 GIS 应用。
- Surface 渲染是 BedrockMap 风格顶视图，web-map 导出现在使用全局 chunk bake 队列；
  跨进程持久化 chunk bake 复用仍是后续优化项。
- V1 不实现光照、阴影、高度混合、实体标记、文字标签或完整 BedrockMap 等价功能。
