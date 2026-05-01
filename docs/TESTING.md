# Testing And Benchmarks

## Required Checks

```powershell
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps --all-features
cargo rustdoc --lib --all-features -- -D missing_docs
cargo run --example palette_tool -- audit --check
cargo run --example palette_tool -- normalize --check
cargo run --example palette_tool -- rebuild-cache --check
```

These checks are expected to pass on a fresh checkout without private world data.

## Optional Fixture

The integration test and benchmarks look for the sample Bedrock world in the
adjacent `bedrock-world` checkout:

```text
../bedrock-world/tests/fixtures/sample-bedrock-world
```

The folder should contain `level.dat` and `db/CURRENT`. If it is missing,
fixture tests and benches skip their world-backed cases instead of failing.

Do not commit real worlds. Add small in-memory tests for regressions whenever
possible.

## Feature Coverage

Run `cargo test --no-default-features` to validate CPU-only compilation. Run
`cargo test --all-features` for async, WebP, PNG, and GPU code paths.

GPU execution depends on host hardware and drivers. Tests should assert CPU
fallback behavior and diagnostics rather than requiring a GPU device.
