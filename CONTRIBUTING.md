# Contributing to AeroMESH

Thanks for your interest in contributing to AeroMESH.

AeroMESH is a distributed pipeline-parallel inference system for running
GGUF-based LLMs across multiple machines using Rust, Python, llama.cpp,
Tailscale, and custom binary activation transport.

Contributions are welcome for bug fixes, documentation improvements,
Windows/Linux compatibility, networking improvements, testing, and UI work.

## License

By contributing to AeroMESH, you agree that your contributions will be
licensed under the same dual license as the project:

- MIT License
- Apache License, Version 2.0

See `LICENSE`, `LICENSE-MIT`, and `LICENSE-APACHE` for details.

## Development Environment

AeroMESH currently targets Windows 10/11 with NVIDIA CUDA support.

Required tools:

- Rust 1.78+
- Python 3.11 or newer, preferably 3.12
- Git
- NVIDIA GPU drivers
- Tailscale, optional for mesh networking
- `uv`, optional but recommended for Python dependency management

Automated setup:

```powershell
.\setup.ps1
```

Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r pyproject.toml
```

Or with `uv`:

```powershell
uv sync
```

## Build

Build the Rust workspace:

```powershell
cargo build --release
```

Run the CLI help:

```powershell
cargo run --bin aeromesh -- --help
```

## Running Tests

Run all Rust tests:

```powershell
cargo test --workspace
```

Run formatting checks:

```powershell
cargo fmt --check
```

Run lint checks:

```powershell
cargo clippy --workspace --all-targets -- -D warnings
```

## Running a Local Test Mesh

Start a full local test mesh:

```powershell
cargo run --release --bin aeromesh -- start --role all --model "models/test.gguf"
```

Then start the web interface:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Code Style

### Rust

- Use Rust 2021 edition conventions.
- Use `snake_case` for functions, modules, and variables.
- Use `CamelCase` for types, structs, enums, and traits.
- Use `SCREAMING_SNAKE_CASE` for constants.
- Prefer explicit error types over panics.
- Use `thiserror` for library error types.
- Use `anyhow::Result` for application-level error propagation.
- Avoid panics in network loops, transport code, and pipeline loops.
- Add tests for protocol parsing, GGUF parsing, activation codec, and transport behavior.

### Python

- Target Python 3.12 where possible.
- Use type annotations for new functions.
- Use `snake_case` for functions and variables.
- Use `CamelCase` for classes.
- Wrap subprocess, network, and file-system calls with clear error handling.

### Frontend

- Keep the frontend dependency-free where possible.
- Use vanilla JavaScript, HTML5, and CSS3.
- Avoid adding heavyweight build systems unless clearly justified.
- Keep UI code readable and directly testable in the browser.

## Pull Request Guidelines

1. Keep changes focused.
2. Update documentation if behavior changes.
3. Add tests where practical.
4. Run `cargo fmt` and `cargo clippy`.
5. Run `cargo test --workspace`.
6. Describe the problem and verification steps in the PR.
7. Reference related issues, for example `Fixes #25` or `Fixes #27`.

## Security

For security-sensitive issues, do not open a public issue.
Follow the guidance in `SECURITY.md` if present.
