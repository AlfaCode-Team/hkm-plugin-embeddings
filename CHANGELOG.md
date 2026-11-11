# Changelog

All notable changes to `hkm-plugin-embeddings` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **First release: `EmbeddingPort` behind a content-hash cache.** Solves
  `embedding.generation`. `embedMany()` sends only the distinct cache misses to the
  provider, in one call per batch, and returns vectors in input order. The key is
  `sha256(model, dimensions, text)`, so a changed text or model misses and nothing else
  does; the value is lossless packed float64.
- **Two drivers: `local` and `openai`.** `local` is a deterministic feature-hashing
  embedder with no semantic meaning, pinned by a known-answer test computed in Python.
  `openai` calls any OpenAI-compatible `/embeddings` through `HttpClientPort`; **no live
  provider was ever called** — it is verified against a recording client.
- **`EMBEDDINGS_DRIVER` is required and has no default**, so a production system cannot
  end up on meaningless vectors, or on a paid API, by omission.
- **The API key is not held as a string.** `EmbeddingsConfig` fetches it from `env()` when
  the header is built; `var_export`, `json_encode`, `print_r`, `var_dump` and `serialize`
  cannot show it. Provider error text is never copied into an exception.
- **`embeddings:probe`** prints the configuration and, for `openai`, the masked request it
  would send; `--live` is required for the billed call.
