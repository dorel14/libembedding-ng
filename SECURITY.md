# Security policy

## Supported versions

Only the latest 1.x release receives security fixes. Upgrade to the latest release
before reporting.

| Version | Supported |
|---|---|
| latest 1.x | yes |
| older 1.x | no |

## Reporting a vulnerability

Do not open a public issue. Use GitHub private vulnerability reporting:
<https://github.com/dorel14/libembedding-ng/security/advisories/new>

Include the library version, OS/architecture, backend (`.onnx` or `.gguf`), the model
file involved and a minimal reproducer (a crashing input file is ideal). You will get
an acknowledgement on the advisory thread; fixes ship in a patch release.

## Untrusted model files

Model files are inputs. Treat files from unknown sources as untrusted.

### GGUF (sparse runtime, in-house ggml)

The loader in `include/libembedding/detail/gguf/gguf_weights.hpp` refuses, with an
error instead of undefined behaviour:

- a tensor table that declares more data than the file holds (checked before any
  allocation);
- hyper-parameters above fixed bounds (`kMaxEmbd`, `kMaxHeads`, `kMaxLayers`,
  `kMaxFf`, `kMaxVocab`, `kMaxPositions`), before any allocation sized by them;
- a `block_count` that the file's tensor count cannot satisfy;
- a missing required tensor or a missing capability, by name;
- metadata of the wrong type or width (`gguf_probe.hpp` checks each key's type).

Quantised tensors are never read past `ggml_nbytes`.

The loader is fuzzed with libFuzzer, ASan and UBSan
(`.github/workflows/fuzz.yml`, harness `tests/fuzz/fuzz_gguf_loader.cpp`, 60 s per
pull request touching the loader).

Dense and reranker GGUF models are loaded by the vendored llama.cpp, which has its own
parser. We do not add validation in front of it.

### ONNX and tokenizer.json

- `.onnx` files are parsed by ONNX Runtime. libembedding does not validate their
  graph. Only load models you trust.
- `tokenizer.json` is parsed by the library's tokenizer
  (`include/libembedding/detail/tokenizer_impl.hpp`). Changes to it trigger the fuzz
  workflow, but the fuzz harness targets the GGUF loader, not the tokenizer.

### Downloads

Models are downloaded from Hugging Face over HTTPS (libcurl). Build with
`-DLIBEMBEDDING_NO_DOWNLOAD=ON` to remove the download code, or pass a local model
path.

## Model licences

The registry accepts MIT-compatible licences only. Entries under a restricted licence
(currently `jinaai/jina-reranker-v2-base-multilingual`, CC-BY-NC-4.0, and
`onnx-community/embeddinggemma-300m-ONNX`, Gemma Terms of Use) are refused at load
time unless you opt in:

```powershell
$env:LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES = "1"
```

Opting in is your responsibility: read the model licence first. A local model path
never goes through this check. These entries leave the registry in 2.0 (see
`docs/adr/ADR-004-2-0-removals.md`).
