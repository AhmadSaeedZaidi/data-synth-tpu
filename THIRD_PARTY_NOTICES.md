# Third-party notices

`kaggle/vendor/glm53/` derives from ARahim3/kaggle-tpu-lab at commit
`1aa1f083ae253470d9f355fe9c3eb12003300e91`. Its MIT license is retained at
`kaggle/vendor/LICENSE`; source provenance and local modifications are recorded
in `kaggle/vendor/PROVENANCE.json`.

GGUF decoder reference fixtures were generated independently with gguf-py.
See `kaggle/REFERENCE_PROVENANCE.json`. Qwen's small synthetic reference fixtures
record their generation metadata in `kaggle/qwen38_tpu/fixtures/`. These are
small random-weight numerical fixtures, not distributed pretrained checkpoints.

The Qwen implementation was studied against official model/Transformers sources
and the Strata project. Checkpoint manifests identify the external model
repositories and revisions. No pretrained weights are included. Applicable
upstream notices and licenses continue to apply to downloaded model assets and
dependencies independently of this repository's license.

Upstream license texts retained for the quantization/model reference lineage:

- [Strata MIT license](licenses/Strata-MIT.txt), commit
  `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
  https://github.com/Niko1221/Strata.
- [llama.cpp / ggml MIT license](licenses/llama.cpp-MIT.txt),
  https://github.com/ggml-org/llama.cpp.
- [Transformers Apache 2.0 license](licenses/Transformers-Apache-2.0.txt),
  reference commit `536ecc007387a50e77603bb5d92100e9b07514cc`,
  https://github.com/huggingface/transformers.

These notices preserve upstream attribution; they do not imply endorsement.
