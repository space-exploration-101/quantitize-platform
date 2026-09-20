# Generate-bin QDQ input incident (2026-09-18)

## Scope and evidence

Task `20260918_172417_Pose21_Gray1_50e_cal462` completed `pt_eval`, `quantize`,
`onnx_eval`, `fpga_test_pack`, and `fpga_input_roundtrip_eval`, then failed in
`generate_bin`. Twenty-one logical layers reported `KeyError: 'input'`.
`merge_bin` and `bundle` did not run. The verified failure archive contains 950
files and 5,215,649,583 bytes.

The failure was not caused by GPU availability, disk capacity, or service
health. The quantized graph contained every required Conv input tensor, but its
QDQ activation inputs (for example
`/model.0/act/Mul_output_0_DequantizeLinear_Output`) were not declared as graph
outputs and had no `value_info`. ONNX Runtime therefore computed those tensors
internally but did not return them to the generator. The generator then read a
missing `result['input']` entry.

## Numbering and ownership

`T32_Lxx_sxx` is a legacy export identifier:

- `T32` means feature and weight data are packed in 32-channel hardware tiles.
- `Lxx` normally groups operations by the top-level `/model.xx/` module.
- `sxx` numbers exportable operations within that module in ONNX node order.
- Pose head branches are a special mapping: twelve branch pipelines become
  `L31` through `L42`, with their internal convolutions numbered by `sxx`.

The identifier is not an ONNX node index and is not always a one-to-one FPGA
instruction number.

## Repair contract

`generate_bin` must export the exact activation tensor consumed by each Conv.
It must not strip `_DequantizeLinear_Output` and substitute the pre-quantized
tensor, because Q/DQ rounding and saturation make those values non-equivalent.

The generator now:

1. clones the quantized ONNX model in memory;
2. finds the data input of every Conv;
3. resolves missing DQ metadata only through the verified
   `DequantizeLinear <- QuantizeLinear <- float tensor` chain;
4. appends the exact DQ tensor to the clone's graph outputs;
5. runs inference once and reads tensors by exact name;
6. validates required input, output, weight, bias, and scale fields before
   writing a layer;
7. writes to a task-local `.partial.<pid>` directory, creates a SHA-256
   inventory, and atomically publishes the final `bin` directory.

The production quantized ONNX file is not rewritten.

## Acceptance

- Synthetic QDQ regression proves the exported Conv input equals the DQ result
  and differs from the original float input.
- The archived failing model completes `generate_bin` without an `input`
  fallback or missing layer error.
- Two identical archive regressions have the same file inventory and SHA-256.
- A controlled full task completes all eight stages, has empty task and worker
  errors, and produces a bundle ZIP.
- Existing PT/ONNX/roundtrip metrics are recorded separately; successful BIN
  generation does not by itself claim accuracy or hardware validation.

## Implementation and regression results

Implemented on 2026-09-20 in `pipeline/engine/check_certain_layer_multi.py`.
The change keeps the production ONNX immutable, exposes exact Conv inputs only
on an in-memory clone, rejects incomplete tensor records with node and tensor
details, and publishes the generated directory atomically after writing its
SHA-256 inventory.

Verification results:

- QDQ-focused unit tests: 3/3 passed.
- Full runner test suite: 17/17 passed.
- The archived failing model completed `generate_bin` twice. Each result had
  154 directories and 767 generated data files (about 3.0 GB), with 137 exact
  Conv input tensors instrumented.
- The two archive regressions had identical relative paths, sizes, and all 767
  file SHA-256 values. Neither result left a `.partial` directory.

The normal slim task bundle intentionally excludes `workspace/bin/`, including
the generator-local inventory, because those files are intermediate debug
artifacts. The merged `renamed_weights_bin/` and `all_bin/` outputs remain in
the deliverable. The two retained regression directories contain the complete
generator inventories used for the deterministic comparison.

## Deployment and controlled rerun

The API was deployed from hotfix image
`quantitize-platform-api:generate-bin-qdq-20260920`
(`sha256:34537dbbb8e1...`). The prior image is retained as
`quantitize-platform-api:pre-generate-bin-qdq-20260920`
(`sha256:4693858842e1...`) for rollback. Only the API container was recreated;
the Web container was not restarted.

The standard Dockerfile rebuild could not complete because its declared base
image `ywang/yolo-gray1-quant:ultralytics-8.3.98-v6` was not present locally
and the configured remote mirror timed out. No global Docker configuration was
changed. A reproducible rebuild from the declared base image remains an
infrastructure follow-up; it does not invalidate the runtime verification of
the deployed hotfix.

Controlled full rerun:

- Task: `20260920_035432_Pose21_Gray1_50e_cal462_QDQ_fix_rerun`
- Inputs: calibration dataset `pose21_gray1_50e_cal462_v1` (462 images), test
  dataset `pose21_gray1_50e_eval462_v2` (462 images), `nc=21`, `imgsz=1280`,
  preprocessing `passthrough`.
- All eight stages completed. `generate_bin` completed in 24.948 seconds,
  `merge_bin` in 2.889 seconds, and `bundle` in 11.777 seconds.
- `task_error` and `worker_error` were empty, `has_zip=true`, and scratch was
  cleaned only after ZIP verification.
- The bundle contains 490 files and 344,800,971 uncompressed bytes. `unzip -t`
  reported no errors. Bundle SHA-256:
  `d118db7b1513d22ffbede285953f28beef57f037afd582b25ab9960576bbe6a2`.
- API and Web were healthy after the run. Five consecutive GPU samples were
  idle with no compute processes after completion.

Accuracy is a separate unresolved issue:

| Evaluation | Precision | Recall | F1 | mAP |
| --- | ---: | ---: | ---: | ---: |
| PT | 0.914365 | 0.875661 | 0.894595 | 0.811800 |
| Quantized ONNX | 0.475410 | 0.076720 | 0.132118 | 0.046623 |
| FPGA input roundtrip | 0.551020 | 0.071429 | 0.126464 | 0.049191 |

The rerun proves that exact-QDQ BIN generation, merging, and packaging now
finish deterministically. It does not establish acceptable model accuracy or
hardware execution correctness; those require a separate quantization-quality
investigation and hardware validation.
