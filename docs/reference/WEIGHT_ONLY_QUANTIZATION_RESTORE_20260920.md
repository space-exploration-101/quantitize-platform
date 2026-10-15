# Weight-only quantization restoration (2026-09-20)

## Reason

The verified golden task `20260902_065807_test_h200` used effective
weight-only quantization: 148 Conv weight `DequantizeLinear` nodes and no
activation `QuantizeLinear` or activation `DequantizeLinear` nodes. Its model
input fed the first Conv directly.

The Pose21 Gray1 task unexpectedly contained 137 activation QDQ pairs. The
input pair used `images_scale=1.0`, reducing normalized gray input to two
levels before the first Conv. One nonzero detection-head weight channel was
also quantized entirely to zero with a scale of 1.0. This occurred before
`generate_bin` and explains the PT-to-quantized-ONNX accuracy loss.

The local ONNX Runtime modules were present and logged as loaded, but importing
the public quantization package had already populated `QDQRegistry` and bound
scale helpers. Replacing entries in `sys.modules` did not replace those cached
class and function references. The live Conv registry therefore continued to
use the system QDQ implementation instead of the local weight-only Conv class.

## Change

- Rebind `QDQRegistry["Conv"]` and `QDQRegistry["ConvTranspose"]` to the local
  weight-only `QDQConv` after loading the patch.
- Rebind the eager ONNX Runtime scale helper to the local small-scale
  implementation.
- Restrict quantization candidates explicitly to `Conv`.
- Remove the unused `tensor_quant_overrides` variable that claimed to protect
  `images` but was never passed to ONNX Runtime.
- Add a post-quantization contract check. A task now fails during quantization
  if it contains any activation QDQ, a Conv weight without DQ, an invalid
  scale, or a nonzero source-weight channel that becomes entirely zero.

The `generate_bin` exact-tensor compatibility fix remains in place. It does
not add QDQ to the production model and is independent of this restoration.

## Verification

- Synthetic FP16 Conv regression: weight DQ only, no activation QDQ, no
  `scale=1.0`, and no nonzero weight channel collapsed to zero.
- Full runner unit suite: 19/19 passed.
- Golden task model contract: `Conv=148`, `weight_DQ=148`,
  `activation_QDQ=0`; passed.
- Faulty Pose21 model contract: rejected with 137 activation Q nodes, 137
  activation DQ nodes, and collapsed channel
  `/model.30/cv3.1/cv3.1.0/conv/Conv[94]`.

## Pose21 production validation

Task `20260920_063903_Pose21_Gray1_weight-only_accuracy_valida` reused the
model and the 462-image calibration and evaluation datasets from
`20260920_035432_Pose21_Gray1_50e_cal462_QDQ_fix_rerun`.

The real model passed the structural contract:

```text
Conv=148, weight_DQ=148, activation_QDQ=0
```

PT and weight-only ONNX produced exactly the same detection metrics:

| Metric | PT | Weight-only ONNX |
| --- | ---: | ---: |
| Precision | 0.9143646409 | 0.9143646409 |
| Recall | 0.8756613757 | 0.8756613757 |
| F1 | 0.8945945946 | 0.8945945946 |
| mAP | 0.8117998498 | 0.8117998498 |
| TP / FP / FN | 331 / 31 / 47 | 331 / 31 / 47 |

Mean pixel error changed from `6.11064 px` to `6.16636 px` (`+0.05572 px`),
without changing any aggregate detection count or metric. The PT-to-ONNX
accuracy regression is therefore considered restored.

The entire pipeline also completed successfully: all eight steps were
`completed`, `generate_bin` completed in 23.172 seconds, the bundle ZIP was
created, scratch was cleaned, and both task and worker error fields were
empty.

### FPGA input roundtrip

The `fpga_input_roundtrip_eval` stage converted each source image through the
production input path (`2000x2000 png2bin -> bin2png side view`), resized the
side view to the model's `1280x1280` input, and ran the same weight-only ONNX
model. Compared with direct ONNX evaluation:

| Metric | Direct ONNX | FPGA input roundtrip | Delta |
| --- | ---: | ---: | ---: |
| Precision | 0.9143646409 | 0.9073569482 | -0.0070076927 |
| Recall | 0.8756613757 | 0.8809523810 | +0.0052910053 |
| F1 | 0.8945945946 | 0.8939597315 | -0.0006348631 |
| mAP | 0.8117998498 | 0.8097521952 | -0.0020476546 |
| TP / FP / FN | 331 / 31 / 47 | 333 / 34 / 45 | +2 / +3 / -2 |

Mean pixel error improved slightly from `6.16636 px` to `6.09158 px`. The
input roundtrip therefore introduces only a small metric fluctuation, not the
large regression seen in the faulty activation-QDQ model.

This stage validates only the FPGA image packing and recovery path. Inference
still runs through ONNX Runtime on CPU; it does not validate numerical error
from real FPGA execution or a layer-by-layer hardware simulator. That final
validation requires FPGA-produced detection output followed by
`fpga_test_pack/scripts/eval_fpga_results.py`.

## Deployment

The restoration was deployed as
`quantitize-platform-api:weight-only-20260920`
(`sha256:f633b01c2f85...`) and tagged as `quantitize-platform-api:latest`.
The preceding generate-bin hotfix image is retained as
`quantitize-platform-api:pre-weight-only-20260920` for rollback. Only the API
container was recreated; the Web container was not restarted. Both services
were healthy after deployment, and the full 19-test suite passed again inside
the deployed API container.
