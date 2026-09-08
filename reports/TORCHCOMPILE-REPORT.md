# torch.compile on RF-DETR-Small: measured on the RTX 4070

Measured 2026-09-08. Eager PyTorch inference on the RF-DETR-Small litter
checkpoint (`checkpoint_best_total.pth`, run `v2-rfdetr-s-20260802T0348`),
against `torch.compile`. Additive: reads the checkpoint, touches nothing shipped.

**Environment.** Windows 11, RTX 4070 (sm_89), PyTorch **2.6.0+cu124**, CUDA
12.4, `rfdetr` 1.9.0, Python 3.12. Batch 1, input `[1,3,512,512]` from one real
fixture, on device, fp32. The underlying module is `LWDETR`
(`RFDETRSmall(...).model.model`), run under `torch.no_grad()` in `eval()`.

The short version: **`torch.compile` produced no runnable compiled model here,
for two independent reasons — one about the platform, one about the model — and
the second is the one worth knowing.**

---

## 1. Eager baseline

| run | p50 | p95 | min |
|---|---|---|---|
| eager run 1 | 11.8–12.4 ms | 16.9–17.3 ms | 11.2–11.4 ms |
| eager run 2 | 13.9 ms | 16.8–17.4 ms | 11.1–11.3 ms |

Run-to-run **p50 noise is 12–18%** across repeats of the identical call — large,
and the number any speedup claim has to clear. (For scale, the TensorRT engine's
own p50 noise was 0.7%; this is eager PyTorch under WDDM, and it wanders.) The
forward returns `{pred_logits: [1,300,44], pred_boxes: [1,300,4]}`.

## 2. The default backend cannot run on this host

`torch.compile(model)` and `torch.compile(model, mode="reduce-overhead")` both
raise before producing a kernel:

```
BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot find a working triton installation.
```

TorchInductor generates Triton kernels, and Triton has no official Windows
build. So on this Windows box the **default** `torch.compile` backend — the one
every "just wrap it in `torch.compile`" recommendation assumes — is unavailable,
and `mode="default"` vs `mode="reduce-overhead"` cannot be compared because
neither compiles. `max-autotune` was not attempted for the same reason. This is
a platform fact, not a model fact: the same call on a Linux box with Triton
would get past this line — and then hit §3.

## 3. The model does not capture under Dynamo — and this is the real finding

Compile time and graph-break *counts* could not be produced, because the model
does not form a graph to begin with. Every capture path tried —
`CompileCounter`, `backend="aot_eager"`, `backend="cudagraphs"`, and
`fullgraph=True` — fails at the **same** construct, none of which involve
Triton or Inductor:

```
fullgraph=True -> Unsupported: there ARE graph breaks
  first break at rfdetr/models/lwdetr.py:505 (torch.stack on a tuple)
aot_eager / cudagraphs / counter ->
  TypeError: expected Tensor as element 0 in argument 0, but got torch.Size
  from rfdetr/models/lwdetr.py:476  samples = nested_tensor_from_tensor_list(samples)
  -> rfdetr/utilities/tensors.py:151  m[: img.shape[1], : img.shape[2]] = False
```

Two host-side constructs in the forward defeat Dynamo:

1. **NestedTensor construction from Python shape ints.** `LWDETR.forward` wraps
   its input with `nested_tensor_from_tensor_list`, which allocates a padded
   tensor and builds a boolean mask by *Python-integer* slice assignment
   (`m[:img.shape[1], :img.shape[2]] = False`). Dynamo cannot trace a slice
   driven by `.shape` ints used as Python values — it sees a `torch.Size` where
   it expects a tensor and raises.
2. **`torch.stack` on a tuple** at `lwdetr.py:505`.

`fullgraph=True` failing is the direct statement that **RF-DETR-Small has graph
breaks**; the capture errors say the breaks are not benign fall-back points but
constructs that abort tracing. Even with Triton present, this model would not
compile into one graph — it would shatter at the NestedTensor boundary on every
forward, which is precisely the case where `torch.compile` quietly stops
helping. The mask-building code runs once per call on the host regardless of
backend.

## 4. The contrast that explains where each tool belongs

The same `LWDETR.forward` that Dynamo cannot capture **exports to ONNX cleanly**
— `export.export_rfdetr` traces it through `torch.onnx.export` and produces the
1,570-node graph the rest of this repo's TensorRT work is built on. The
difference is the whole point:

- **`torch.onnx.export`** uses a tracing path that runs the Python once and
  *specializes on the concrete shapes it sees*. The host-side mask construction
  and the `.shape`-int slicing execute as plain Python during the trace and bake
  their result into constants. Export tolerates host-side Python because it is
  *leaving* Python — the artifact never runs that code again.
- **`torch.compile`** tries to preserve a runnable Python callable *with*
  dynamism, so it must trace those same constructs symbolically — and chokes on
  exactly the code export was happy to constant-fold.

So for this model: **export is the tool that fits, and it already works.**
`torch.compile` is for keeping a model fast *inside* a Python training/eval loop
on a host that has the framework and a working Inductor/Triton stack — neither
of which holds here (Windows, no Triton), and the model's own forward would
fragment capture even where they did. An artifact shipping to a device with no
PyTorch (the Lambda ONNX-Runtime path, or the TensorRT engines in
`TENSORRT-REPORT.md`) is the export case, not the compile case. The two are not
competing options for this model; they are answers to different questions, and
only one of them is answerable on the artifact this project actually ships.

---

## What was attempted and blocked

| attempt | outcome |
|---|---|
| `torch.compile(model)` (inductor, default) | blocked — no Triton on Windows |
| `torch.compile(model, mode="reduce-overhead")` | blocked — same |
| `max-autotune` | not attempted — needs Inductor |
| `backend="aot_eager"` | fails at NestedTensor construction (§3) |
| `backend="cudagraphs"` | fails at the same construct |
| `fullgraph=True` (graph-break probe) | fails — confirms graph breaks exist |
| graph-break *count* | not obtainable — capture aborts at the first break rather than breaking-and-continuing |
| eager baseline + noise floor | measured (§1) |

### Reproduce
Load `RFDETRSmall(pretrain_weights=<ckpt>).model.model.eval().cuda()`, feed one
preprocessed `[1,3,512,512]` fixture, and compare `model(x)` under
`torch.no_grad()` against `torch.compile(model, ...)` for the backends above.
Scripts kept out of the repo; they import `export.parity.preprocess_detr`.
