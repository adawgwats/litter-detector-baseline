# Registering TensorRT engines: what fits the registry, and what doesn't

Measured 2026-09-08 against `export/registry.py` at HEAD, using the nine
TensorRT engine provenance records under `reports/trt-provenance/` and the two
engine-vs-ONNX parity reports from `reports/trt-parity-*.json`. Additive: this
report adds **no** records to `registry/`, because the central finding is that
the engine records **cannot** be added without either colliding or
misrepresenting the artifact. Every result below is a captured outcome from a
real registration attempt, not a reading of the code.

> **Update (see end):** all four findings below were fixed upstream after this
> was written, in two rounds. Both fp32 TensorRT engines (TF32 on and off) now
> coexist with the fp32 ONNX and each other, and all three are registered in
> `registry/`. The body below is the original pre-fix analysis; the closing
> "Update" section records what changed across both rounds.

The registry was built to key identity on the full provenance tuple —
including the target — rather than on the version string. That is the right
model. These findings are about the distance between that stated model and the
code that enforces it, and they surface exactly when a **second target** for a
model that already has a record arrives. Which is the shape of the task the
registry exists for: one source, many targets.

---

## What fits

An engine **constructs as a `ReleaseRecord` without modification.** `Target`,
`Conversion`, and `ParityRef` are expressive enough to describe a TensorRT
target from an ONNX source:

- `Target(runtime="tensorrt", runtime_version="11.2.1.2", execution_provider="cuda", host_class=…)`
- `Conversion(exporter="export.trt_build.build_engine", mechanism="tensorrt-11.2.1.2-build (onnx->plan, TF32=False)", opset=17, source_sha256=<onnx sha>)` — `is_conversion` is true, and the source ONNX sha is recorded, so the conversion is provenance-linked to its input.
- The engine-vs-ONNX parity report satisfies the required `ParityRef`: it names the plan's sha256 on the candidate side, was run at the recommended 0.4 threshold, and compared against the exact source ONNX the conversion names. All three `__post_init__` parity gates pass.

The constructed record gets a distinct `record_id` (`299e3719ff3d` for the
TF32-off engine) from the fp32 ONNX record already in the registry
(`31486ca5ebe0`). **`identity()` correctly considers them different releases.**

## What doesn't fit — four findings

### 1. The uniqueness namespace excludes the target, so a second target collides

`identity()` includes the target; `Registry.register`'s collision check does
not. It refuses any second artifact under an existing `(model_name, version)`:

```
register engine into repo registry ->
  VersionCollisionError: rfdetr-s-litter v2.0.1-fp32 is already registered as
  artifact a59fc4174a77… (record 31486ca5ebe0); this artifact is 9eb7a83f26fc…
  One version string cannot name two sets of bytes — rev the version
```

So the fp32 ONNX (`a59fc417`, target `onnxruntime-node`/`cpu`/`lambda-arm64`)
and its own fp32 TensorRT engine (`9eb7a83f`, target
`tensorrt`/`cuda`/`rtx-4070`) **cannot coexist**, though they are the same
model, version, and precision differing only in target — and though
`identity()` already rates them distinct. The registry's own identity model
says "two releases"; its uniqueness rule says "collision". That contradiction
is the finding.

This is not specific to ONNX-vs-engine. Two **engines** from one source
collide the same way:

```
register TF32-off engine (9eb7a83f)  -> OK
register TF32-on  engine (3b1e7817)  ->
  VersionCollisionError: rfdetr-s-litter v2.0.1-fp32 is already registered as
  artifact 9eb7a83f26fc…; this artifact is 3b1e7817f7bd…
```

These two engines are the multi-target case in miniature: same ONNX, different
builder config (TF32), **measurably different numerics** (max `labels` delta
9.32 vs 2.71, 2 operating-point flips vs 0 — see `TENSORRT-REPORT.md §5`). They
are exactly what a multi-target registry must keep distinct, and the version
namespace cannot hold both.

The six determinism builds (identical recipe, six different plan hashes) are
the same collision with N=6. They are also individually un-registerable as a
group for a second reason: each build has different bytes and so needs its own
parity run — a single parity report names one sha and the parity gate correctly
rejects it for the other five. That gate is working as designed; the point is
that even with six parity runs, the six would collide on `(model, version)` as
finding 1 shows for N=2.

**Root cause and minimal fix.** The collision key is `(model_name, version)`.
It should be `(model_name, version, target-digest)` — the target is already in
`identity()`, so the fix is to make the uniqueness check agree with the
identity model the module docstring already commits to. The `(model, version)
→ one artifact` guarantee is still valuable and should hold **within a single
target** (that is the S3-key-overwrite defect it was built to catch); it just
should not forbid a second target.

### 2. The version-string vocabulary cannot express the target axis

`VERSION_PATTERN` is `^v\d+\.\d+\.\d+-(fp32|fp16|int8)$`. The only axes a
version can encode are the semver triple and the precision. There is no room
for runtime, execution provider, host class, or a build-config flag like TF32.
So even a human trying to disambiguate two targets by hand cannot do it in the
version string without either:

- **revving the semver** (`v2.0.2-fp32`) — which the module explicitly warns
  against, since it lies that the model changed when only the target did; or
- **forking the model name** (`rfdetr-s-litter-trt-tf32off`) — which does let
  both register (verified), but severs the registry's knowledge that they are
  the same model. Both workarounds defeat the thing the registry is for.

### 3. `precisionEvidence` cannot be derived from a plan, so the "checked not declared" guarantee silently doesn't apply

The registry's headline property is that precision is inferred from the graph's
initializer element types and a declaration contradicting the bytes is refused.
A serialized TensorRT plan has no ONNX initializers. The CLI path calls
`read_onnx_facts`, which fails outright on a plan:

```
read_onnx_facts(fp32_notf32.plan) ->
  DecodeError: Error parsing message with type 'onnx.ModelProto': Wire format was corrupt
```

Via the API, `precision_evidence=None` is accepted (it means "never inspected"),
so an engine record can be built — but its precision claim is then back to being
an unchecked assertion, the exact condition the registry was built to make loud.
For an opaque artifact this should be **explicit**, not silent: a record for a
non-ONNX artifact should carry a marker like `precisionEvidence: {"opaque":
true, "reason": "serialized TensorRT plan; precision not recoverable from
bytes"}` so a reader sees that the guarantee was waived, rather than seeing a
bare `null` that also means "ONNX we forgot to inspect".

### 4. `Target` has no structured field for the toolchain that changes the numerics

The peer's own framing — and the parity data — say a target's identity includes
GPU compute capability and the CUDA and driver versions: the same plan bytes
under a different driver can produce different numbers, so it is a different
release. `Target` has only `runtime`, `runtime_version`, `execution_provider`,
`host_class`. Those four were shaped for the Lambda ONNX-Runtime target and have
no home for compute capability, CUDA runtime, or driver. They currently have to
be packed into `host_class` as a compound string
(`"NVIDIA GeForce RTX 4070|sm8.9|cuda13.1|drv591.86|win"`), which means they are
in the identity digest by accident of concatenation rather than as first-class,
individually-queryable fields. For a GPU target they are load-bearing and
should be structured.

---

## Recommendation

The registry's data model is close, and the fix is small and local:

1. **Add the target to the uniqueness key** (`(model, version, target-digest)`),
   so identity and uniqueness agree and a second target stops colliding. This is
   the one change that unblocks "one source, many targets".
2. **Make opaque-artifact precision explicit** rather than a silent `null`, so
   the "checked not declared" guarantee visibly degrades for non-ONNX targets
   instead of quietly not applying.
3. **Give `Target` structured fields** for compute capability, CUDA, and driver
   (optional, populated for GPU targets), so the toolchain that changes the
   numerics is part of identity by design rather than by string-packing.

Until (1) lands, no engine records were added to `registry/`: doing so would
require faking a semver bump or forking the model name, and the registry is
better left honest — holding three ONNX records and this report saying why the
engines can't join them yet — than padded with records that misrepresent what
they are. The nine provenance records and two parity reports are the ready
corpus for the fix; the throwaway registration runs that produced every quoted
error are in the reproduce note below.

### Reproduce

The exploration script constructs engine records from `reports/trt-provenance/`
and `reports/trt-parity-*.json`, attempts registration into the repo registry
and into throwaway temp registries, and prints each captured outcome. It writes
nothing under `registry/`. (Script kept out of the repo; it imports
`export.registry`'s public API — `ReleaseRecord`, `Target`, `Conversion`,
`ParityRef`, `Registry`, `class_list_sha256` — and builds the `ParityRef` by
hand rather than via `parity_ref_from_report`, because that helper assumes both
sides ran under ONNX Runtime and reads a `byThreshold[]` block the TensorRT
parity report does not carry — itself a small instance of finding 1: the
transcription path is ONNX-Runtime-shaped.)

---

## Update — after the fix (registry keyed on `(model, version, target-digest)`)

All four findings were addressed upstream. Re-verified on this box:

- **Finding 1, cross-runtime: fixed.** The fp32 ONNX (`onnxruntime-node` target)
  and its fp32 TensorRT engine (`tensorrt`/`cuda` target) now **coexist** — the
  engine registered as `357df9832c8d` alongside the ONNX's `31486ca5ebe0` under
  the same `v2.0.1-fp32`. The registry now holds one source across two targets,
  which was the goal. That record is in `registry/`.
- **Findings 3 & 4: fixed and used.** The engine record carries
  `PrecisionEvidence.opaque("tensorrt plan: weights are not inspectable")` and
  structured `Target.compute_capability`/`cuda_version`/`driver_version` rather
  than a string-packed `host_class`.

**One residual, and it is a real one.** The uniqueness key is now
`(model, version, target-digest)`. The TF32-on and TF32-off engines have the
**same** `target.digest` (`03f57c99…`) — same GPU, CUDA, driver, runtime, EP —
because TF32 is a **builder-config** decision recorded in `Conversion`, not a
target property. So they still collide:

```
register TF32-off (9eb7a83f) -> OK
register TF32-on  (3b1e7817) ->
  VersionCollisionError: rfdetr-s-litter v2.0.1-fp32 on target
  tensorrt/cuda@11.2.1.2 is already registered as artifact 9eb7a83f26fc…;
  this artifact is 3b1e7817f7bd…
```

Two builds from one source **for one target**, with measurably different
numerics (max `labels` delta 9.32 vs 2.71; 2 operating-point flips vs 0), cannot
both be registered. The fix solved one-source-many-**targets**; it does not
cover one-source-one-target-many-**builds**, which the TF32 pair is. Whether
that case should be expressible is a design call: either TF32 (and peers like
it) is modelled as part of the target, or the uniqueness key also admits a
build-config digest, or the position is taken that two builds for one target are
not two releases and the registry is right to demand a version rev. Only the
TF32-off engine was registered; the TF32-on engine is left out, with this as the
reason rather than a forced semver bump.

### Second round — the residual was fixed too

The `(model, version, target-digest)` key left one gap: TF32-on and TF32-off
share a target digest (TF32 is builder config, not a destination), so two
*builds* for one target still collided. That was fixed by keying uniqueness on
the **release key** = the full identity tuple minus the artifact hash — the
recipe *and* the destination, without the bytes. TF32-on and TF32-off differ in
`Conversion.mechanism`, so they get different release keys and now coexist. Both
are registered:

```
registry/rfdetr-s-litter.v2.0.1-fp32.357df9832c8d.json   TF32-off engine
registry/rfdetr-s-litter.v2.0.1-fp32.644e1873370d.json   TF32-on  engine
registry/rfdetr-s-litter.v2.0.1-fp32.31486ca5ebe0.json   fp32 ONNX (onnxruntime target)
```

One source checkpoint is now represented in the registry as an ONNX target and
two distinct TensorRT builds, kept straight — the one-source-many-outputs shape,
demonstrated rather than asserted.

**And the six determinism builds still collide — now by design.** Same recipe,
same target, different bytes share a release key, so registering a second is
refused. That is the correct behaviour: it is the registry surfacing the
non-reproducibility that `TENSORRT-REPORT.md §3` measured, rather than storing
six rows that each claim to be the same build. Recording all six would need a
build identifier the recipe does not currently carry — a deliberate design
change, not a workaround. The refusal is the finding, so the six were not
registered.

### Named open item — verify conflates "not retained" with "gone"

`python -m export.registry verify` reports `MISSING artifact dist/trt/<plan>` for
both engine records, because plan binaries are ~119 MB and deliberately not
committed (`dist/` is gitignored). The records' `parity` evidence verifies `ok`.
The gap is real: verify cannot distinguish **"this artifact is deliberately not
stored in-tree"** from **"this artifact should be here and is gone."** For a
119 MB plan the first is the normal case, not an error, so reporting it as
`MISSING` is misleading.

**The fix, deliberately not made here.** A record should be able to declare that
its artifact is *not retained* (an explicit retention field), and `verify`
should report that as a distinct status — "not retained, as declared" — rather
than as a failure. That is a record-schema change, and making it now would
invalidate the two engine records already written against the current schema for
no gain today, with nothing forcing the decision. So it is left as a stated open
item with its fix named, not a rushed schema edit: the honest position is "the
registry cannot yet distinguish an artifact we deliberately do not retain from
one that is gone, and here is the field that would fix it."

### Postscript — a content address computed over a local rendering isn't one

The two engine records above shipped with a real defect, and the registry caught
it on its own evidence. Each pinned `reportSha256` over the bytes of its parity
report *as rendered in the working tree it was registered from* — Windows, CRLF
line endings — while git stored the blob as LF. `export.registry verify` re-hashes
the report on disk, so on an LF checkout the pin disagreed with the file and every
engine record read as `CHANGED`.

It was fixed in two commits, and the two-step is the point:

1. **Re-pin to the LF blob hash** (`08cd747`) fixed the *write* side — the pin now
   names the committed bytes.
2. **A line-ending policy** (`.gitattributes`, `* text=auto eol=lf`, `f70f9f8`)
   fixed the *read* side — `verify` now hashes LF bytes on every platform, because
   the working tree matches the blob regardless of `core.autocrlf`.

Re-pinning alone was not enough: between those two commits, `verify` on a Windows
checkout still reported `CHANGED` on records that were correct, because the
*reader* was platform-dependent even after the *pin* was fixed. **Both sides were
hashing a local rendering of the file rather than the file.** A content address is
only an address if the bytes it is taken over are the same everywhere; line
endings are part of those bytes, so they cannot be left to the platform. Every
committed blob was already LF, so the policy rewrote nothing and invalidated no
pin — it only makes future checkouts and future registrations deterministic. Two
CI steps now guard it: one checks that every `reportSha256` matches the report it
names (the identity check never looked at the pins, which is what let these two
sit), and one checks that the line-ending policy is still in force — because a
POSIX-only `verify` in CI passes CRLF pins happily, so it is necessary but not
sufficient. (An existing CRLF working tree needs a one-time `git add
--renormalize .`, or a delete-and-re-checkout of the tracked files — a plain
`git checkout -- .` skips files git considers unchanged under autocrlf.)
