# CTMS / RVI MULTI-MATRIX TRANSFORMER
# COMPLETE COMMAND-LINE OPERATOR'S MANUAL
## v0.4.0

This manual is the operational layer for the canonical CTMS kernel and the
RVI Multi-Matrix Transformer research package.

The command-line executable is:

```bash
ctms
```

An alias is also installed:

```bash
rvi
```

Both invoke the same program.

---

# 0. OPERATIONAL DOCTRINE

The project has four distinct layers. Do not collapse them.

```text
CANONICAL CTMS KERNEL
    kernel/CTMS.md

EXECUTABLE RESEARCH MODEL
    rvi_multimatrix/

TRAINING / INDEX DATA
    corpus/, indices/, checkpoints/

CLI / OPERATOR SURFACE
    ctms
```

The canonical CTMS file remains domain-agnostic. RVI, multi-matrix meaning,
transformer attention, corpus annotations, DAC rendering, and benchmarks are
applications around the kernel.

The executable CLI emits observable state, diagnostics, hashes, metrics,
matrix routes, operator probabilities, relational-field statistics, and
explicit files. It does not expose private chain-of-thought.

The top-level scientific rule is:

> A model result is not an architectural result until the relevant component
> survives ablation, matched controls, held-out evaluation, and reproducibility.

---

# 1. INSTALLATION

From the package root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify:

```bash
ctms --version
ctms doctor --smoke
```

Without installation:

```bash
python -m rvi_multimatrix --version
python -m rvi_multimatrix doctor --smoke
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
ctms doctor --smoke
```

---

# 2. FIRST RUN

Create a workspace:

```bash
ctms init ~/rvi-lab
cd ~/rvi-lab
```

The workspace layout is:

```text
corpus/
  raw/
  annotated/
runs/
checkpoints/
indices/
exports/
traces/
audio/
configs/
reports/
workspace.json
```

Validate the kernel and config:

```bash
ctms kernel validate
ctms config validate --config configs/config.json
```

Inspect model size:

```bash
ctms model info --config configs/config.json
```

---

# 3. GLOBAL CONVENTIONS

## 3.1 Exit codes

```text
0   success
2   command-line usage error
3   data/config/file error
4   model/checkpoint incompatibility
5   runtime/device failure
6   reserved safety boundary
130 keyboard interrupt
```

Shell use:

```bash
ctms corpus validate corpus/train.jsonl || exit $?
```

## 3.2 Debug mode

Normal CLI errors are compact.

For full Python tracebacks:

```bash
CTMS_DEBUG=1 ctms ...
```

## 3.3 Device selection

Most model commands accept:

```bash
--device auto
--device cpu
--device cuda
--device mps
```

`auto` resolves in this order:

```text
CUDA → Apple MPS → CPU
```

For deterministic research, record the actual resolved device.

## 3.4 Seeds

Whenever a command exposes `--seed`, record it in the run manifest.

Recommended default:

```text
1337
```

The number is not special. Reproducibility is.

---

# 4. KERNEL COMMANDS

The kernel commands operate on `kernel/CTMS.md`.

## 4.1 Print the canonical kernel

```bash
ctms kernel show
```

Alternate path:

```bash
ctms kernel show --path /path/to/CTMS.md
```

## 4.2 Hash the kernel

```bash
ctms kernel hash
```

Use the hash in every serious training run, paper, benchmark, and artifact
manifest. It lets another operator establish which canonical file governed a
run.

## 4.3 Validate the kernel

```bash
ctms kernel validate
```

JSON:

```bash
ctms kernel validate --json
```

Validation checks the presence of the core traversal terms and operator glyphs.

The validator also warns if the historical CTMS source requests chain-of-thought
logging. The executable system intentionally replaces that with structured,
observable operator/state traces.

## 4.4 Operator dictionary

```bash
ctms kernel operators
```

JSON:

```bash
ctms kernel operators --json
```

Core operators:

```text
♢  NEXT
◇  PAUSE
◆  RESUME
↺  ABANDON / RESET
⇝  REDIRECT
⊢  AXIOM
⊨  VALIDATED
⊍  EXPLORE
⊝  DISCARD
⋈  JOIN
⋔  SPLIT
↑  TRANSCEND
⍟  METAMORPHOSE
∞  RECURSE
§  SELF-REFERENCE CHECK
⊥  INCOMPLETE
⊤  COMPLETE
```

## 4.5 Read this manual

```bash
ctms kernel manual
ctms manual show
ctms manual path
```

Search:

```bash
ctms manual search "matrix collapse"
ctms manual search "checkpoint" --context 6
```

---

# 5. CONFIGURATION

Default configuration:

```bash
ctms config show
```

Validate:

```bash
ctms config validate
```

Specify another file:

```bash
ctms config validate --config configs/large.json
```

Create a fresh default:

```bash
ctms config init --out configs/test.json
```

Overwrite:

```bash
ctms config init --out configs/test.json --force
```

Diff two configurations:

```bash
ctms config diff configs/a.json configs/b.json
```

## 5.1 Base transformer parameters

```text
vocab_size
d_model
n_heads
n_layers
d_ff
max_seq_len
dropout
```

## 5.2 RVI parameters

```text
rvi_enabled
rvi_rank
rvi_alpha
rvi_beta
```

Interpretation:

```text
rvi_rank
    number of phase/amplitude interference coordinates

rvi_alpha
    contribution of the refined RVI field to attention logits

rvi_beta
    persistent relational-state contribution across depth
```

## 5.3 Multi-matrix parameters

```text
multimatrix_enabled
matrix_names
matrix_dim
matrix_top_k
matrix_residual_alpha
```

`matrix_top_k` controls sparse matrix routing per token.

Start conservatively:

```text
M = 8–12 matrices
top_k = 2–4
```

Do not set `top_k=M` merely because the model can. A matrix system where every
matrix activates equally has ceased to be a routing system.

## 5.4 CTMS differentiable parameters

```text
ctms_enabled
ctms_recurse
operator_names
```

The trainable CTMS analogue is not the canonical CTMS kernel.

`ctms_recurse` is the number of relation-of-relations refinement passes.

## 5.5 Memory parameters

```text
memory_decay
```

Higher values preserve more prior cross-chunk matrix summary.

## 5.6 Auxiliary losses

```text
matrix_loss_weight
operator_loss_weight
transition_loss_weight
diversity_loss_weight
```

These must be ablated independently.

---

# 6. CORPUS FORMAT

The recommended corpus is JSONL.

One record per line:

```json
{
  "id": "TD-001:000042",
  "text": "passage text",
  "matrices": [
    "semantic",
    "temporal",
    "contradiction"
  ],
  "operator": "REDCHECK",
  "metadata": {
    "document_id": "TD-001",
    "title": "source title",
    "page": 42,
    "source_hash": "sha256...",
    "license": "research-use",
    "notes": ""
  }
}
```

## 6.1 Why annotations are external

Do not rewrite the source document to force it into the matrix system.

The source remains intact.

The annotation layer gives the same source span multiple addresses.

```text
SOURCE PASSAGE
   ├─ semantic address
   ├─ temporal address
   ├─ spatial address
   ├─ contradiction address
   ├─ operator address
   └─ provenance address
```

## 6.2 Validate corpus

```bash
ctms corpus validate corpus/annotated/train.jsonl
```

JSON report:

```bash
ctms corpus validate corpus/annotated/train.jsonl --json
```

Checks include:

```text
unique record id
non-empty text
known matrix names
known operator names
metadata shape
document_id warning
```

## 6.3 Corpus statistics

```bash
ctms corpus stats corpus/annotated/train.jsonl
```

Reports:

```text
record count
document count
byte count
mean / median / min / max record size
matrix usage
operator usage
```

Use this before training. Matrix and operator imbalance can otherwise masquerade
as architectural behavior.

## 6.4 Corpus split

```bash
ctms corpus split corpus/all.jsonl --out corpus/splits
```

Default:

```text
80% train
10% validation
10% test
seed 1337
```

Alternate:

```bash
ctms corpus split corpus/all.jsonl \
  --train 0.85 \
  --val 0.10 \
  --seed 2026 \
  --out corpus/splits
```

For publication-quality evaluation, prefer split by source document or author,
not merely random passage split, when leakage is possible.

## 6.5 Hash corpus artifacts

```bash
ctms corpus hash corpus/splits/train.jsonl
```

Record hashes in run manifests.

---

# 7. MATRIX DESIGN

Default matrices:

```text
semantic
temporal
spatial
causal
embodied
media
lineage
contradiction
operator
provenance
```

These are defaults, not ontology law.

## 7.1 Semantic matrix

Question:

```text
What does this passage assert or signify?
```

## 7.2 Temporal matrix

Question:

```text
What changes through time?
What preceded what?
What is state versus transition?
```

## 7.3 Spatial matrix

Question:

```text
Where is the object/event?
What changes because the observer moves?
```

## 7.4 Causal matrix

Question:

```text
What is proposed as cause, mechanism, dependency, or intervention?
```

Do not collapse correlation into this matrix without explicit labeling.

## 7.5 Embodied matrix

Question:

```text
Which body, sensorium, organism, gesture, posture, or physiological state matters?
```

## 7.6 Media matrix

Question:

```text
Which carrier mediates the relation?
book / wall / video / audio / packet / PCM / photograph / model output
```

## 7.7 Lineage matrix

Question:

```text
Which source, predecessor, transformation, quotation, adaptation, or inheritance
connects this object to others?
```

## 7.8 Contradiction matrix

Question:

```text
What materially conflicts with this?
What would falsify it?
What alternative reading would change the result?
```

This is not "negative sentiment."

## 7.9 Operator matrix

Question:

```text
What transformation should happen next?
```

## 7.10 Provenance matrix

Question:

```text
Where did this statement come from?
Can the source be reconstructed?
What is measured, inferred, quoted, simulated, remembered, or generated?
```

---

# 8. MODEL INSPECTION

## 8.1 Parameter count and architecture

```bash
ctms model info
```

Alternate config:

```bash
ctms model info --config configs/large.json
```

## 8.2 Full backward smoke test

```bash
ctms model smoke
```

GPU:

```bash
ctms model smoke --device cuda --batch-size 2 --seq-len 64
```

Run this after:

```text
changing tensor shapes
changing matrix count
changing rank
changing routing
changing CTMS recursion
changing PyTorch versions
```

---

# 9. TRAINING

## 9.1 Plain text training

Useful for language-model chassis testing:

```bash
ctms train \
  --text corpus/raw/corpus.txt \
  --steps 2000 \
  --seq-len 256 \
  --batch-size 8 \
  --out checkpoints/plain.pt
```

Plain text does not provide explicit matrix/operator supervision.

## 9.2 Annotated training

```bash
ctms train \
  --jsonl corpus/splits/train.jsonl \
  --steps 5000 \
  --seq-len 256 \
  --batch-size 8 \
  --out checkpoints/multimatrix.pt
```

## 9.3 Reproducible run

```bash
ctms train \
  --jsonl corpus/splits/train.jsonl \
  --config configs/config.json \
  --seed 1337 \
  --steps 5000 \
  --batch-size 8 \
  --seq-len 256 \
  --lr 3e-4 \
  --run-dir runs/2026-08-18-a \
  --out checkpoints/2026-08-18-a.pt
```

Record additionally:

```text
git commit
kernel hash
corpus hashes
Python version
PyTorch version
CUDA version
GPU
config file
command line
wall-clock time
```

## 9.4 Gradient clipping

Default:

```text
1.0
```

Change only with logged rationale.

## 9.5 Periodic checkpointing

```bash
ctms train \
  --jsonl corpus/train.jsonl \
  --steps 10000 \
  --save-every 1000 \
  --run-dir runs/long-run
```

## 9.6 torch.compile

```bash
ctms train --jsonl corpus/train.jsonl --compile
```

Validate numerical behavior before relying on compiled training for comparison.

---

# 10. TRAINING METRICS

Every logged step can include:

```text
loss
grad_norm
loss_lm
loss_matrix
loss_operator
loss_transition
loss_diversity
matrix_route_entropy
operator_entropy
rvi_field_mean
rvi_field_std
matrix_coupling_entropy
memory_norm
elapsed_s
```

## 10.1 Matrix-route entropy

High entropy:

```text
router is diffuse
```

Low entropy:

```text
router is highly selective
```

Neither is inherently good.

Watch for:

```text
always selecting the same matrices
matrix routing unrelated to annotation
route collapse after a few hundred steps
```

## 10.2 Matrix coupling entropy

Tracks whether matrix-to-matrix coupling becomes concentrated or remains diffuse.

## 10.3 Field standard deviation

If `rvi_field_std → 0`, the interference field may have collapsed.

If it grows without bound, normalization or optimization may be failing.

## 10.4 Diversity loss

Prevents all matrix representations from becoming identical.

It is not proof that the matrices have learned their intended semantics.

---

# 11. GENERATION

```bash
ctms generate \
  --checkpoint checkpoints/model.pt \
  --prompt "The relation changes when" \
  --tokens 300
```

Temperature:

```bash
--temperature 0.8
```

Top-k:

```bash
--top-k 50
```

For evaluation, do not rely on generation aesthetics alone.

---

# 12. CHECKPOINT OPERATIONS

## 12.1 Inspect

```bash
ctms checkpoint info checkpoints/model.pt
```

Reports:

```text
checkpoint hash
configuration
tensor count
parameter count
```

## 12.2 Compare

```bash
ctms checkpoint diff a.pt b.pt
```

Only changed tensors:

```bash
ctms checkpoint diff a.pt b.pt --only-changed
```

Reports per tensor:

```text
shape mismatch
L2 parameter delta
maximum absolute delta
```

---

# 13. MATRIX ROUTING INSPECTION

## 13.1 List matrices

```bash
ctms matrix list
```

## 13.2 Inspect routing for text

```bash
ctms matrix route \
  --checkpoint checkpoints/model.pt \
  --text "Vague ideas must be confronted with clear images."
```

Output is a probability distribution over configured meaning matrices.

This command is one of the main ways to detect whether the multi-matrix system is
actually differentiating reading frames.

## 13.3 Inspect matrix-to-matrix coupling

```bash
ctms matrix coupling \
  --checkpoint checkpoints/model.pt \
  --layer 0
```

One head:

```bash
ctms matrix coupling \
  --checkpoint checkpoints/model.pt \
  --layer 0 \
  --head 2
```

Interpret carefully. Coupling magnitude is a learned model parameter, not a
semantic proof.

---

# 14. CTMS OPERATOR ROUTING

List trainable operator labels:

```bash
ctms operator list
```

Predict operator distribution:

```bash
ctms operator predict \
  --checkpoint checkpoints/model.pt \
  --text "The evidence supports the claim, but an alternative explanation remains."
```

Possible outputs include:

```text
SEED
BLOOM
FORK
RECURSE
REDCHECK
FLATTEN
TRANSMIT
```

The operator router is a differentiable control surface.

It does not replace the canonical kernel.

---

# 15. RELATIONAL MEMORY

Encode text into compressed multi-matrix memory:

```bash
ctms memory encode \
  --checkpoint checkpoints/model.pt \
  --text "A changing relation can carry information unavailable to either endpoint."
```

Save:

```bash
ctms memory encode \
  --checkpoint checkpoints/model.pt \
  --text "..." \
  --out exports/memory.pt
```

The memory tensor is:

```text
[B, M, Dm]
```

not a full historical token-token field.

Use it for:

```text
cross-chunk experiments
document-state carry
persistent reading-frame experiments
retrieval/index embeddings
memory ablations
```

---

# 16. OBSERVABLE TRACE

Inspect per-layer model state:

```bash
ctms trace inspect \
  --checkpoint checkpoints/model.pt \
  --text "The object changes when its carrier changes."
```

Save:

```bash
ctms trace inspect \
  --checkpoint checkpoints/model.pt \
  --text "..." \
  --out traces/example.json
```

The trace can include:

```text
field statistics
memory statistics
matrix routing entropy
operator entropy
RVI field statistics
auxiliary losses when present
```

The trace is explicitly an observable model-state log, not private chain-of-thought.

---

# 17. MULTI-MATRIX MEANING INDEX

## 17.1 Build

```bash
ctms index build \
  --checkpoint checkpoints/model.pt \
  --jsonl corpus/splits/train.jsonl \
  --out indices/train.pt
```

## 17.2 Inspect

```bash
ctms index info indices/train.pt
```

## 17.3 Query all matrices

```bash
ctms index query \
  --checkpoint checkpoints/model.pt \
  --index indices/train.pt \
  --text "movement through the city changes what becomes perceptible"
```

## 17.4 Weighted query

```bash
ctms index query \
  --checkpoint checkpoints/model.pt \
  --index indices/train.pt \
  --text "..." \
  --weights "spatial=2,temporal=1,semantic=.5"
```

## 17.5 Contradiction retrieval

```bash
ctms index query \
  --checkpoint checkpoints/model.pt \
  --index indices/train.pt \
  --text "claim to test" \
  --weights "contradiction=2,provenance=1" \
  --contradiction-flip
```

This is designed to retrieve useful opposition rather than only nearest semantic
neighbors.

Always inspect retrieved provenance before treating a hit as evidence.

---

# 18. DAC / PCM

Render model hidden-state frames into a software audio waveform:

```bash
ctms dac render \
  --checkpoint checkpoints/model.pt \
  --text "The relation changes." \
  --out audio/relation.wav
```

Options:

```text
--sample-rate 48000
--hop-length 240
--harmonics 32
```

The DAC is a research sonification/sensory renderer.

It is not a direct neural-stimulation driver.

Do not connect ordinary audio outputs to electrodes or neural stimulation
hardware.

---

# 19. ABLATION

Run structural profiles:

```bash
ctms ablate \
  --text corpus/raw/evaluation.txt \
  --steps 300
```

Default profiles:

```text
baseline
rvi
matrix
full_no_ctms
full
```

Definitions:

```text
baseline
    no RVI, no multi-matrix, no CTMS analogue

rvi
    legacy single-field RVI + CTMS analogue

matrix
    multi-matrix projection/routing, no RVI

full_no_ctms
    multi-matrix + cross-matrix RVI, no CTMS refinement

full
    multi-matrix + cross-matrix RVI + CTMS refinement
```

Save:

```bash
ctms ablate \
  --text corpus/raw/evaluation.txt \
  --out reports/ablation.json
```

## 19.1 What ablation can establish

Ablation can show that a component changes behavior under a given experimental
design.

It does not by itself establish why.

## 19.2 Required future matched-compute suite

For serious comparison, add:

```text
matched parameter count
matched FLOPs
matched training tokens
matched wall-clock budget
matched memory budget
multiple seeds
confidence intervals
held-out document families
long-context retrieval
counterfactual retrieval
calibration
robustness under distribution shift
```

---

# 20. BENCHMARKING

Forward throughput:

```bash
ctms benchmark
```

GPU:

```bash
ctms benchmark \
  --device cuda \
  --batch-size 8 \
  --seq-len 512 \
  --iters 100
```

Reports:

```text
tokens/s
ms/iteration
parameter count
CUDA max allocated memory when available
```

Benchmark every architectural change.

---

# 21. REPRODUCIBILITY WORKFLOW

Before training:

```bash
ctms kernel hash > reports/kernel.sha256
ctms corpus hash corpus/splits/train.jsonl > reports/train.sha256
ctms config show --config configs/config.json > reports/config.resolved.json
ctms doctor --json > reports/environment.json
```

Train:

```bash
ctms train ...
```

After training:

```bash
ctms checkpoint info checkpoints/model.pt > reports/checkpoint.json
ctms ablate ... --out reports/ablation.json
ctms benchmark ... > reports/benchmark.json
```

Archive the exact command line.

---

# 22. DATA PROVENANCE

Every record should eventually carry:

```text
document_id
record_id
title
author / creator
date
page / location
source file hash
extraction method
annotation author/system
annotation date
license/copyright status
transformation history
confidence
notes
```

For scientific sources also consider:

```text
DOI
journal
version
retraction/correction status
study type
sample size
measurement modality
```

For archival evidence:

```text
custodian
acquisition route
file timestamp
chain of custody
record authority
```

For generated material:

```text
model
checkpoint hash
prompt/input hash
sampling parameters
generation timestamp
```

---

# 23. CORPUS GOVERNANCE

Do not silently mix:

```text
primary source
secondary source
fiction/literature
scientific measurement
scientific interpretation
personal report
model generation
simulation
unverified claim
```

Use provenance to make the difference recoverable.

A corpus can intentionally include contradiction.

It must not unintentionally erase epistemic type.

---

# 24. COPYRIGHT / LICENSING

Do not assume possession of a PDF grants the right to redistribute a derived
training corpus.

For each source track:

```text
public domain
licensed
permission obtained
research exception/fair-use analysis
unknown
restricted
```

The package does not bundle the user's training books.

It stores schemas and model code.

---

# 25. HIGH-RISK SOURCE MATERIAL

A corpus may contain historically or scientifically relevant material that is
unsafe to turn into operational instructions.

Examples include:

```text
clandestine drug manufacture
weapons construction
harmful biological procedures
criminal evasion techniques
```

For these sources, keep:

```text
historical context
rhetorical structure
bibliographic metadata
high-level conceptual relations
safety-relevant analysis
```

separate from actionable operational instruction.

The CLI is a transformer research tool, not an automation layer for harmful
procedures.

---

# 26. MATRIX-COLLAPSE DIAGNOSTICS

The central failure mode of the multi-matrix system is:

```text
all matrices learn the same thing
```

Symptoms:

```text
matrix-route distributions identical across inputs
matrix embeddings highly cosine-similar
coupling matrix becomes uniform
removing a matrix has no measurable effect
matrix labels are predictable from superficial document identity only
```

Tests:

```text
per-matrix cosine Gram matrix
route entropy by corpus class
matrix-label confusion matrix
matrix ablation
matrix permutation test
cross-author holdout
cross-document holdout
```

The existing diversity loss is only a first defense.

---

# 27. OPERATOR-COLLAPSE DIAGNOSTICS

Failure mode:

```text
operator router always selects RECURSE
```

or another dominant label.

Check:

```text
operator frequency in corpus
predicted operator entropy
operator confusion matrix
operator ablation
balanced sampling
class-weighted loss
```

Do not mistake a frequently selected operator for a universally useful operator.

---

# 28. RVI FAILURE MODES

## 28.1 Field collapse

```text
rvi_field_std ≈ 0
```

## 28.2 Field domination

RVI logits overwhelm semantic attention.

Reduce or schedule:

```text
rvi_alpha
```

## 28.3 Delay degeneracy

All learned phase lags become identical or extreme.

Inspect per head/rank.

## 28.4 Matrix coupling degeneracy

All heads learn the same coupling topology.

Compare heads.

## 28.5 Recursion instability

Repeated composition amplifies noise or produces near-uniform fields.

Ablate:

```text
ctms_recurse
REDCHECK gate
field normalization
```

---

# 29. TRANSITION OBJECTIVE FAILURE MODES

The transition loss exists to model becoming, not merely state.

Failure:

```text
transition head predicts trivial local smoothness
```

Tests:

```text
time-shuffled negative examples
longer-lag transition prediction
event-boundary prediction
state-vs-transition contrast
derivative/change-point labels
```

---

# 30. 4-D AND 5-D TOKEN-FIELD EXTENSION

Current multi-matrix latent:

```text
[B, M, T, Dm]
```

Conceptual 5-D token field:

```text
[B, T, M, K, D]
```

where:

```text
T = token/time
M = meaning matrix
K = CTMS branch / trajectory / perspective
D = latent state
```

A future sparse implementation should avoid dense:

```text
M × K × T × T
```

materialization.

Recommended design:

```text
top-k matrix router
top-k trajectory router
sparse branch objects
cross-branch RVI only for selected pairs
branch pruning with explicit provenance
flatten only at output
```

Future CLI commands should include:

```text
ctms branch list
ctms branch fork
ctms branch join
ctms branch prune
ctms branch compare
ctms trajectory inspect
ctms trajectory replay
```

These are documented as roadmap commands; they are not yet implemented in v0.4.

---

# 31. BCI / BIOFEEDBACK ROADMAP

Safe initial architecture:

```text
biosignal read
→ timestamp
→ encoder
→ multi-matrix state
→ RVI/CTMS
→ DAC / visual / haptic feedback
→ user
→ new measurement
```

Required future CLI surfaces:

```text
ctms bio ingest
ctms bio clocks
ctms bio calibrate
ctms bio label
ctms bio session start
ctms bio session stop
ctms bio export
ctms bio sham
ctms bio analyze
```

The first write path should remain sensory feedback.

Direct neural stimulation requires a separate medical/engineering safety system
and is outside this CLI.

---

# 32. CLOCK / TIMEBASE ROADMAP

Multi-device work requires one master timebase.

Future event record:

```json
{
  "t_master_ns": 0,
  "device": "eeg",
  "device_timestamp": 0,
  "offset_estimate_ns": 0,
  "uncertainty_ns": 0,
  "payload": {}
}
```

Required diagnostics:

```text
clock drift
buffer latency
audio output latency
sensor acquisition latency
user-event latency
dropped frames
jitter
```

---

# 33. DISTRIBUTED TRAINING ROADMAP

v0.4 is intentionally a compact research implementation.

Before scaling:

```text
DDP/FSDP
gradient accumulation
mixed precision
activation checkpointing
FlashAttention/SDPA integration
compiled kernels
sparse matrix routing
distributed corpus loader
streaming checkpoints
experiment tracker
```

Do not scale an unablated idea merely because more GPUs are available.

---

# 34. PERFORMANCE ROADMAP

Likely hotspots:

```text
token-token RVI field O(T²)
CTMS relation composition
multi-matrix projections
matrix mixing
Python corpus loader
autoregressive generation without KV cache
```

Priority engineering targets:

```text
PyTorch SDPA
KV cache
persistent RVI cache
block-sparse relation field
chunked matrix memory
top-k route fusion
Triton/CUDA kernels
quantized inference
```

---

# 35. TESTING PYRAMID

## Unit

```text
phase wrapping
matrix projection shapes
router normalization
coupling normalization
memory update
index search
DAC bounds
config validation
```

## Integration

```text
forward + backward
cross-chunk memory
annotated training
checkpoint save/load
index build/query
DAC WAV write
```

## Scientific

```text
baseline vs RVI
matrix vs no-matrix
CTMS vs no-CTMS
transition objective ablation
matrix-label shuffle
operator-label shuffle
contradiction retrieval
long-context retrieval
```

## Reproducibility

```text
multiple seeds
same checkpoint hash
same corpus hash
same kernel hash
same environment
```

---

# 36. SECURITY

Treat checkpoints, corpora, and indices as data-bearing artifacts.

Do not load untrusted PyTorch checkpoints casually. `torch.load` can be unsafe
with malicious serialized objects in some contexts.

Prefer artifacts you created or trust.

Future hardening:

```text
safetensors
signed manifests
read-only corpus mounts
hash verification
sandboxed preprocessing
dependency lockfile
SBOM
```

---

# 37. BACKUPS

Keep at least:

```text
canonical source documents
canonical CTMS.md
annotated corpus
config
checkpoint
run metrics
index
environment manifest
hash manifest
```

A generated sample is reproducible only if the checkpoint and sampling
parameters survive.

---

# 38. EXPERIMENT NAMING

Recommended:

```text
YYYYMMDD-HHMM-purpose-seed
```

Example:

```text
20260818-2305-matrix-collapse-1337
```

Avoid names such as:

```text
final
final2
final-real
final-real-new
```

The universe has suffered enough.

---

# 39. RUN MANIFEST

Every serious run should eventually write:

```json
{
  "run_id": "...",
  "kernel_sha256": "...",
  "config_sha256": "...",
  "train_sha256": "...",
  "val_sha256": "...",
  "checkpoint_sha256": "...",
  "git_commit": "...",
  "seed": 1337,
  "device": "...",
  "torch": "...",
  "python": "...",
  "command": "...",
  "started": "...",
  "finished": "..."
}
```

This is a planned automation target for the CLI.

---

# 40. RECOMMENDED DAILY WORKFLOW

```bash
ctms doctor
ctms corpus validate corpus/splits/train.jsonl
ctms corpus stats corpus/splits/train.jsonl
ctms config validate --config configs/current.json
ctms model smoke --config configs/current.json
ctms train ...
ctms checkpoint info checkpoints/current.pt
ctms matrix route ...
ctms operator predict ...
ctms trace inspect ...
ctms ablate ...
ctms benchmark ...
```

---

# 41. NEW CORPUS WORKFLOW

For each new canonical source:

```text
1. preserve source
2. hash source
3. assign document ID
4. segment without destroying provenance
5. annotate matrices
6. annotate operator(s)
7. mark epistemic/source class
8. validate JSONL
9. inspect corpus balance
10. split with leakage controls
11. train
12. ablate
13. inspect routes
14. index
15. redcheck retrieved relations
```

---

# 42. EVALUATION QUESTIONS

Ask the model:

```text
Does the matrix router change with the reading frame?
Does contradiction retrieval find actual counterevidence?
Does RVI help when two useful but incompatible readings are preserved?
Does CTMS recursion improve the result over equivalent extra depth?
Does transition prediction capture real change rather than token adjacency?
Does cross-chunk memory preserve useful relations without contaminating later text?
Do gains survive author/document holdout?
Do gains survive matched compute?
```

---

# 43. WHAT NOT TO CALL SUCCESS

Do not call these success:

```text
beautiful prose
interesting samples
one low validation loss
one seed
one author
one benchmark
high phase-lock diagnostic
a visually striking coupling matrix
a narrative that matches the theory
```

Success requires controlled comparison.

---

# 44. WHAT WOULD COUNT AS A STRONG RESULT

Examples:

```text
multi-matrix model improves contradiction retrieval on held-out authors
while matched for compute;

RVI improves long-range relational retrieval but not local language loss;

CTMS refinement improves calibration under counterfactual prompts;

transition loss improves event-boundary prediction without harming language loss;

specific matrices survive permutation/ablation tests and specialize reproducibly
across multiple seeds.
```

---

# 45. CLI COMMAND MAP

Implemented in v0.4:

```text
ctms init
ctms doctor

ctms kernel show
ctms kernel hash
ctms kernel validate
ctms kernel operators
ctms kernel manual

ctms config show
ctms config validate
ctms config init
ctms config diff

ctms corpus validate
ctms corpus stats
ctms corpus split
ctms corpus hash

ctms model info
ctms model smoke

ctms checkpoint info
ctms checkpoint diff

ctms train
ctms generate

ctms matrix list
ctms matrix route
ctms matrix coupling

ctms operator list
ctms operator predict

ctms memory encode
ctms trace inspect

ctms index build
ctms index query
ctms index info

ctms dac render

ctms ablate
ctms benchmark

ctms manual show
ctms manual path
ctms manual search
```

Roadmap:

```text
ctms branch ...
ctms trajectory ...
ctms bio ...
ctms clock ...
ctms serve ...
ctms export ...
ctms quantize ...
ctms profile ...
ctms distributed ...
ctms manifest ...
ctms compare-runs ...
ctms report ...
```

---

# 46. QUICK REFERENCE

Install:

```bash
pip install -e .
```

Check:

```bash
ctms doctor --smoke
```

Initialize:

```bash
ctms init ./lab
```

Validate corpus:

```bash
ctms corpus validate corpus/train.jsonl
```

Train:

```bash
ctms train --jsonl corpus/train.jsonl --out checkpoints/model.pt
```

Generate:

```bash
ctms generate --checkpoint checkpoints/model.pt --prompt "The system"
```

Inspect routing:

```bash
ctms matrix route --checkpoint checkpoints/model.pt --text "..."
```

Inspect operator:

```bash
ctms operator predict --checkpoint checkpoints/model.pt --text "..."
```

Trace:

```bash
ctms trace inspect --checkpoint checkpoints/model.pt --text "..."
```

Index:

```bash
ctms index build --checkpoint checkpoints/model.pt --jsonl corpus/train.jsonl
```

Query:

```bash
ctms index query --checkpoint checkpoints/model.pt \
  --index indices/meaning_index.pt --text "..."
```

Ablate:

```bash
ctms ablate --text corpus/eval.txt
```

Benchmark:

```bash
ctms benchmark
```

Render PCM:

```bash
ctms dac render --checkpoint checkpoints/model.pt --text "..."
```

---

# 47. FINAL OPERATOR RULE

The CLI exists to make the system inspectable.

Do not use the command line merely to produce an answer.

Use it to preserve the relation between:

```text
source
configuration
operator
matrix
relational field
memory
checkpoint
evaluation
output
```

The emitted token stream is the transmission surface.

The experiment is everything that had to remain visible before it was flattened.
