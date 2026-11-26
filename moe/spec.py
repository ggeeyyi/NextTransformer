"""The frozen MoE comparison protocol: shared controls plus two tracks (d768, d1024).

Everything a method may not change lives here. A configuration file carries the protocol and track
fingerprints; loading it fails when either differs, exactly like nexttransformer.protocol.spec.

Shapes and budgets follow Marin's "May Recipe" compute-optimal Grug MoE cells
(marin-community/marin, experiments/grug/moe/README.md, issue #6074):

    moe_may_compute_opt_d768_ep1    8 layers, 64 x 4096-token rows, 16,875 updates, 4.42e9 tokens
    moe_may_compute_opt_d1024_ep1  11 layers, 128 x 4096-token rows, 16,080 updates, 8.43e9 tokens

The model, optimizer, schedule and balancing are Marin's (see moe/model.py and moe/optim.py). The tokenizer and
data differ (Dolma 3 HQ with the dolma2 tokenizer instead of Nemotron-CC with the Llama 3 tokenizer), so absolute
losses are not comparable with Marin's Paloma numbers.
"""
import math
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"

# Data: xjxyys/dolma3_hq_300B_shuffled_clean, pinned. Shards are uniform random samples and each is shuffled.
DATA_REPO = "xjxyys/dolma3_hq_300B_shuffled_clean"
DATA_REVISION = "128a685684288a0953b1e7cec5f3bf57bf80dc73"
TRAIN_SHARD = "data/train-{index:05d}-of-00256.parquet"
TRAIN_SHARDS = 256
DEV_SHARD = 255  # model selection only; never part of a training stream (streams use shards 0, 1, 2, ...)
VALIDATION_SOURCES = ("common_crawl", "olmocr_science_pdfs", "stack_edu", "finemath-3plus",
                      "rpj-proofpile-arxiv", "dolma1_7-wiki-en")
VALIDATION_FILE = "validation/{source}.parquet"

# Tokenizer: allenai/dolma2-tokenizer (cl100k_base ranks + 22 added tokens), reproduced with tiktoken.
TOKENIZER_REPO = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_JSON_SHA256 = "969c214487b744f1457d8a4f2055fd4ad348edff5322d4f21f2906ceb158a636"


@dataclass(frozen=True)
class Track:
    """One comparison track: the reference shape and the training budget every method shares."""
    name: str
    marin_run: str
    width: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    routed_experts: int
    top_k: int
    expert_width: int
    shared_expert_width: int
    rows_per_update: int
    updates: int
    qb_group_rows: int              # rows per quantile-balancing group: one Marin device's share (16 groups/update)
    core_steps: tuple
    marin_budget_flops: float       # Marin's own C = 3 * forward FLOPs (no LM head, non-causal attention) * tokens
    marin_tokens_per_second: int    # Marin's reported throughput for the cell (TPU v4-32, EP=1), for reference only

    @property
    def tokens_per_update(self):
        return self.rows_per_update * PROTOCOL.sequence_len

    @property
    def total_tokens(self):
        return self.updates * self.tokens_per_update

    @property
    def warmup_steps(self):
        return int(PROTOCOL.warmup_fraction * self.updates)

    @property
    def adam_lr(self):
        """Marin's corrected law (issue #5951): 0.06602 * tokens^-0.395 * width^-0.150 * sqrt(tokens per update)."""
        return min(0.05, 0.06602 * self.total_tokens ** -0.395 * self.width ** -0.150 * math.sqrt(self.tokens_per_update))

    @property
    def muonh_lr(self):
        return 13 / 3 * self.adam_lr

    @property
    def beta2(self):
        return min(0.9999, max(0.95, 0.999 ** (self.tokens_per_update / 131072)))

    @property
    def eps(self):
        return 9.676e-18 * math.sqrt(self.total_tokens / self.tokens_per_update)

    @property
    def peak_lr(self):
        return self.muonh_lr

    @property
    def validation_steps(self):
        every = PROTOCOL.val_every
        return (0, *range(every, self.updates, every), self.updates)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Protocol:
    version: str = "nextt-moe-v1"
    data_repo: str = DATA_REPO
    data_revision: str = DATA_REVISION
    tokenizer_repo: str = TOKENIZER_REPO
    tokenizer_revision: str = TOKENIZER_REVISION
    tokenizer_sha256: str = TOKENIZER_JSON_SHA256
    vocab_size: int = 100278            # dolma2 tokenizer
    padded_vocab_size: int = 100352     # embedding and head rows (multiple of 128); padding rows never targeted
    document_separator: str = "<|endoftext|>"   # prepended to every document; also the BOS of every eval prompt
    packing: str = "concatenate-and-chunk"      # documents in file order, rows of sequence_len + 1 overlapping by one
    sequence_len: int = 4096
    sliding_window: int = 2048          # short layers; long layers use full causal attention
    long_layer_every: int = 4           # layers 3, 7, 11, ... and the last layer are long
    seed: int = 42
    data_parallel_size: int = 8
    dtype: str = "bfloat16"
    # Optimizer: Marin's MuonH recipe (moe/optim.py). Matrices: MuonH (Nesterov momentum 0.95, five quintic
    # Newton-Schulz steps per matrix, Frobenius-hyperball update that keeps each matrix's norm). Head: AdamH.
    # Embedding, router, head-wise gates and gains: Adam. Learning rates, beta2 and eps per track (Track).
    optimizer: str = "muonh"
    beta1: float = 0.9062
    muon_momentum: float = 0.95
    newton_schulz_steps: int = 5
    weight_decay: float = 0.0
    warmup_fraction: float = 0.01
    schedule: str = "linear-decay"      # lr = peak * t / W for t < W (zero at step 0), then linear decay to 0
    final_lr_fraction: float = 0.0
    grad_clip: float = 0.0              # none
    # Evaluation cadence (identical for both tracks)
    val_every: int = 500
    val_rows_per_source: int = 128      # 6 sources x 128 rows x 4096 = 3,145,728 tokens per validation event
    final_val: str = "all complete rows of every validation source"
    core_max_tokens: int = 4096         # CORE prompts are left-truncated to the training context
    # Compute control, as in the dense protocol
    preflight_tolerance: float = 0.01   # admitted configuration: FLOPs/token within 1% of the track anchor
    cumulative_tolerance: float = 0.02  # finished run: charged FLOPs within 2% of the anchor budget
    tracks: tuple = field(default=("d768", "d1024"))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


PROTOCOL = Protocol()

TRACKS = {
    "d768": Track(
        name="d768", marin_run="moe_may_compute_opt_d768_ep1",
        width=768, layers=8, heads=6, kv_heads=1, head_dim=128,
        routed_experts=256, top_k=4, expert_width=384, shared_expert_width=768,
        rows_per_update=64, updates=16_875, qb_group_rows=4,
        core_steps=(4000, 8000, 12000, 16_875),
        marin_budget_flops=2.81e18, marin_tokens_per_second=294_726),
    "d1024": Track(
        name="d1024", marin_run="moe_may_compute_opt_d1024_ep1",
        width=1024, layers=11, heads=8, kv_heads=2, head_dim=128,
        routed_experts=256, top_k=4, expert_width=512, shared_expert_width=1024,
        rows_per_update=128, updates=16_080, qb_group_rows=8,
        core_steps=(4000, 8000, 12000, 16_080),
        marin_budget_flops=1.16e19, marin_tokens_per_second=219_720),
}


def track(name):
    if name not in TRACKS:
        raise ValueError(f"unknown MoE track {name!r}; choose one of {', '.join(TRACKS)}")
    return TRACKS[name]


def long_layers(layers, every=PROTOCOL.long_layer_every):
    """Full-attention layers: every `every`-th layer (1-based) and the last one."""
    return tuple(sorted({i for i in range(layers) if (i + 1) % every == 0} | {layers - 1}))


def schedule(step, track):
    """Marin's multiplier of the peak rate: linear warmup from zero over int(1% of updates), then linear decay."""
    warmup = max(1, track.warmup_steps)
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, track.updates - warmup)
    floor = PROTOCOL.final_lr_fraction
    return floor + (1 - floor) * max(0.0, 1 - progress)


def learning_rate(step, track):
    """The MuonH/AdamH rate; the Adam group uses schedule(step) * track.adam_lr."""
    return track.muonh_lr * schedule(step, track)


def config_path(method, track_name, root=CONFIGS):
    return Path(root) / track_name / f"{method}.json"


def write_config(path, config, calibration):
    from moe.model import MoEConfig
    if not isinstance(config, MoEConfig):
        raise TypeError("write_config takes a MoEConfig")
    tr = track(config.track)
    payload = dict(protocol=asdict(PROTOCOL), protocol_hash=PROTOCOL.fingerprint,
                   track=asdict(tr), track_hash=tr.fingerprint,
                   model=config.serialized(), model_hash=config.fingerprint, calibration=calibration)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def _jsonable(value):
    return json.loads(json.dumps(value))


def load_config(path):
    """An admitted configuration, verified against the current protocol and track."""
    from moe.model import MoEConfig
    data = json.loads(Path(path).read_text())
    if data.get("protocol") != _jsonable(asdict(PROTOCOL)) or data.get("protocol_hash") != PROTOCOL.fingerprint:
        raise ValueError("configuration changes a controlled MoE protocol variable")
    tr = track(data["model"]["track"])
    if data.get("track") != _jsonable(asdict(tr)) or data.get("track_hash") != tr.fingerprint:
        raise ValueError(f"configuration changes the controlled {tr.name} track")
    config = MoEConfig.from_serialized(data["model"])
    if data.get("model_hash") != config.fingerprint:
        raise ValueError("model hash mismatch")
    return config, data.get("calibration", {})
