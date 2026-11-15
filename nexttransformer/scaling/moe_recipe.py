"""Architecture and native-cost estimates for the 2K scaling MoE family.

Routing coefficients are pilot settings, not a registered long-training recipe.
This module does not import Megatron and can describe every T3 cell on CPU.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

TESTED_MCORE_COMMIT = "b6d7b650df555987a618ec440654306b9c8539a8"


@dataclass(frozen=True)
class MoERecipe:
    layers: int
    width: int
    heads: int
    vocab_size: int = 32768
    sequence_len: int = 2048
    routed_experts: int = 256
    top_k: int = 8
    expert_width: int | None = None
    seed: int = 42
    global_batch_tokens: int = 1048576
    training_flops: int = 10**20
    cell_id: str = ""
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001
    router_init_std: float = 0.02

    def __post_init__(self):
        if self.expert_width is None:
            object.__setattr__(self, "expert_width", self.width // 4)
        if min(self.layers, self.width, self.heads, self.vocab_size, self.sequence_len,
               self.routed_experts, self.expert_width, self.global_batch_tokens, self.training_flops) <= 0:
            raise ValueError("model dimensions and budgets must be positive")
        if self.layers < 2 or self.width < 24 or self.width % self.heads or self.width % 4:
            raise ValueError("MoE needs >=2 layers, width>=24 divisible by heads and four")
        if (self.width // self.heads) % 2 or self.expert_width != self.width // 4:
            raise ValueError("even head dimension and two-matrix d/4 experts are required")
        if not 1 <= self.top_k <= self.routed_experts:
            raise ValueError("top-k must fit the expert count")
        if self.global_batch_tokens % self.sequence_len:
            raise ValueError("batch must contain whole sequences")
        if not all(math.isfinite(x) and x >= 0 for x in (self.aux_loss_coeff, self.z_loss_coeff)):
            raise ValueError("routing loss coefficients must be finite and nonnegative")
        if not math.isfinite(self.router_init_std) or self.router_init_std <= 0:
            raise ValueError("router initialization standard deviation must be positive")

    def counts(self):
        d, e, h = self.width, self.routed_experts, self.expert_width
        attention = 4 * self.layers * d * d
        first_ffn = 8 * d * d
        expert = 2 * d * h
        routers = (self.layers - 1) * d * e
        total = attention + first_ffn + (self.layers - 1) * (e + 1) * expert + routers
        active = attention + first_ffn + (self.layers - 1) * (self.top_k + 1) * expert + routers
        # NextT's inherited residual/smear/backout parameters, unchanged by this adapter.
        small = 2 * self.layers + 26
        io = 2 * d * math.ceil(self.vocab_size / 64) * 64
        return dict(block_total=total, block_active=active, embedding_and_head=io,
                    native_small_parameters=small, model_total=total + io + small)

    def estimate(self):
        # Native main-term convention; elementwise/auxiliary and engineering costs still
        # need the same operator-level cost proof as the other scaling families.
        d, s = self.width, self.sequence_len
        flops = 6 * self.counts()["block_active"] + 12 * s * d * self.layers + 6 * d * self.vocab_size
        updates = round(self.training_flops / (flops * self.global_batch_tokens))
        return dict(flops_per_token_main_term=flops, estimated_updates=updates,
                    estimated_tokens=updates * self.global_batch_tokens, cost_proof_complete=False)

    def manifest(self):
        return dict(protocol="nextt-scale-2k-v1", recipe=asdict(self), counts=self.counts(),
                    estimate=self.estimate(), attention="native full causal MHA; no VE",
                    ffn="first dense 4d; then routed top-k plus one shared d/4; ReLU squared",
                    optimizer="native Muon + AdamW; AdamW router; unsharded states in initial adapter",
                    routing_status="pilot coefficients; not frozen for formal training",
                    tested_megatron_commit=TESTED_MCORE_COMMIT, gpu_validated=False,
                    formal_training_ready=False)

    def debug(self, *, routed_experts=8, top_k=2):
        return replace(self, layers=3, width=32, heads=2, vocab_size=64, sequence_len=8,
                       routed_experts=routed_experts, top_k=top_k, expert_width=8,
                       global_batch_tokens=32, cell_id=self.cell_id + "-cpu-debug")


def load_t3_recipe(cell_id="moe-T3-L20", path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/scaling/t3.json"
    data = json.loads(path.read_text())
    cell = next((c for c in data["cells"] if c["id"] == cell_id and c["family"] == "moe"), None)
    if cell is None:
        raise ValueError(f"unknown T3 MoE cell: {cell_id}")
    rule = next(f["recipe"] for f in data["families"] if f["id"] == "moe")
    expected = dict(attention="full causal MHA", firstLayerDense=True, ffn="ReLU²",
                    routedExperts=256, selectedExperts=8, sharedExperts=1, expertWidthDivisor=4)
    if rule != expected or data["valueEmbeddings"] or data["protocol"] != "nextt-scale-2k-v1":
        raise ValueError("T3 MoE architecture changed; review the adapter before using it")
    return MoERecipe(layers=cell["layers"], width=cell["width"], heads=cell["heads"],
                     vocab_size=data["vocabSize"], sequence_len=data["sequenceLength"],
                     seed=data["seed"], global_batch_tokens=data["globalBatchTokens"],
                     training_flops=data["trainingFlops"], expert_width=cell["expertWidth"], cell_id=cell_id)
