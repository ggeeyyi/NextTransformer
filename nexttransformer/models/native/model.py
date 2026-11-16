"""Native nanochat blocks with explicit sharing, causal routes and FLOP accounting.

Dense delegates directly to the pinned upstream GPT forward. Sparse MoR keeps
original RoPE positions and window distances when compacting surviving tokens.
Circle gathers surviving Q/K/V and MLP; selected residual gates train the halter.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE
from nexttransformer.models.backbone import GPT, GPTConfig, norm, apply_rotary_emb
from nexttransformer.models.native.conditioning import LoopConditioner
from nexttransformer.models.native.budget import SurvivalBudget, CircleBudget
from nexttransformer.models.native.halter import DeltaDepthHalter
from nexttransformer.models.native.circle import SparseCircleMixin
from nexttransformer.models.native.depth import depth_reference


def uses_value_embeddings(config):
    """The 2K scaling shell can switch them off; the 4K native comparison always ran with them."""
    return getattr(config, 'value_embeddings', True)


def window(config, virtual_layer):
    short = -(-config.sequence_len//4//128)*128
    full = (virtual_layer == config.max_depth-1 or
            config.window_pattern[virtual_layer % len(config.window_pattern)] == 'L')
    return (config.sequence_len if full else short, 0)


def block_forward_flops(config, batch, queries, keys=None, window_size=None):
    """Dense-matmul convention matching upstream estimate_flops()/3.

    Lookup/scalar/elementwise FLOPs are excluded, as in upstream. For a sparse
    explicit-mask attention kernel, executed attention is rectangular, even
    though many cells are masked. This conservative estimate includes padding.
    """
    d = config.width
    k = queries if keys is None else keys
    attended = min(k, window_size[0]) if window_size is not None else k
    return batch*(24*queries*d*d + 4*queries*attended*d)


def theoretical_training_flops(config):
    c = config
    t, d = c.sequence_len, c.width
    # Ideal routes: exact target capacity; actual padded execution is logged.
    if c.architecture == 'mor':
        rates = [1.]*c.pre_layers
        for rate in (1., *c.mor_target_survival): rates += [rate]*c.core_layers
        rates += [1.]*c.post_layers
    elif c.architecture == 'circle':
        _,ring_rates = depth_reference(c.core_layers,c.circle_max_steps,c.circle_target_steps)
        rates = [1.]*c.pre_layers+list(ring_rates)+[1.]*c.post_layers
    else:
        rates = [1.]*c.max_depth
    forward = 2*t*d*c.vocab_size + 2*t*24  # head + smear gate
    for j, rate in enumerate(rates):
        n = t*rate
        # Sparse MoR uses explicit original-position masks and rectangular QK.
        sparse = ((c.architecture == 'mor' and rate < 1) or
                  (c.architecture == 'circle' and c.pre_layers+c.core_layers <= j < len(rates)-c.post_layers))
        w = None if sparse else window(c, j)
        forward += block_forward_flops(c, 1, n, window_size=w)
        # VE gate on physical alternating layers (last physical layer has VE).
        if c.architecture == 'dense': physical = j
        elif j < c.pre_layers: physical = j
        elif j >= len(rates)-c.post_layers: physical = c.pre_layers+c.core_layers+j-(len(rates)-c.post_layers)
        else: physical = c.pre_layers+(j-c.pre_layers)%c.core_layers
        if uses_value_embeddings(c) and physical % 2 == (c.n_layer-1) % 2: forward += 24*c.heads*n
    if c.architecture == 'mor': forward += 2*(c.recursions-1)*t*d
    if c.architecture == 'circle': forward += 2*t*d*sum(ring_rates[c.core_layers-1:-1])
    if c.architecture == 'loopformer':
        # Two timestep MLPs once per recursion, one modulation per physical block.
        forward += c.recursions*(4*(c.fourier_dim*d+d*d)+8*c.core_layers*d*d)
    return 3*forward/t


class NativeExperiment(SparseCircleMixin, GPT):
    def __init__(self, config):
        native = GPTConfig(sequence_len=config.sequence_len, vocab_size=config.vocab_size,
                           n_layer=config.n_layer, n_embd=config.width, n_head=config.heads,
                           n_kv_head=config.heads, window_pattern=config.window_pattern)
        super().__init__(native, pad_vocab_size_to=64)
        self.config = config
        self.max_seq_len = config.sequence_len
        self.last_stats = {}
        if not uses_value_embeddings(config):
            self.value_embeds = nn.ModuleDict()
            for block in self.transformer.h: block.attn.ve_gate = None
        if config.architecture == 'mor':
            self.routers = nn.ModuleList(nn.Linear(config.width, 1) for _ in range(config.recursions-1))
            self.survival_budget = SurvivalBudget(config)
        elif config.architecture == 'circle':
            self.halter = DeltaDepthHalter(config.width,config.circle_max_steps-config.core_layers,config.circle_halter_eps)
            self.circle_budget = CircleBudget(config)
        elif config.architecture == 'loopformer':
            self.conditioner = LoopConditioner(config)

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        c = self.config
        if c.architecture == 'mor':
            previous = 1.
            for router, target in zip(self.routers, c.mor_target_survival):
                nn.init.normal_(router.weight, std=c.width**-.5)
                # Scores initially have roughly unit variance after RMSNorm.
                quantile = torch.distributions.Normal(0., 1.).icdf(torch.tensor(target/previous))
                nn.init.constant_(router.bias, float(quantile))
                previous = target
            self.survival_budget.dual.zero_(); self.survival_budget.updates.zero_()
            self.survival_budget.last_step.fill_(-1)
        elif c.architecture == 'circle':
            self.circle_budget.reset()
            _,survival = depth_reference(c.core_layers,c.circle_max_steps,c.circle_target_steps)
            self.halter.init_weights([survival[s]/survival[s-1] for s in range(c.core_layers,c.circle_max_steps)])
        elif c.architecture == 'loopformer':
            self.conditioner.init_weights()

    def _stats(self, idx):
        return dict(input_tokens=idx.numel(), forward_flops=0., recompute_flops=0.,
                    active_tokens=[0]*self.config.routing_steps, active_depth_sum=0.,halter_extra_forward_ops=0.)

    def _embed(self, idx):
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if x.shape[1] > 1:
            gate = self.smear_lambda.to(x.dtype)*torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat((x[:, :1], x[:, 1:]+gate*x[:, :-1]), 1)
        self.last_stats['forward_flops'] += 2*idx.numel()*24
        return x

    def _layer(self, x, x0, idx, physical, virtual, conditioning=None, active=None):
        c = self.config
        b, n, d = x.shape
        block = self.transformer.h[physical]
        w = window(c, virtual)
        has_ve = str(physical) in self.value_embeds
        cos, sin = self.cos[:, :n], self.sin[:, :n]
        if active is None:
            h = self.resid_lambdas[physical]*x + self.x0_lambdas[physical]*x0
            ve = self.value_embeds[str(physical)](idx).to(x.dtype) if has_ve else None
            if conditioning is None:
                out = block(h, ve, (cos, sin), w, None)
            else:
                # Identity modulation init: native c_proj starts at zero, so
                # zero-initialized multiplicative gates would kill both gradients.
                ga, gm, sa, sm = conditioning.to(h.dtype).chunk(4, -1)
                out = h+(1+ga[:, None])*block.attn(norm(h)*(1+sa[:, None]), ve, (cos, sin), w, None)
                out = out+(1+gm[:, None])*block.mlp(norm(out)*(1+sm[:, None]))
            cost = block_forward_flops(c, b, n, window_size=w)
            ve_count = b*n
        else:
            # Compact surviving tokens, keeping original positions for RoPE and
            # window masks. Padded queries never scatter into a real token.
            m = int(active.sum(1).max().item())
            if m == 0: return x
            positions = torch.arange(n, device=x.device).expand(b, -1).masked_fill(~active, n).sort(1).values[:, :m]
            valid = positions < n
            positions = positions.clamp_max(n-1)
            gather = positions[:, :, None].expand(-1, -1, d)
            selected = (self.resid_lambdas[physical]*x.gather(1,gather)
                        +self.x0_lambdas[physical]*x0.gather(1,gather))
            z = norm(selected)
            attn = block.attn
            q = attn.c_q(z).view(b, m, c.heads, d//c.heads)
            k = attn.c_k(z).view_as(q); v = attn.c_v(z).view_as(q)
            if has_ve:
                sve = self.value_embeds[str(physical)](idx.gather(1,positions)).to(x.dtype).view_as(v)
                gate = 3*torch.sigmoid(attn.ve_gate(z[..., :12]))
                v = v+gate.unsqueeze(-1)*sve
            cs, sn = cos[0, positions], sin[0, positions]
            q = norm(apply_rotary_emb(q, cs, sn))*1.2
            k = norm(apply_rotary_emb(k, cs, sn))*1.2
            distance = positions[:, :, None]-positions[:, None, :]
            allowed = (distance >= 0) & (distance <= w[0]) & valid[:, :, None] & valid[:, None, :]
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                              attn_mask=allowed[:, None]).transpose(1, 2).reshape(b, m, d)
            candidate = selected+attn.c_proj(y)
            candidate = candidate+block.mlp(norm(candidate))
            # Replacement is relative to the original state, including scalar mixing.
            delta = (candidate-x.gather(1, gather))*valid[:, :, None].to(x.dtype)
            out = x.scatter_add(1, gather, delta)
            cost = block_forward_flops(c, b, m)  # padded rectangular execution
            ve_count = b*m
        if has_ve: cost += 24*c.heads*ve_count
        self.last_stats['forward_flops'] += cost
        return out

    def _variant_hidden(self, idx):
        c = self.config
        b, n = idx.shape
        x = self._embed(idx); x0 = x
        virtual = 0; backout = None

        def run(physical, active=None, conditioning=None):
            nonlocal x, virtual, backout
            x = self._layer(x, x0, idx, physical, virtual, conditioning, active)
            if virtual == c.max_depth//2: backout = x
            virtual += 1

        if c.architecture == 'loopformer':
            for i in range(c.pre_layers):run(i)
            for r in range(c.recursions):
                conditioning = self.conditioner(r,b,x.device)
                self.last_stats['forward_flops'] += 4*b*(c.fourier_dim*c.width+c.width*c.width)
                for i,modulation in enumerate(conditioning):
                    self.last_stats['forward_flops'] += 8*b*c.width*c.width
                    run(c.pre_layers+i, conditioning=modulation)
                self.last_stats['active_tokens'][r] = b*n
            for i in range(c.post_layers):run(c.pre_layers+c.core_layers+i)
            self.last_stats['active_depth_sum'] = b*n*c.max_depth
            penalty = x.new_zeros((), dtype=torch.float32)
        else:
            for i in range(c.pre_layers): run(i)
            if c.architecture == 'mor':
                alive = torch.ones((b, n), device=x.device)
                means = []
                for r in range(c.recursions):
                    if r:
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            logits = self.routers[r-1](norm(x.float())).squeeze(-1)
                            q = logits.sigmoid()
                            hard = (logits >= 0).float()
                            alive = alive*(hard-q.detach()+q if self.training else hard)
                        means.append(alive.mean())
                        self.last_stats['forward_flops'] += 2*b*n*c.width
                    active = alive.detach() > .5
                    self.last_stats['active_tokens'][r] = int(active.sum())
                    before = x
                    for i in range(c.core_layers): run(c.pre_layers+i, active=None if r == 0 else active)
                    if r:
                        # Same gated residual in train/eval. Selection itself is
                        # causal; no sequence top-k or batch-dependent threshold.
                        x = before+q.to(x.dtype)[:, :, None]*(x-before)
                penalty = self.survival_budget.penalty(torch.stack(means)) if self.training else x.new_zeros(())
                self.last_stats['active_depth_sum'] = b*n*(c.pre_layers+c.post_layers)+c.core_layers*sum(self.last_stats['active_tokens'])
            else:
                def circle_run(physical, state, active=None):
                    nonlocal x
                    x = state
                    run(physical,active=active)
                    return x
                x,penalty,backout = self._sparse_circle(x,idx,circle_run)
                x = norm(x)
            for i in range(c.post_layers): run(c.pre_layers+c.core_layers+i)
        if backout is not None: x = x-self.backout_lambda.to(x.dtype)*backout
        return norm(x), penalty

    def forward(self, idx, targets=None, loss_reduction=None):
        if idx.shape[1] > self.max_seq_len: raise ValueError('input exceeds experiment context')
        self.last_stats = self._stats(idx)
        if self.config.architecture == 'dense':
            # Keep native forward/initialization/state dict exactly equivalent.
            logits = super().forward(idx)
            d = self.config.width
            flops = 2*self.num_matmul_params()+sum(4*d*min(w[0],idx.shape[1]) for w in self.window_sizes)
            self.last_stats.update(forward_flops=idx.numel()*flops,
                                   active_depth_sum=idx.numel()*self.config.dense_layers)
            penalty = logits.new_zeros(())
        else:
            hidden, penalty = self._variant_hidden(idx)
            logits = self.lm_head(hidden)[..., :self.config.vocab_size].float()
            logits = 15*torch.tanh(logits/15)
            self.last_stats['forward_flops'] += 2*idx.numel()*self.config.width*self.lm_head.out_features
        if targets is None: return logits
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                 reduction='none', ignore_index=-1)
        if loss_reduction is not None:
            if loss_reduction == 'none': return losses
            if loss_reduction == 'sum': return losses.sum()
            if loss_reduction == 'mean': return losses.mean()
            raise ValueError('invalid loss reduction')
        lm = losses.mean()
        # Skipped physical blocks still receive explicit zero gradients on every
        # rank, preserving DDP and optimizer state-sharding collective order.
        zero = sum(p.reshape(-1)[0]*0 for p in self.parameters()) if self.training else lm.new_zeros(())
        return dict(loss=lm+penalty+zero,lm_loss=lm,router_loss=penalty,objective=lm+penalty.detach(),
                    compute_loss=self.last_stats.get('circle_depth_loss',lm.new_zeros(())))

    def update_route_budget(self, counts, tokens, step):
        if not self.training: raise ValueError('cannot change routing budgets in evaluation')
        if self.config.architecture == 'mor': return self.survival_budget.update(counts[1:], tokens, step)
        if self.config.architecture == 'circle': return self.circle_budget.update(counts, tokens, step)
        return {}

    def estimate_flops(self):
        # The inherited physical-layer estimate is valid only for Dense.
        return theoretical_training_flops(self.config)


def build_model(config, device):
    with torch.device('meta'): model = NativeExperiment(config)
    model.to_empty(device=device)
    model.init_weights()
    return model
