"""
BBB gMLP model + Optuna hyperparameter optimization.
This is the file the autoresearch agent modifies.

The agent may freely change:
- Model architecture: SpatialGatingUnit, gMLPBlock, gMLP, MultiModalGMLPFromFlat
- Hyperparameter search space inside objective()
- Training details inside train_model() (optimizer type, scheduler, etc.)

The agent must NOT change:
- bbb_prepare.py (data loading, splitting, evaluation)
- The composite_score metric (f1 + mcc + roc_auc)
- The output format (the --- block at the end of __main__)

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_train.py > bbb_run.log 2>&1
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbb_prepare import *  # fixed utilities, constants, device

import optuna
from optuna.trial import Trial
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Model Architecture  (agent modifies this section)
# ---------------------------------------------------------------------------

class CrossModalFiLM(nn.Module):
    """Cross-modal FiLM conditioning before gMLP.
    Computes a global context from all modality tokens,
    then predicts per-modality scale (gamma) and shift (beta).
    token_i' = token_i * (1 + gamma_i) + beta_i
    Zero-initialized output projections → identity at init for stability.
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.norm         = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, d_model)
        self.gamma_proj   = nn.Linear(d_model, d_model * seq_len)
        self.beta_proj    = nn.Linear(d_model, d_model * seq_len)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x):
        # x: (B, seq_len, d_model)
        ctx   = F.gelu(self.context_proj(self.norm(x.mean(dim=1))))  # (B, d_model)
        gamma = self.gamma_proj(ctx).view(-1, self.seq_len, self.d_model)
        beta  = self.beta_proj(ctx).view(-1, self.seq_len, self.d_model)
        return x * (1 + gamma) + beta


class CrossModalAdaLN(nn.Module):
    """Adaptive LayerNorm conditioning before gMLP (DiT-style).
    Zero-initialized modulation → identity at init for stability.
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.norm         = nn.LayerNorm(d_model, elementwise_affine=False)
        self.context_proj = nn.Linear(d_model, d_model)
        self.modulation   = nn.Linear(d_model, 2 * d_model * seq_len)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x):
        # x: (B, seq_len, d_model)
        ctx = F.silu(self.context_proj(x.mean(dim=1)))             # (B, d_model)
        mod = self.modulation(ctx).view(-1, self.seq_len, 2 * self.d_model)
        gamma, beta = mod.chunk(2, dim=-1)                         # each (B, seq_len, d_model)
        return self.norm(x) * (1 + gamma) + beta


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        # Learnable residual scale: init=1 (same as plain residual at start)
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return residual + self.res_scale * x


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=8, num_layers=4):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1024,
                 depth=4, dropout=0.2, use_gated_pool=True, cond_type='none',
                 modal_drop_p=0.0):
        super().__init__()
        self.mod_names    = list(mod_dims.keys())
        self.mod_dims     = [mod_dims[n] for n in self.mod_names]
        self.seq_len      = len(self.mod_names)
        self.use_gated_pool = use_gated_pool
        self.modal_drop_p = modal_drop_p

        # LayerNorm after projection normalizes diverse modality scales (binary FPs vs embeddings)
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })

        # Cross-modal conditioning before gMLP (none / film / adaLN)
        if cond_type == 'film':
            self.cross_cond = CrossModalFiLM(d_model, self.seq_len)
        elif cond_type == 'adaLN':
            self.cross_cond = CrossModalAdaLN(d_model, self.seq_len)
        else:
            self.cross_cond = None

        self.backbone = gMLP(
            d_model=d_model, d_ffn=d_ffn, seq_len=self.seq_len, num_layers=depth
        )
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk) for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)           # (B, seq_len, d_model)

        # Modality dropout: randomly zero one modality token per sample during training
        if self.training and self.modal_drop_p > 0.0:
            B = X.size(0)
            mask = torch.ones(B, self.seq_len, 1, device=X.device)
            drop_indices = torch.bernoulli(
                torch.full((B, self.seq_len), self.modal_drop_p, device=X.device)
            ).bool()
            # Zero at most one modality per sample (the first one that fires)
            first_drop = drop_indices.float().argmax(dim=1)  # (B,) index to drop
            any_drop   = drop_indices.any(dim=1)             # (B,) bool
            for b in range(B):
                if any_drop[b]:
                    mask[b, first_drop[b], 0] = 0.0
            X = X * mask

        if self.cross_cond is not None:
            X = self.cross_cond(X)               # cross-modal conditioning
        X = self.backbone(X)
        if self.use_gated_pool:
            w  = torch.softmax(self.alpha, dim=0)
            Xp = (X * w.view(1, -1, 1)).sum(dim=1)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)

# ---------------------------------------------------------------------------
# Training loop  (agent may modify optimizer, scheduler, etc.)
# ---------------------------------------------------------------------------

def train_model(model, optimizer, train_loader, val_loader, loss_fn,
                num_epochs=NUM_EPOCHS, patience=PATIENCE):
    # Early-stop on composite score (f1+mcc+roc_auc) instead of val loss.
    # Loss uses smoothed labels, so loss-optimal != metric-optimal.
    best_score = -float('inf')
    best_state = None
    bad        = 0

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_score = composite_score(eval_model(model, val_loader))

        if val_score > best_score:
            best_score = val_score
            best_state = deepcopy(model.state_dict())
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------------------
# Optuna objective  (agent may modify the search space)
# ---------------------------------------------------------------------------

HPO_SEEDS = [42, 600, 900]  # 3-seed average for stable val score estimation


def objective(trial: Trial, hpo_splits, mod_dims):
    # --- Hyperparameter search space (agent can modify ranges / add params) ---
    d_model        = trial.suggest_categorical('d_model', [256, 512, 768, 1024])
    depth          = trial.suggest_int('depth', 2, 8, step=2)
    ffn_multiplier = trial.suggest_categorical('ffn_multiplier', [2, 3, 4])
    d_ffn          = d_model * ffn_multiplier
    dropout        = trial.suggest_float('dropout', 0.1, 0.4)
    lr             = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay   = trial.suggest_float('weight_decay', 5e-6, 5e-4, log=True)
    cond_type      = trial.suggest_categorical('cond_type', ['none', 'film', 'adaLN'])
    modal_drop_p   = trial.suggest_float('modal_drop_p', 0.0, 0.4)
    ls_eps         = trial.suggest_float('ls_eps', 0.0, 0.25)

    seed_scores = []
    for seed, (train_ds, val_ds) in hpo_splits.items():
        set_seed(seed)
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
        val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = MultiModalGMLPFromFlat(
            mod_dims=mod_dims, d_model=d_model, d_ffn=d_ffn,
            depth=depth, dropout=dropout, use_gated_pool=True,
            cond_type=cond_type, modal_drop_p=modal_drop_p,
        ).to(device)

        y_train   = train_ds.tensors[1].cpu().numpy()
        n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
        pos_weight = None
        if n_pos > 0:
            pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device)

        optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        base_loss = (nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                     if pos_weight is not None else nn.BCEWithLogitsLoss(reduction='none'))

        def loss_fn(logits, targets):
            # Label smoothing for better calibration and generalization
            smooth = targets * (1.0 - ls_eps) + 0.5 * ls_eps
            return base_loss(logits, smooth).mean()

        model = train_model(model, optimizer_obj, train_loader, val_loader, loss_fn)

        # Guard against NaN outputs (can happen with conditioning at bad HPs)
        model.eval()
        with torch.no_grad():
            sample_x = val_ds.tensors[0][:4].to(device)
            if torch.isnan(model(sample_x)).any() or torch.isinf(model(sample_x)).any():
                raise optuna.exceptions.TrialPruned()

        metrics = eval_model(model, val_loader)
        seed_scores.append(composite_score(metrics))

    score = float(np.mean(seed_scores))
    trial.report(score, step=NUM_EPOCHS)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return score

# ---------------------------------------------------------------------------
# Per-split-mode study runner
# ---------------------------------------------------------------------------

def run_study(dataset, mod_dims, split_mode):
    # Pre-split for all HPO seeds so each trial uses identical splits
    hpo_splits = {}
    for seed in HPO_SEEDS:
        train_ds, val_ds, _, _, _, _ = split_then_normalize(
            dataset, split_mode=split_mode,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )
        hpo_splits[seed] = (train_ds, val_ds)

    study = optuna.create_study(
        study_name=f'bbb_gmlp_{split_mode}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )
    study.optimize(
        lambda trial: objective(trial, hpo_splits, mod_dims),
        n_trials=N_OPTUNA_TRIALS,
        gc_after_trial=True,
    )
    # Decode d_ffn from ffn_multiplier
    best = study.best_params.copy()
    best['d_ffn'] = best.pop('d_model') * best.pop('ffn_multiplier')
    best['d_model'] = study.best_params['d_model']  # restore
    return study.best_value, study.best_params, study

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t_start = time.time()

    print('Loading BBB dataset (once)...')
    dataset       = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims      = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)
    print(f'Dataset: {len(dataset)} samples, feature_dim={dataset.features.shape[1]}')
    print(f'Mod dims: {dict(mod_dims)}')

    best_scores     = {}
    best_params_all = {}

    for split_mode in SPLIT_MODES:
        print(f"\n{'='*60}")
        print(f"Optuna HPO | split_mode={split_mode} | n_trials={N_OPTUNA_TRIALS}")
        print('='*60)

        best_val, best_params, study = run_study(dataset, mod_dims, split_mode)
        best_scores[split_mode]     = best_val
        best_params_all[split_mode] = best_params

        print(f"[{split_mode}] Best composite score : {best_val:.4f}")
        print(f"[{split_mode}] Best params          : {best_params}")

    # Save best params for analysis / next experiments
    os.makedirs(os.path.join(os.path.dirname(__file__), 'bbb_artifacts'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), 'bbb_artifacts', 'best_params.json')
    with open(out_path, 'w') as f:
        json.dump(best_params_all, f, indent=2)
    print(f'\nBest params saved to {out_path}')

    # -----------------------------------------------------------------------
    # Output block — DO NOT CHANGE FORMAT (parsed by autoresearch loop)
    # -----------------------------------------------------------------------
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                    if torch.cuda.is_available() else 0.0)
    print('\n---')
    for sm in SPLIT_MODES:
        key = sm.replace('_', '')
        print(f'composite_{key}:   {best_scores[sm]:.6f}')
    print(f'total_seconds:     {time.time() - t_start:.1f}')
    print(f'peak_vram_mb:      {peak_vram_mb:.1f}')
    print(f'n_trials:          {N_OPTUNA_TRIALS}')
