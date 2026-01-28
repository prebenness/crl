# Run with: python colored_mnist.py
# Requirements:
#   pip install jax jaxlib flax optax torch torchvision matplotlib tqdm

import os
os.environ["WANDB_SILENT"] = "true"     # Stops WandB from cluttering the terminal

import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import wandb
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random as jrandom
import jax.lax as lax
from jax.scipy.special import logsumexp

from flax.training import train_state
import optax

import matplotlib.pyplot as plt

from src.models.ib_classifiers import VIBClassifier
from src.utils.data.datasets import ColoredMNIST, StandardMNISTBin, make_epoch_batches, dataset_to_jax_arrays
from src.utils.plotting.colored_mnist_plots import wandb_summary_plot

# ============================================================
# Global settings (safe on RTX 3080)
# ============================================================

# Generate a unique ID for this entire execution (The "Experiment")
# We use timestamp so all models in this script share it.
WANDB_ENTITY='prebenness-crl'
WANDB_PROJECT='colored-mnist-vib'
LAMBDA_RANGE = (-3, 3, 10)         # (a, b, N) -> N lambda values, evenly logspaced in 10**a to 10**b or linspaced
LOG_SWEEP = True                    # Logspaced or linspaced sweep
timestamp = time.strftime("%Y-%m-%d-T%H-%M-%S", time.gmtime())
experiment_group_id = f"{timestamp}-sweep-lambda-{LAMBDA_RANGE[0]}-{LAMBDA_RANGE[1]}-{LAMBDA_RANGE[2]}"

TRAIN_MC_SAMPLES = 2
EVAL_MC_SAMPLES = 8

# ControlVAE controller gains (tune these; defaults chosen for this classifier/KL scale)
BETA_MIN = 0.0
BETA_MAX = 1.0
# CTRL_KP = 10.0
CTRL_KI = 1.0


@dataclass
class TrainConfig:
    num_classes = 2
    bottleneck_width = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 5
    batch_size: int = 128
    seed: int = 0
    alpha: float = 0.01      # Alpha is share of loss from CE. Alpha = 0 -> only recon loss
    lambdas = jnp.linspace(*LAMBDA_RANGE) if not LOG_SWEEP else jnp.logspace(*LAMBDA_RANGE)


print("JAX devices:", jax.devices())
jax.config.update("jax_default_matmul_precision", "high")


# ============================================================
# Training / eval (epoch-scanned, donated state)
# ============================================================

# ---- ControlVAE controller fixed bounds (not swept) ----
class ControlTrainState(train_state.TrainState):
    beta: jnp.ndarray
    int_err: jnp.ndarray


def create_state(rng, model, input_shape, cfg):
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )["params"]
    tx = optax.adamw(cfg.lr, cfg.weight_decay)
    opt_state = tx.init(params)

    beta0 = jnp.array(BETA_MIN, dtype=jnp.float32)      # Start unconstrained
    int0  = jnp.array(0.0, dtype=jnp.float32)

    return ControlTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        beta=beta0,
        int_err=int0,
    )


@jax.jit
def train_step(state, batch, rng, lamb, alpha):
    x, y = batch
    lamb = jnp.asarray(lamb, jnp.float32)  # capacity target C
    alpha = jnp.asarray(alpha, jnp.float32)

    beta_min = jnp.array(BETA_MIN, dtype=jnp.float32)
    beta_max = jnp.array(BETA_MAX, dtype=jnp.float32)

    def loss_fn(params):
        # ---- Training MC: average CE over K samples of z ~ q(z|x) ----
        keys = jrandom.split(rng, TRAIN_MC_SAMPLES)

        def one_sample(k):
            logits, aux = state.apply_fn(
                {"params": params},
                x,
                train=True,
                rngs={"noise": k},
            )
            kl_raw = aux.get("kl", 0.0)
            x_recon_logits = aux["x_recon_logits"]
            return logits, kl_raw, x_recon_logits

        logits_K, kl_K, x_recon_logits_K = jax.vmap(one_sample)(keys)   # logits_K: [K,B,C], kl_K: [K], x_recon_logits_K: [K,B,H,W,C]

        # Average cross-entropy across samples (same objective as K=1, lower variance)
        def ce_from_logits(lgts):
            return optax.softmax_cross_entropy_with_integer_labels(lgts, y).mean()

        ce_loss = jnp.mean(jax.vmap(ce_from_logits)(logits_K))

        # Reconstruction loss across samples (pixelwise BCE on logits; sum pixels, mean batch)
        def recon_from_logits(xl):
            bce = optax.sigmoid_binary_cross_entropy(xl, x)  # [B,H,W,C]
            #return bce.sum(axis=(1, 2, 3)).mean()
            return bce.mean()
        recon_loss = jnp.mean(jax.vmap(recon_from_logits)(x_recon_logits_K))


        # KL doesn't depend on eps (same across samples), so take first.
        kl = (kl_K[0]).astype(jnp.float32)

        # Use mean logits for reporting acc (cheap, stable)
        logits = jnp.mean(logits_K, axis=0)

        # ---- Dual-ascent (inequality) controller, applied immediately ----
        # Stop-grad so beta update doesn't backprop through kl.
        kl_sg = lax.stop_gradient(kl)

        beta_candidate = state.beta + (CTRL_KI * (kl_sg - lamb))
        beta_used = jnp.clip(beta_candidate, beta_min, beta_max)

        # IMPORTANT: use beta_used (interpreted as lambda in [0,1]) inside this step's loss (to avoid one batch lag)
        lam_sg = lax.stop_gradient(beta_used)
        task_loss = (ce_loss * alpha) + (recon_loss * (1.0 - alpha))
        total_loss = task_loss * (1.0 - lam_sg) + lam_sg * kl
        return total_loss, (logits, ce_loss, recon_loss, kl, beta_used)


    (loss, (logits, ce_loss, recon_loss, kl, beta_used)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # Commit the beta we used for this step (so next step starts from it)
    state = state.replace(beta=beta_used)

    acc = (jnp.argmax(logits, -1) == y).mean()
    metrics = {
        "loss": loss,
        "ce": ce_loss,
        "recon": recon_loss,
        "kl": kl,
        "beta": beta_used,
        "acc": acc,
    }
    return state, metrics


@jax.jit
def eval_step(state, batch, rng):
    x, y = batch  # y: [B]
    keys = jrandom.split(rng, EVAL_MC_SAMPLES)

    def one_sample(k):
        # IMPORTANT: use train=True so VIB samples z via rngs["noise"]
        logits, _ = state.apply_fn(
            {"params": state.params},
            x,
            train=True,
            rngs={"noise": k},
        )
        return logits  # [B, C]

    logits_K = jax.vmap(one_sample)(keys)  # [K, B, C]

    # log p(y|x) = log mean_k softmax(logits_k)
    log_probs_K = jax.nn.log_softmax(logits_K, axis=-1)          # [K, B, C]
    log_probs = logsumexp(log_probs_K, axis=0) - jnp.log(EVAL_MC_SAMPLES)  # [B, C]

    # NLL / CE under the mixture predictive distribution
    nll = -jnp.take_along_axis(log_probs, y[:, None], axis=-1).mean()

    probs = jnp.exp(log_probs)
    acc = (jnp.argmax(probs, axis=-1) == y).mean()
    return nll, acc


def train_epoch(state, xb, yb, rng, lamb, alpha):
    n_batches = xb.shape[0]
    rngs = jrandom.split(rng, n_batches)   # [n_batches, 2]

    def body(carry, inputs):
        st = carry
        x, y, r = inputs                   # unpack one batch + one key
        st, metrics = train_step(st, (x, y), r, lamb, alpha)
        return st, metrics

    state, metrics_history = lax.scan(
        body,
        state,
        (xb, yb, rngs),                    # scan over x, y, rng in parallel
    )

    avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
    return state, avg_metrics

# IMPORTANT: JIT **after** defining function, using wrapper syntax
train_epoch = jax.jit(train_epoch, donate_argnums=(0,))


def eval_epoch(state, xb, yb, rng):
    n_batches = xb.shape[0]
    rngs = jrandom.split(rng, n_batches)

    def body(carry, batch):
        x, y, r = batch
        loss, acc = eval_step(state, (x, y), r)
        return carry, (loss, acc)

    _, (losses, accs) = lax.scan(body, None, (xb, yb, rngs))
    return losses.mean(), accs.mean()

eval_epoch = jax.jit(eval_epoch)


def run_train_eval(x_train, y_train, x_test, y_test, model, cfg, lamb, wandb_run):
    rng = jrandom.PRNGKey(cfg.seed)
    input_shape = (cfg.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state(rng_init, model, input_shape, cfg)

    for ep in range(cfg.epochs):
        t0 = time.time()

        seed_tr = cfg.seed * 10_000 + ep
        seed_te = cfg.seed * 20_000 + ep

        # new RNG for this epoch's noise
        rng, rng_epoch = jrandom.split(rng)

        xb, yb = make_epoch_batches(x_train, y_train, cfg.batch_size, seed_tr)
        xt, yt = make_epoch_batches(x_test,  y_test,  cfg.batch_size, seed_te)


        state, metrics = train_epoch(state, xb, yb, rng_epoch, lamb, cfg.alpha)

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = eval_epoch(state, xt, yt, rng_eval)

        # TODO: fix this lambda meaning. Lammbda 0 means that reg is off, not that capacity is 0

        # --- WANDB: Log Metrics ---
        results = {
            "epoch": ep + 1,
            "train_loss": float(metrics["loss"]),
            "train_acc": float(metrics["acc"]),
            "train_kl": float(metrics["kl"]),   # KL in nats
            "train_recon": float(metrics["recon"]),
            "train_beta": float(metrics["beta"]),   # controller output (lambda in [0,1])
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "cap": float(lamb),
            "kl_err": float(lamb - metrics["kl"]),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.epochs}"
            f"  Acc. Train: {float(metrics['acc']):.4f} Test: {float(te_acc):.4f}"
            f"  Loss Train: {float(metrics['loss']):.4f} Test: {float(te_loss):.4f}"
            f"  Train CE: {float(metrics['ce']):.4f} Recon {float(metrics['recon']):.4f}"
            f"  KL: {float(metrics['kl']):.4f}  Beta: {float(metrics['beta']):.3f}"
            f"      Time: {time.time()-t0:.2f}s"
        )

    # Returns results from final epoch
    return results


# ============================================================
# Sweep + plots + timing
# ============================================================

def main():
    # We will collect results here to log a summary table at the very end
    all_summary_data = []

    cfg = TrainConfig()

    # correlations
    p_train = 0.9
    p_test = 0.1

    print("Loading PyTorch datasets and converting to JAX arrays once...")
    t0 = time.time()

    # train_col = IRMColoredMNIST(train=True,  p_corr=p_train, seed=cfg.seed)
    # test_col  = IRMColoredMNIST(train=False, p_corr=p_test,  seed=cfg.seed + 1)
    train_col = ColoredMNIST(train=True,  p_corr=p_train, seed=cfg.seed)
    test_col  = ColoredMNIST(train=False, p_corr=p_test,  seed=cfg.seed + 1)
    train_std = StandardMNISTBin(train=True)
    test_std  = StandardMNISTBin(train=False)

    x_train_col, y_train_col = dataset_to_jax_arrays(train_col)
    x_test_col,  y_test_col  = dataset_to_jax_arrays(test_col)
    x_train_std, y_train_std = dataset_to_jax_arrays(train_std)
    x_test_std,  y_test_std  = dataset_to_jax_arrays(test_std)

    print(f"Data ready in {time.time()-t0:.2f}s")

    results_colored: List[Dict] = []
    results_mnist: List[Dict] = []

    sweep_start = time.time()

    # ==========================================================
    # 1) Baseline: Standard MNIST (lambda = 0 => no constraint)
    # ==========================================================
    lamb0 = 0.0
    model = VIBClassifier(
        bottleneck_width=cfg.bottleneck_width,
        num_classes=cfg.num_classes,
    )

    print(f"\n--- MNIST Baseline (lamb={lamb0:.1e}) ---")
    run_config = asdict(cfg)
    run_config.update({"lambda": lamb0, "dataset": "mnist", "type": "baseline"})

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=experiment_group_id,
        name="mnist-baseline",
        config=run_config,
        reinit=True
    )

    t_start = time.time()
    res_std = run_train_eval(
        x_train_std, y_train_std, x_test_std, y_test_std,
        model, cfg, lamb0, wandb_run=run
    )
    res_std["run_time"] = time.time() - t_start
    res_std["dataset"] = "mnist"
    res_std["lambda"] = lamb0

    run.summary["final_test_acc"] = res_std["test_acc"]
    run.summary["final_train_acc"] = res_std["train_acc"]
    run.finish()

    results_mnist.append(res_std)
    all_summary_data.append(res_std)

    # ==========================================================
    # 2) Baseline: ColoredMNIST (lambda = 0 => no constraint)
    # ==========================================================
    model = VIBClassifier(
        bottleneck_width=cfg.bottleneck_width,
        num_classes=cfg.num_classes,
    )

    print(f"\n--- Colored Baseline (lamb={lamb0:.1e}) ---")
    run_config = asdict(cfg)
    run_config.update({"lambda": lamb0, "dataset": "colored_mnist", "type": "baseline"})

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=experiment_group_id,
        name="cmnist-baseline",
        config=run_config,
        reinit=True
    )

    t_start = time.time()
    res_col0 = run_train_eval(
        x_train_col, y_train_col, x_test_col, y_test_col,
        model, cfg, lamb0, wandb_run=run
    )
    res_col0["run_time"] = time.time() - t_start
    res_col0["dataset"] = "colored_mnist"
    res_col0["lambda"] = lamb0

    run.summary["final_test_acc"] = res_col0["test_acc"]
    run.summary["final_train_acc"] = res_col0["train_acc"]
    run.finish()

    results_colored.append(res_col0)
    all_summary_data.append(res_col0)

    # ==========================================================
    # 3) Sweep: ColoredMNIST lambda sweep (experiment)
    # ==========================================================
    for lamb in cfg.lambdas:
        lamb = float(lamb)
        model = VIBClassifier(
            bottleneck_width=cfg.bottleneck_width,
            num_classes=cfg.num_classes,
        )

        print(f"\n--- Colored Sweep (lamb={lamb:.1e}) ---")
        run_config = asdict(cfg)
        run_config.update({"lambda": lamb, "dataset": "colored_mnist", "type": "sweep"})

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=experiment_group_id,
            name=f"cmnist-lamb_{lamb:.1e}",
            config=run_config,
            reinit=True
        )

        t_start = time.time()
        res_col = run_train_eval(
            x_train_col, y_train_col, x_test_col, y_test_col,
            model, cfg, lamb,
            wandb_run=run
        )
        res_col["run_time"] = time.time() - t_start
        res_col["dataset"] = "colored_mnist"
        res_col["lambda"] = lamb

        run.summary["final_test_acc"] = res_col["test_acc"]
        run.summary["final_train_acc"] = res_col["train_acc"]
        run.finish()

        results_colored.append(res_col)
        all_summary_data.append(res_col)


    total_time = time.time() - sweep_start
    print("\n" + "#" * 70)
    print(f"Total sweep wall time: {total_time/60:.2f} minutes")
    print("#" * 70)

    # Prepare plots
    def to_xy(rs):
        x = np.array([r["lambda"] for r in rs])
        tr = np.array([r["train_acc"] for r in rs])
        te = np.array([r["test_acc"] for r in rs])
        return x, tr, te

    xs_c, tr_c, te_c = to_xy(results_colored)
    xs_m, tr_m, te_m = to_xy(results_mnist)


    # ---------------------------------------------------------
    # 3. The "Summary" Run
    # ---------------------------------------------------------
    summary_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=experiment_group_id,
        name="experiment-summary",
        job_type="summary"
    )

    wandb_summary_plot(all_data=all_summary_data, wandb_run=summary_run)
    
    # Log the raw data table too if you want to inspect values
    summary_run.log({"raw_data": wandb.Table(dataframe=pd.DataFrame(all_summary_data))})


    # Update Plot labels
    plt.figure()
    plt.scatter(xs_c, tr_c, label="ColoredMNIST train", marker="o")
    plt.scatter(xs_c, te_c, label="ColoredMNIST test", marker="x")
    if LOG_SWEEP:
        plt.xscale("symlog", linthresh=0.01) # Better for 0.0 values
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("ColoredMNIST accuracy vs Information capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("coloredmnist_sweep.png", dpi=300)

    plt.figure()
    plt.scatter(xs_m, tr_m, label="MNIST train", marker="o")
    plt.scatter(xs_m, te_m, label="MNIST test", marker="x")
    if LOG_SWEEP:
        plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Standard MNIST accuracy vs Information capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_sweep.png", dpi=300)

    print("Saved plots: coloredmnist_ib_sweep.png, mnist_ib_sweep.png")
    
    summary_run.finish()

    # Print a compact timing/accuracy table
    def pretty_print(results, name):
        # Sort by lambda
        results = sorted(results, key=lambda r: r["lambda"])
        print("\n" + name)
        print("Lambda\t\tRun Time\tTrain Acc.\tTest Acc.")
        for r in results:
            print(f"{r['lambda']:.3E}\t\t{r['run_time']:.2f}\t{r['train_acc']:.3f}\t\t{r['test_acc']:.3f}")

    pretty_print(results_colored, "ColoredMNIST results")
    pretty_print(results_mnist, "Standard MNIST results")


if __name__ == "__main__":
    main()
