# Driving policy: behaviour cloning (DAgger) from pure pursuit, then PPO

The shipped policy `checkpoints/ab_policy.pt` (with `checkpoints/ab_policy.npz`, the same weights
and normaliser as plain numpy arrays) drives A->B and closed loops with 100% success on fresh
seeds. It was trained by imitating the hand-coded pure-pursuit controller (`drivesim/baselines.py`)
with DAgger. PPO alone (`scripts/train.py`) did not reach that level in the budget tried; see
"What did not work".

## Quick start

```bash
# 1. imitation warm start (about 10 min on 4 cores; writes a train.py-compatible checkpoint)
python scripts/bc_pretrain.py --out runs/bc/bc.pt --episodes 300 --dagger-iters 4 \
    --episodes-per-iter 150 --workers 4 --epochs 20 --hidden 128 --closed-frac 0.35

# 2. evaluate on 20 fresh seeds (seeds 900M+ never appear in training; training seeds start at 100M)
python scripts/eval.py --ckpt runs/bc/bc.pt --task ab --difficulty 0.5 --episodes 20 --workers 4
python scripts/eval.py --ckpt runs/bc/bc.pt --task loop --difficulty 0.5 --episodes 10 --workers 4
python scripts/eval.py --baseline pure_pursuit --task ab --difficulty 0.5 --episodes 20 --workers 4

# 3. optional PPO fine-tune from the warm start (lower exploration noise, slow lr)
python scripts/train.py --task ab --init-from runs/bc/bc.pt --init-log-std -2.0 --lr 1e-4 \
    --num-envs 3 --rollout-len 256 --total-steps 300000 --pool 0 --save-dir runs/ppo_ft
```

Export to numpy (no torch or MLX needed to run the policy):

```python
import pickle, numpy as np
from drivesim.rl.ppo import export_npz, numpy_mlp
export_npz(pickle.load(open("runs/bc/bc.pt", "rb")), "runs/bc/bc.npz")
```

`numpy_mlp(weights, "actor", x, n_layers=2)` then gives the mean action. Normalise the observation
first: `clip((obs - obs_mean) / sqrt(obs_var + 1e-8), -10, 10)`, then clip the action to
`[-1, 0, 0] .. [1, 1, 1]` (steer, throttle, brake).

## Files

| path | purpose |
|---|---|
| `scripts/bc_pretrain.py` | DAgger from `PurePursuitController`; writes a PPO-format checkpoint |
| `scripts/train.py` | PPO; `--init-from` warm-starts weights and observation normaliser |
| `scripts/eval.py` | deterministic evaluation on fresh seeds, or `--baseline pure_pursuit` |
| `drivesim/rl/ppo.py` | PPO core, numpy policy, `export_npz` |
| `checkpoints/ab_policy.pt` | shipped policy (pickle, about 200 kB) |
| `checkpoints/ab_policy.npz` | same weights as numpy arrays |

## Behaviour cloning details (`scripts/bc_pretrain.py`)

* Expert: `PurePursuitController` with continuous actions `(steer, throttle, brake)`. The
  observation holds everything the controller uses (lookahead points, curvature ahead, speed,
  lateral offset), so the MLP can represent it.
* Iteration 0 rolls out the expert only. Each later iteration rolls out the learner with
  probability `1 - beta` per step (`beta` = 0.5^i), labels every visited state with the expert
  action, and aggregates the data. Four DAgger iterations were used; no run with more iterations was tried.
* Actor: `Linear-tanh-Linear-tanh-Linear` (`--hidden 128 --layers 2`), trained with MSE on the
  mean output. The observation normaliser is fixed from iteration-0 data. The saved `log_std` is
  `--log-std` (default -2.0), so PPO fine-tuning starts near deterministic.
* `--closed-frac 0.35` mixes in closed loops (3000 steps); the observation has a closed/open flag,
  so one policy drives both tasks.
* Each iteration saves the checkpoint, so an interrupted run keeps its progress.

Run times on this 4-core box: 300 expert episodes plus 4 x 150 DAgger episodes took about 10 min;
the 890k-sample dataset is trained for 20 epochs per iteration, batch 512, Adam lr 1e-3.

## PPO (`scripts/train.py`)

Compact PPO, Gaussian policy with state-independent log-std, separate actor and critic MLPs,
GAE, reward normalisation, observation normalisation, minibatch epochs. Key flags:

* batch size = `--num-envs x --rollout-len` (the default 8 x 256 = 2048). Raise `--num-envs` to
  about cores minus one for speed. Each env is one subprocess.
* `--pool 0` gives a fresh random track on every reset (use this for generalisation).
* `--init-bias 0,0.5,-1.5 --init-log-std -1.0` avoids the crawl at start (the default brakes and
  throttles at once).
* `--init-from ckpt` warm-starts weights and the observation normaliser. The `--init-log-std`
  flag then overrides the checkpoint's log-std only if given.
* `--reward key=value ...` overrides `RewardConfig` weights (e.g. `--reward time=0.02`).

## Results (20 fresh seeds unless noted, deterministic policy)

Seeds are `900,000,000 + i`. Success means reaching the goal (A->B) or completing one lap (loop).

| policy | AB 0.0 | AB 0.5 | AB 1.0 | loop 0.5 |
|---|---|---|---|---|
| pure pursuit (baseline) | 15/20 (75%) | 19/20 (95%) | 19/20 (95%) | 10/10 (100%) |
| **BC+DAgger, `ab_policy.pt`** | **20/20 (100%)** | **20/20 (100%)** | **20/20 (100%)** | **10/10 (100%)** |
| BC+DAgger, open tracks only (`bc1`) | 20/20 | 20/20 (40 more on seeds 910M+: 39/40) | 19/20 (95%) | not run |
| PPO run B (random tracks, 400k steps, `--pool 0`) | not run | 5/20 (25%) | not run | not run |
| PPO run A (fixed pool of 32 tracks, 150k steps) | - | 15-20% | 0% | - |

Loop results for the shipped policy at other difficulties: 10/10 at 0.0 and 10/10 at 1.0 (average
return 181 and 159; the baseline was not run there). On the shipped policy's AB seed block 910M+
(40 episodes, difficulty 0.5), success is 39/40 with one off-track failure.

Steps / updates / wall-clock to solve:

* BC+DAgger (`ab_policy.pt`): 300 expert episodes (about 276k samples) plus 4 DAgger iterations
  (about 870k samples in total; each iteration retrains from the previous weights). Wall-clock about 10 min on 4 cores.
  The learner's success in the DAgger rollouts of the closed-loop-mixed run (`bc2`) was 0.97 (expert only),
  0.95, 0.99, 0.83, 0.87 per iteration; the final iteration (beta 0.06) reached 0.87 overall and 0.90 on closed loops.
* PPO run B: 400k env steps, 391 updates, 17 min, reached 97% on training-distribution random tracks
  at difficulty 0 but only 25% at difficulty 0.5 on fresh seeds (distribution shift in difficulty).
  The learning curve is in `scratchpad/runs/expB/learning_curves.png` (not committed). Success
  rises from 0 to 0.97 over about 0.18M steps and then flattens. The clip fraction stays around
  0.1 and the entropy falls from 1.2 to -0.3. Those are signs of a policy that became deterministic
  on the training distribution without learning the harder ones.

## What did not work

* PPO from scratch, even with fresh random tracks, 400k steps, and a 1024-sample batch. The policy
  learns to reach the goal on easy tracks but generalises poorly to difficulty 0.5. Run A (fixed
  pool of 32 tracks) overfit the pool, and its failures were mostly `stuck`.
* Plain BC (iteration 0 only) was not evaluated on its own, so the gain from DAgger iterations is not measured here.

## Known issues / untested

* PPO fine-tune from the BC checkpoint (`--init-from`) is implemented but was not run to
  completion, so its numbers are not in the table. Treat it as untested.
* The MLX backend (`--backend mlx`, `pip install "mlx[cpu]"`) was not tested in this environment.
  The Apple GPU path is untested.
* `scripts/eval.py` loads `.pt`/pickle checkpoints only. Use `export_npz` plus `numpy_mlp` for
  the numpy path.
* The "stuck" failure traces of PPO run A were not analysed in detail.
* The BC script's obs normaliser is fixed from the iteration-0 expert data. Later PPO fine-tuning
  updates it, so fine-tuning changes the input scaling.
