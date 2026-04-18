# Tredence: AI and Data Analytics Intelligence.

# Self-Pruning Neural Network - Tredence Case Study

Rubric-first CIFAR-10 training script with an optional `--enhanced` mode for extra engineering tricks.

## Project Structure

```text
self_pruning_network/
  train.py
  REPORT.md
  README.md
  requirements.txt
  results_summary.json          # generated after training
  plots/                        # wiped and regenerated each run
    gate_hist_lambda_0.0001.png
    gate_hist_lambda_0.001.png
    gate_hist_lambda_0.01.png
    gate_heatmap_best.png
    lambda_tradeoff.png
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (rubric default)

```bash
python train.py
```

Defaults:

- Exact backbone: `PrunableLinear(3072,512) -> ReLU -> PrunableLinear(512,256) -> ReLU -> PrunableLinear(256,10)`
- `get_sparsity_loss()` uses `sum(sigmoid(gate_scores))` across all `PrunableLinear` layers
- λ sweep: `1e-4, 1e-3, 1e-2`
- Epochs per λ: **40** (tuned so the rubric’s hard threshold sparsity metric is measurable in typical CPU wall-clock time)

### Optional: enhanced engineering mode

```bash
python train.py --enhanced
```

`--enhanced` enables non-rubric upgrades (augmentation, BN/GELU/Dropout backbone, split validation, etc.). The rubric `get_sparsity_loss()` definition remains `sum(sigmoid(gate_scores))`.

### Logging tips

If you redirect output to a file, prefer unbuffered logs:

```bash
PYTHONUNBUFFERED=1 python -u train.py > train_run.log 2>&1
```

### Other useful flags

```bash
python train.py --epochs 40 --batch-size 128 --seed 42 --plots-dir plots
python train.py --progress   # verbose tqdm (can be very chatty)
```

## What the objective is

The model uses `PrunableLinear` gates \(g=\sigma(s)\) multiplied into weights before `F.linear`.

Training minimizes:

`CrossEntropy + lambda * sum(sigmoid(gate_scores))`

Sparsity reporting uses the rubric metric:

`% of gates with sigmoid(gate_scores) < 0.01`
