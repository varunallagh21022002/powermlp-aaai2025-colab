# PowerMLP in Google Colab

An independent PyTorch implementation of **PowerMLP: An Efficient Version of KAN** (Qiu et al., AAAI 2025), with a runnable notebook, measured experiments, presentation, and 20 viva questions with answers.

Paper: https://doi.org/10.1609/aaai.v39i19.34210  
Official source: https://github.com/Reed-qiuu/PowerMLP

## Run in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varunallagh21022002/powermlp-aaai2025-colab/blob/main/notebooks/PowerMLP_Colab.ipynb)

Open `notebooks/PowerMLP_Colab.ipynb` in https://colab.research.google.com using File → Upload notebook, then Runtime → Run all. The notebook embeds all required source files. It needs no dataset download or API key. CPU is sufficient. GPU is optional: choose a GPU runtime and set `DEVICE='cuda'`.

The saved notebook outputs are from a local CPU execution, not a claimed Google-hosted execution. Rerunning in Colab generates results for that runtime. Download the results ZIP at the end before disconnecting.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment --steps 1000 --seeds 0 1 2 --size large --out results
```

Use `--size small` for the paper's approximately 25-parameter comparison. Use `--device cuda` for CUDA. `requirements.txt` gives compatible ranges; `results/results.json` records versions used for the supplied measurements. `requirements-tested.txt` records the local environment and may need platform adjustments elsewhere.

## Method

Each hidden layer computes `alpha(SiLU(x)) + relu(Wx+b)**k`, with a bias-free alpha projection. The final layer is affine. Default k=3. The large shapes match the paper's section 5.1 parameter budgets: PowerMLP [2,32,8,1] with 689 parameters, MLP [2,32,18,1] with 709, and KAN [2,2,1,1] with G=100, k=3, and 735.

The experiment uses the section 5.4 target `x*exp(-y)`. Our protocol samples uniform inputs in [-1,1]^2, uses 1,024 train / 256 validation / 1,024 test examples, trains with full-batch Adam (learning rate 0.001), and selects the checkpoint with lowest validation MSE every 25 updates. Tests evaluate only the selected checkpoint. Seeds 0,1,2 have independent splits, shared across models within each seed. Ablations retain widths and therefore have different capacity when the basis branch is removed.

## What this reproduces

- The PowerMLP architecture in Eq. (6), including the basis branch and linear output.
- The small and large parameter budgets used in the paper.
- A reduced experiment on the paper's ablation target, with baselines and component removals.

## What it does not reproduce

This is **not a full reproduction of the paper's benchmark tables**. Our data split, initialization, optimizer and training budget are project choices. Our KAN implements Eqs. (1)-(4) with a fixed uniform grid and vectorized basis evaluation. It does not use pykan 0.2.5, adaptive grids, pruning, or symbolic operations. The timing ratio is therefore not a reproduction of the paper's roughly 40× claim. The full 16-function suite, knot-theory experiments, classification suite, and complete depth/width sweeps remain outside this project.

## Files

- `notebooks/PowerMLP_Colab.ipynb`: standalone teaching notebook with recorded execution.
- `src/models.py`: PowerMLP, MLP, and fixed-grid KAN.
- `src/experiment.py`: repeatable training, evaluation, timing and plots.
- `tests/test_models.py`: equation, gradient, parameter-count, spline and backward checks.
- `results/`: genuine per-seed metrics, summaries, plots and first-seed model checkpoints.
- `output/PowerMLP_Presentation.pptx`: editable presentation with speaker notes.
- `VIVA_20_Questions_and_Answers.md`: exactly 20 questions and short answers.

## Timing interpretation

Training seconds include validation and best-state copying. The separate `step_ms` measurement averages 100 forward/backward passes on a batch of 256 after 10 warmups, without optimizer updates. CUDA measurements synchronize before and after timing. Error bars show sample standard deviation across seeds, not confidence intervals. Equal updates do not imply equal convergence, and shared learning rates may favor one model.

## Attribution

This educational implementation was written independently from the paper's mathematical specification and checked against the authors' public implementation. It is not the authors' official repository. The original paper and third-party libraries retain their respective rights. The paper PDF is not redistributed in this repository.

```bibtex
@article{qiu2025powermlp,
  title={PowerMLP: An Efficient Version of KAN},
  author={Qiu, Ruichen and Miao, Yibo and Wang, Shiwen and Zhu, Yifan and Yu, Lijia and Gao, Xiao-Shan},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={19}, pages={20069--20076}, year={2025},
  doi={10.1609/aaai.v39i19.34210}
}
```
