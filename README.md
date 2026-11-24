# TIW-DSM reproduction: unbiased diffusion models from a biased dataset

Reproduction and deployment of **Kim et al., "Training Unbiased Diffusion Models From Biased Dataset", ICLR 2024**
on **CIFAR-10 LT with a 5% reference set** (D_bias: 10,000 long-tailed images, 100:1 imbalance; D_ref: 500 unbiased images).

Group: Syed Zain Afaq (36234), Shaheer Ahmed Farooqui (19722), Huzaifa (19193). ML Fall 2026, Deliverable 3.

| Artifact | Link |
|---|---|
| Code (this repo) | https://github.com/shaheer-geeksofkolachi/tiw-dsm-repro |
| Trained weights (`tiw_dsm.pt`, `dsm_obs.pt`, EMA at 3.51 Mimg) | In this repo: [`checkpoints/deploy/`](https://github.com/shaheer-geeksofkolachi/tiw-dsm-repro/tree/main/checkpoints/deploy). Also in the Kaggle output of tiw-dsm-eval (with `discriminator.pt`): https://www.kaggle.com/code/shaheerfarooqui19722/tiw-dsm-eval/output |
| Live app (Streamlit Cloud) | https://tiwdsm-cifar10-demo.streamlit.app |
| Final report (PDF) | `report/main.pdf` |

> **Freeze rule:** the last commit and the cloud files must be dated before **25 Nov 2026, 23:55**.

---

## 1. Setup

```bash
git clone https://github.com/shaheer-geeksofkolachi/tiw-dsm-repro tiw-dsm-repro && cd tiw-dsm-repro
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # install a CUDA build of torch first if you have a GPU
```
Python 3.10+. Training was run on one free Kaggle GPU (T4 or P100, 16 GB). Inference and the app also run on CPU.

## 2. Data and pretrained helpers (about 5 minutes)

```bash
python scripts/prepare_data.py
```
This downloads the authors' `fake_data.npz` / `real_data.npz`, plus CIFAR-10 train and test (Hugging Face), and
recovers class labels by exact SHA-1 matching (used for analysis only, **never in training**). It also fetches the
authors' ADM feature extractor `32x32_classifier.pt` and the CIFAR-10 VGG13-BN used for the bias metric.

* If the feature-extractor download fails, get `discriminator/feature_extractor/32x32_classifier.pt` from the authors'
  [checkpoint folder](https://drive.google.com/drive/folders/1vYLH8UNlXWZarn0IOtiPuU8FvBFqJvTP) and place it at
  `checkpoints/feature_extractor/32x32_classifier.pt`.
* If `torch.hub` cannot fetch VGG13-BN, run `python scripts/train_eval_classifier.py` (about 15 min on GPU).

## 3. Experiments

```bash
# 0) 2-D sanity check of the implementation (CPU, about 15 min)
python scripts/toy_2d.py

# 1) Time-dependent discriminator (about 30-60 min on T4)
python scripts/train_discriminator.py --iters 10000

# 2) Score models. Same budget for both; resumable across 12-h sessions (just re-run the command)
python scripts/train_score.py --mode obs --outdir runs/obs --duration_mimg 11
python scripts/train_score.py --mode tiw --outdir runs/tiw --duration_mimg 11
#    optional baselines/ablations: --mode iw | --mode ref | --mode tiw --no_correction | --mode tiw --no_reweight

# 3) Main evaluation (FID-50k, precision/recall, class-balance metrics)
python scripts/evaluate.py --ckpt runs/obs/ema_final.pt --name obs --num 50000
python scripts/evaluate.py --ckpt runs/tiw/ema_final.pt --name tiw --num 50000
for r in bias ref target; do python scripts/evaluate.py --real $r --name data_$r; done

# 4) Additional experiment: sampling steps / solver vs quality, bias and time
python scripts/steps_experiment.py --ckpt obs=runs/obs/ema_final.pt --ckpt tiw=runs/tiw/ema_final.pt --num 10000

# 5) Report tables and figures from the JSON results, then compile
python scripts/make_report_assets.py --runs obs=runs/obs tiw=runs/tiw
cd report && latexmk -pdf main.tex
```
**Short of time?** Use `--duration_mimg 4`, or `--batch 128` if you run out of memory. Use the **same** value for obs and tiw, and
record it in the report's configuration table. `notebooks/run_on_kaggle.ipynb` runs everything step by step.

## 4. Deployment: Streamlit app

```bash
mkdir -p checkpoints/deploy
cp runs/tiw/ema_final.pt checkpoints/deploy/tiw_dsm.pt
cp runs/obs/ema_final.pt checkpoints/deploy/dsm_obs.pt      # optional, enables side-by-side comparison
streamlit run app/streamlit_app.py                           # opens http://localhost:8501
```
* **Input:** a seed (any integer, e.g. one the examiner gives), number of images, solver and steps. Or upload any image
  for image-to-image (SDEdit).
* **Preprocessing:** seeded N(0, I) noise scaled to sigma_max = 80. For an upload: centre-crop, resize to 32x32, scale to [-1, 1],
  add noise at the chosen strength.
* **Model:** the EDM-preconditioned U-Net trained with TIW-DSM, solved with a deterministic Heun/Euler ODE.
  The discriminator is **not** needed at inference.
* **Output:** 32x32 RGB images (grid, downloadable PNG), generation time, NFE, checkpoint SHA-256, the class predicted
  by VGG13-BN for each image and a class histogram.

## 5. Repository map and code attribution

| Path | What | Origin |
|---|---|---|
| `tiwdsm/data.py` | data loading, D_obs construction, label recovery | **team** |
| `tiwdsm/unet.py` | compact DDPM++-style U-Net | **team** (design follows DDPM++ / EDM) |
| `tiwdsm/edm.py` | EDM preconditioning, sigma sampling, Heun/Euler samplers, EMA | **adapted** from EDM (Karras et al.) formulas |
| `tiwdsm/vpsde.py` | sigma to VP-time conversion, importance time sampling | **adapted** from official `classifier_lib.vpsde` |
| `tiwdsm/discriminator.py` | time-dependent discriminator, 2d weight, grad log d | **team**, architecture as in official code |
| `tiwdsm/losses.py` | DSM(obs/ref), IW-DSM, TIW-DSM (+ ablations) | **team** (from paper Eq. 10, 36, Alg. 2) |
| `tiwdsm/metrics.py` | FID, precision/recall, Eq. 51 bias, class shares | **team** (Inception net from `pytorch-fid`) |
| `scripts/*` | training / evaluation / experiment / report scripts | **team** |
| `app/streamlit_app.py` | deployment app | **team** |
| `third_party/guided_diffusion/` | ADM encoder used by the discriminator | **reused unchanged** (OpenAI, via official TIW-DSM repo) |
| `32x32_classifier.pt` (downloaded) | pretrained ADM feature extractor | **reused** (authors' release) |
| VGG13-BN CIFAR-10 (downloaded) | classifier for the bias metric | **reused** (`chenyaofo/pytorch-cifar-models`) |

We did not run the official training scripts. All training and evaluation used the code above.
