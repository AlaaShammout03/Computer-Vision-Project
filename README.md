# Single-Image Crowd Counting for Public Gatherings Using Deep Learning

**Course:** COE486 - Computer Vision, Spring 2026  
**Institution:** American University of Sharjah  
**Project Context:** 3-person computer vision team project

This project re-implements **CSRNet** (Li et al., CVPR 2018) for single-image crowd counting and extends it with an expanded augmentation pipeline, a stable training recipe with density rescaling, and a per-density-bucket error analysis. Models are trained and evaluated on the **ShanghaiTech** dataset (Parts A and B).

## Final Results

| Split | Test MAE | Test RMSE |
|-------|---------:|----------:|
| ShanghaiTech Part A | 106.11 | 178.18 |
| ShanghaiTech Part B |  18.01 |  36.16 |

We outperform two of four published baselines and approach the original CSRNet paper's performance on sparse scenes (Part B Low-density bucket: 9.1% relative error).

## What's in this Repository

This project is organized into a few main folders. Here's what you'll find:

- **`CSRNet_Crowd_Counting_Project_Final.ipynb`** — The main notebook. Running this from top to bottom will train the model and produce all our results.

- **`models/`** — Contains `csrnet.py`, which is the CSRNet model architecture (VGG-16 frontend + dilated backend).

- **`utils/`** — Contains `dataset.py`, our custom PyTorch Dataset class. This is what loads the images and density maps, and applies our augmentations.

- **`results/`** — All the outputs from our experiments:
  - `comparison_table.csv` — Our final numbers compared to four published baselines.
  - `test_predictions_partA.csv` and `test_predictions_partB.csv` — Per-image predictions on the test set. Useful if you want to look at specific examples.
  - `error_by_density_partA.csv` and `error_by_density_partB.csv` — Our per-density-bucket error analysis (Low / Medium / High).
  - `plots/` — Training curve plots and the error analysis figures used in the report.
  - `predictions/` — Visualized predictions on selected best, median, and worst test images (input image + ground-truth density + predicted density side by side).

- **`requirements.txt`** — Python packages needed to run the project.

- **`README.md`** — The file you are reading right now.

## Requirements

- Python 3.10+
- NVIDIA GPU or colab T4 - CPU-only works but is much slower
- around 5 GB free disk space

## My Contributions

- Re-implemented and trained CSRNet for crowd counting using PyTorch.
- Prepared the ShanghaiTech dataset and generated density maps.
- Ran experiments on ShanghaiTech Part A and Part B.
- Applied density rescaling to improve training stability.
- Evaluated the model using MAE and RMSE.
- Contributed to per-density-bucket error analysis and result visualization.

## How To Run

The fastest way to reproduce our results is to open the notebook in Google Colab. No local setup required

1. Open `CSRNet_Crowd_Counting_Project_Final.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Switch runtime to GPU: **Runtime --> Change runtime type --> T4 GPU**
3. Mount your Google Drive
4. Run cells top-to-bottom

Total runtime end-to-end: approximately 2 hours

## Step-by-Step Instructions

### Setup

1. **Mount Google Drive** The project will be saved at `MyDrive/crowd-counting-csrnet/` by default

2. **Set the random seed** to 42 for reproducibility

3. **Install dependencies** Colab has most of these pre-installed; the cell uses `pip install -q` to handle the rest

### Data Preparation

4. **Download the ShanghaiTech dataset** The notebook downloads from a public Dropbox mirror and unzips into `data/raw/ShanghaiTech/`

5. **Generate density maps** For Part A we use the geometry-adaptive Gaussian (k-NN-based per-head sigma); for Part B we use a fixed sigma of 15. Density maps are computed on Colab's local SSD for speed and synced to Drive when complete. This step takes about 3 minutes total

### Training

6. **Choose which part to train** Set the `PART` variable:
   ```python
   PART = "part_B_final"   # train Part B first (easier, faster)
   ```
   or
   ```python
   PART = "part_A_final"   # train Part A
   ```

7. **Build train/val/test loaders** The training set is split 80/20 train/val with a fixed random seed. The test set is held out and only used after training

8. **Set up the model and optimizer**:
   - CSRNet (VGG-16 frontend pretrained on ImageNet + dilated backend)
   - Adam optimizer, learning rate 1e-5, weight decay 5e-4
   - StepLR scheduler (decay by 0.5 every 30 epochs)
   - Gradient clipping at L2 norm = 1.0
   - Density rescaling factor = 100 for stable training

9. **Run the sanity check cell** before main training: It performs 5 mini-training steps and reports the loss magnitude. **If the loss is below 0.5, stop and debug** — do not waste 50 minutes on broken training.

10. **Train** The training cell:
    - Trains for 80 epochs
    - Evaluates on the validation set every epoch
    - Saves the best checkpoint to `checkpoints/<part>/csrnet_best_<part>.pth` whenever validation MAE improves
    - Prints loss, val MAE/RMSE, learning rate, and epoch time per iteration

    Training time: around 50 minutes per part on a T4 GPU

11. **Repeat for the other part.** Set `PART` to the other value and re-run cells in Section 8

### Testing / Evaluation

12. **Load the best checkpoint** The notebook loads `csrnet_best_<part>.pth` and reports final test MAE and RMSE on the held-out test set

13. **Generate per-image predictions** Saved to `results/test_predictions_<part>.csv` for each image: predicted count, actual count, absolute error, relative error percentage

14. **Compute the per-density-bucket analysis** Test images are grouped into Low (`<50`), Medium (`50–200`), and High (`>= 200`) crowd-density buckets. MAE, RMSE, and mean relative error are reported per bucket. Saved to `results/error_by_density_<part>.csv`.

15. **Plot training curves and error analysis**:
    - `results/plots/training_curves_<part>.png` - loss, val MAE/RMSE, LR over epochs
    - `results/plots/error_analysis_<part>.png` - predicted-vs-actual scatter and per-bucket bar chart

16. **Save qualitative examples** Best, median, and worst predictions are saved as side-by-side image / GT density / predicted density plots in `results/predictions/`.

### Final Comparison Table

17. **Run the comparison table cell** after both parts are trained. It loads both checkpoints, evaluates on both test sets, and produces a table comparing our results against four published baselines (MCNN, FCN, SaCNN, CSRNet paper). Saved to `results/comparison_table.csv`.

## Demo: Predicting on Custom Images

Use the demo cell in section 11 to run inference on any image:

```python
predict_count("/content/your_image.jpg", part_tag="partA")
```

This loads the trained model, predicts the count, and displays the input image alongside the predicted density map.

## Reproducibility

All randomness is seeded:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The 80/20 train/validation split uses `sklearn.train_test_split` with `random_state=SEED`. Re-running the notebook should produce numerically identical results, modulo small CUDA non-determinism in convolution kernels.

## Common Issues we Faced During Running

**`File not found` errors during density-map generation.** The notebook expects the dataset at `data/raw/ShanghaiTech/part_A_final/` and `data/raw/ShanghaiTech/part_B_final/`. If your folder structure differs, edit the paths in Section 3.

**Loss is around 1e-4 and val MAE is not improving.** The density rescaling factor isn't being applied. Verify the dataset class has the line `density = density * self.density_scale`.

**`CUDA out of memory`.** Reduce batch size to 2 or 1 in the data-loaders cell.

**Colab disconnects mid-training.** The best checkpoint is saved to Drive every time validation MAE improves, so you don't lose everything. Reconnect, re-run setup cells, and load the saved checkpoint with `torch.load(...)`.

## References

- **CSRNet (paper):** Li, Y., Zhang, X., & Chen, D. (2018). *CSRNet: Dilated convolutional neural networks for understanding highly congested scenes.* CVPR 2018.
- **ShanghaiTech dataset:** Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). *Single-image crowd counting via multi-column convolutional neural network.* CVPR 2016.
