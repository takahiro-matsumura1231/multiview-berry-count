# multiview-berry-count

Multi-view berry counting for grape bunches. The input is a short handheld video that orbits a
bunch, and the output is an estimate of the number of berries on it.

**Model.** The video is sampled into K=9 views that follow the camera's orbit. Each view is
encoded as one token: a 512-d feature from an ImageNet-initialised ResNet-18 fine-tuned on
grape frames, concatenated with 26 berry-detection features from a YOLOv8 detector (berry
statistics plus one bunch-enclosing box). A Transformer with view-position encoding aggregates
the 9 tokens and regresses the count. Each video is predicted from 5 fixed view-sets, and the
reported value is their median.

## Layout

```
configs/dataset.yaml   frame extraction and admission rules (fps, head trim, image size)
configs/split.yaml     nested bunch-level cross-validation (5 outer x 4 inner folds)
configs/model.yaml     model and training hyper-parameters
src/grapecount/        library: data build, detection features, splits, models, training
scripts/train.py       nested-CV training; saves a reloadable checkpoint per seed x fold
scripts/evaluate.py    MAE / RMSE / counting fidelity with bunch-bootstrap intervals
scripts/predict.py     count berries in your own videos with trained checkpoints
tests/                 unit tests (pytest)
```

The repository ships no videos, detector weights or trained checkpoints: the dataset and the
trained models of the paper are not publicly released. By default, inputs and
outputs live under `data/`, `models/` and `results/`, and all three are git-ignored.

## Setup

```bash
conda create -n berrycount python=3.10 -y && conda activate berrycount
pip install -r requirements.txt   # pinned versions used for the paper (see the file for CUDA builds)
pip install -e . --no-deps
sudo apt install ffmpeg      # ffmpeg / ffprobe extract the frames
export PYTHONPATH=src
```

The berry detector is a YOLOv8 model exported to TorchScript. Put it at
`models/yolov8.torchscript`, or point to it with `export GRAPECOUNT_DETECTOR=/path/to/model.torchscript`.

## Data format

Place the videos (`.mp4` / `.mov`) under one directory. Put each capture session in its own
sub-directory; the split is stratified by session, which is the top-level directory name
(a leading `berry_` and everything after the first `_` are dropped). Training videos must be
named

```
<bunch>_<stage>_gt_<count>.mp4        e.g.  B1_0_gt_60.mp4,  B1_1_gt_54.mp4
```

- `<bunch>` is letters followed by digits. It identifies the physical bunch, and every video of
  a bunch is kept in the same fold.
- `<stage>` is an integer that distinguishes the videos of one bunch, for example successive
  thinning stages.
- `<count>` is the ground-truth berry count.

The first 2 s of every video are discarded, because the opening frames may show a board with
the bunch ID and count. Frames are then taken at 4 fps. A video that leaves fewer frames than
`admission.min_frames` in `configs/dataset.yaml` is skipped. Skipped videos and their reasons
are listed in `<dataset>/manifests/excluded.json` and `audit.json`.

## Training on your dataset

```bash
RAW=/path/to/videos          # your video directory
DS=data/dataset_v2           # where frames and features are written

python -m grapecount.data.build    --raw-root $RAW --out $DS --workers 16
python -m grapecount.data.verify   --raw-root $RAW --out $DS --all-images
python -m grapecount.splits        --dataset-root $DS
python -m grapecount.features.detection    --dataset-root $DS   # GPU; for several GPUs run one
                                       # process per GPU: CUDA_VISIBLE_DEVICES=i ... --shard i --n-shards N
python -m grapecount.features.det_features --dataset-root $DS

python scripts/train.py    --dataset-root $DS --out results/model   # 3 seeds x 5 outer folds
python scripts/evaluate.py --runs results/model/runs
```

`scripts/train.py` writes one unit per seed and outer fold under `results/model/runs/seed{s}/outer{o}/`.
Each unit holds `checkpoint.pt`, the outer-test `predictions.tsv`, the per-epoch
`inner_predictions.tsv`, and `run_manifest.json`, which is written last. Before a unit is
published, its checkpoint is re-read in a separate process and re-predicts one video. To run
only part of the grid, use `--seeds 0 --outer 0`. Completed units are skipped when the script
is run again. A run can only be resumed with the same config and dataset, which is checked by
hash; after changing either, use a new `--out`.

## Predicting on new videos

```bash
# labeled names (<bunch>_<stage>_gt_<count>.mp4): MAE / RMSE are also printed
python scripts/predict.py --videos /path/to/videos --checkpoints results/model/runs --out pred.csv

# any file names, no ground truth
python scripts/predict.py --videos /path/to/videos --unlabeled \
    --checkpoints results/model/runs --out pred.csv
```

`predict.py` runs the same steps as training (frame extraction, detection, features) inside
a work directory (`pred.work/` for `--out pred.csv`). It rebuilds that directory whenever the
input videos change. Skipped videos are printed with the reason, for example videos too
short to leave 16 frames after the 2 s head trim. Every checkpoint it finds then predicts every video, and the CSV reports the
mean over the checkpoints (`pred`) together with their range.

## Evaluation protocol

- Splits are made at the bunch level, so a bunch never appears in both training and test.
- The training epoch is chosen on the inner folds only. Each outer test fold is scored once,
  after the model has been refit on the whole outer-train set.
- Feature standardisation statistics come from the training bunches of each fit.
- Intervals are percentile bootstraps that resample whole bunches.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Tests that need a built dataset are skipped when `data/dataset_v2` is absent.

## Citation

If you use this code, please cite the accompanying paper. The reference will be added here
once the paper is published.

## License

MIT. See [LICENSE](LICENSE).
