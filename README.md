--------------------------------------------------------------------------------
Multispectral UAV Data of Jujube Orchard dataset (~170 GB, 15,414 files) is
hosted on Kaggle under CC BY 4.0:
https://www.kaggle.com/datasets/xuyalei/jujube-uav-multispectral-20251107
--------------------------------------------------------------------------------

This dataset contains raw UAV RGB and multispectral imagery of jujube orchards
in southern Xinjiang, China, acquired with a DJI Mavic 3 M on 2025-11-06,
11-07, 11-08, 11-14 and 11-17. It was collected for RGB-to-multispectral
reconstruction research.

Site       : 224th Regiment, Kunyu City, 14th Division, XPCC, Xinjiang, China
Platform   : DJI Mavic 3 M (RGB + 4 multispectral cameras, PPK)
Bands      : G 560 +/- 16 nm, R 650 +/- 16 nm, RE 730 +/- 16 nm, NIR 860 +/- 26 nm
RGB images : 5280 x 3956 px, 8-bit JPEG
MS images  : 2592 x 1944 px, 16-bit TIFF, one file per band
Capture set: 1 RGB JPG + 4 MS TIFF of the same instant, 3,049 sets in total
Total size : ~170.3 GB, 15,423 files

Note: the data are raw. RGB and multispectral images are not spatially aligned,
no radiometric calibration, mosaicking or annotation has been applied.

--------------------------------------------------------------------------------
Directory structure
--------------------------------------------------------------------------------
11.6_multispectral video\     video only, 5 flights (18 MP4 + 18 SRT telemetry)
11.7_multispectral image\     764 capture sets, 3 flights, 2025-11-07
11.8_multispectral image\     896 capture sets, 4 flights, 2025-11-08
UAV flight route map\         1,389 capture sets, 6 waypoint routes

File naming (example):
  DJI_20251107113232_0001_D.JPG       RGB image
  DJI_20251107113232_0001_MS_G.TIF    Green band
  DJI_20251107113232_0001_MS_R.TIF    Red band
  DJI_20251107113232_0001_MS_RE.TIF   Red edge band
  DJI_20251107113232_0001_MS_NIR.TIF  Near-infrared band

PPK files (*_PPKNAV.nav, *_PPKOBS.obs, *_PPKRAW.bin, *_Timestamp.MRK) are
included for each flight.

--------------------------------------------------------------------------------
Citation
--------------------------------------------------------------------------------
Yalei Xu, Jingbin Li, Zhiyuan Zhang, Hongfei Yang, Yang Li, Huizhe Ding.
RGB-multispectral image registration and generative spectral reconstruction for
UAV-based jujube orchard monitoring. Computers and Electronics in Agriculture,
2026, article 112519. https://doi.org/10.1016/j.compag.2026.112519

BibTeX:

@article{xu2026rgb,
  title     = {RGB--multispectral image registration and generative spectral
               reconstruction for UAV-based jujube orchard monitoring},
  author    = {Xu, Yalei and Li, Jingbin and Zhang, Zhiyuan and Yang, Hongfei
               and Li, Yang and Ding, Huizhe},
  journal   = {Computers and Electronics in Agriculture},
  pages     = {112519},
  year      = {2026},
  publisher = {Elsevier}
}


--------------------------------------------------------------------------------
Dataset layout

Dataset/
├── source/             # 3-channel RGB jpg (condition / hint)
├── target/             # 4-band tif, band order G, R, RE, NIR
├── prompt.json         # training pairs
└── prompt_test.json    # validation pairs
```

- File names must share a base ID: `DJI_20251108162420_0144.jpg` ↔ `DJI_20251108162420_0144.tif`.
- Any resolution is fine (loader resizes to 512×512); 16-bit tif is converted to 8-bit.
- `prompt.json`, one JSON per line:

```json
{"source": "source/DJI_20251108162420_0144.jpg", "target": "target/DJI_20251108162420_0144.tif", "prompt": "DJI Mavic 3M"}
```

- Generate the JSON files:

```bash
python generate_prompt_json.py --source_dir /path/dataset/source \
    --target_dir /path/dataset/target --output /path/dataset/prompt.json
```

- Change the dataset root and JSON name in `tutorial_dataset.py` (lines 49, 69, 71), and the test path in `val.py` (line 214).

--------------------------------------------------------------------------------

## 1. tutorial_train_VAE.py — train the VAE

```bash
python tutorial_train_VAE.py
```

Set at the top of the file:

```python
resume_path   = 'vae_4ch_epoch10.pth'   # starting VAE weights
batch_size    = 1
learning_rate = 1e-7
total_epochs  = 500
save_every    = 10
kl_weight     = 0.001
```

Trains `first_stage_model` only, loss = MSE + KL. Saves `vae_4ch_kl_epoch{N}.pth` every 10 epochs in the current directory.

--------------------------------------------------------------------------------

## 2. tutorial_train.py — train ControlNet

```bash
python tutorial_train.py
```

Set at the top of the file:

```python
resume_path       = 'lightning_logs/version_53/checkpoints/epoch=499-step=39499.ckpt'
vae_weights_path  = 'vae_4ch_epoch160.pth'   # VAE from step 1
batch_size        = 1
learning_rate     = 1e-5
sd_locked         = True
devices           = 3
max_epochs        = 300
```

Loads the ControlNet checkpoint (skipping VAE / text encoder keys), then loads the VAE strictly and freezes it. Only `control_model` is trained. Images are logged to `lightning_logs/version_N/image_log/train/` every 600 batches.

--------------------------------------------------------------------------------

## 3. val.py — evaluate

```bash
python val.py
python val.py --dataset test --noise_level medium --color_cast_level slight
```

Flags: `--dataset train|test`, `--noise_level none|slight|medium|heavy`, `--color_cast_level none|slight|medium|heavy`.

Set inside the file: `controlnet_ckpt`, `vae_weights_path`, `batch_size`, `output_dir` (`val_outputs/`).

Outputs: 4-band tif, RGB png, and per-band files under `target_bands/` and `generated_bands/`. Prints RMSE, MRAE, PSNR, SSIM per band and overall SAM.

Note: `pred = images['reconstruction']` scores the VAE reconstruction, not the diffusion output. Use `images['samples_cfg_scale_9.00']` to score ControlNet generation.

--------------------------------------------------------------------------------

## Run order

```bash
python tutorial_train_VAE.py    # 1. new VAE
python tutorial_train.py        # 2. set vae_weights_path first
python val.py --dataset test    # 3. set controlnet_ckpt / vae_weights_path first
```
