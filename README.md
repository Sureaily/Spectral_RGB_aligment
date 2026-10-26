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
