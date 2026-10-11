# FCHMamba

### Frequency-Complementary Fusion with Prior-Compatible Propagation for Remote Sensing Single-View Height Estimation

[中文说明](README.zh-CN.md) · [GitHub](https://github.com/BPXwithNQP/FCHMamba)

**Qipei Nong**<sup>1,†</sup>, **Zhengpeng Li**<sup>1,2,†</sup>, **Zheng Gao**<sup>3</sup>, **Weichun Guan**<sup>1</sup>, **Shuxian Li**<sup>1</sup>, **Yifan Zhang**<sup>1</sup>, **Yuanyuan Zhang**<sup>1,*</sup>, **Jun Hu**<sup>2,**</sup>

<sup>1</sup> School of Computer Science and Software Engineering, University of Science and Technology Liaoning, Anshan, 114051, Liaoning, China  
<sup>2</sup> School of Electronic and Information Engineering, University of Science and Technology Liaoning, Anshan, 114051, Liaoning, China  
<sup>3</sup> School of Computer Science and Engineering, University of New South Wales, Sydney, NSW 2052, Australia

<sup>†</sup> Equal contribution. <sup>*</sup> Corresponding author. <sup>**</sup> Co-corresponding author.

Yuanyuan Zhang's ORCID: [0009-0003-7773-1856](https://orcid.org/0009-0003-7773-1856).

## Overview

FCHMamba estimates a dense metric height map from a single RGB remote sensing image. It combines pretrained structural representations with local high-frequency information within a single VMamba backbone.

The method addresses two related questions: **what local information should complement pretrained features, and how far should that compensation propagate?**

## Dataset access

Obtain the datasets through their official pages and follow the providers' terms of use. This repository does not redistribute the raw data.

| Dataset | Official page |
| --- | --- |
| Vaihingen | [ISPRS Vaihingen dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) |
| DFC2019 Track 1 | [IEEE GRSS 2019 Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/) |
| DFC2023 Track 2 | [IEEE GRSS 2023 Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/) |

## Code and model weights

The repository includes the FCHMamba model assembly, configuration files, dataset and evaluation utilities, and training/testing entry points.

- `models/model.py`: FCHMamba model assembly and prediction heads.
- `models/Blocks/GSFB.py`: gated multiscale skip fusion.
- `configs/` and `models/Blocks/block.yaml`: backbone and module configurations.
- `train_config/`: training configurations for Vaihingen, DFC2019, and DFC2023.
- `train.py`, `test.py`, `utils/`, and `nqp_utils/`: training, evaluation, metrics, and profiling utilities.

Download the trained FCHMamba checkpoints and pretrained backbone weights from [Baidu Netdisk](https://pan.baidu.com/s/1Ba6Yo-GnbK30OgYpWCxZtg?pwd=2026) using extraction code `2026`. Set the corresponding `weights_path` in the backbone configuration before training or evaluation.

### Environment

The code was developed for Python 3.10 and CUDA-enabled PyTorch. Install the supplied dependency sets according to the local CUDA environment:

```bash
pip install -r requirements_rest.txt
pip install -r requirements_mambavision.txt
```

The state-space backbones require their corresponding selective-scan CUDA implementation. Adjust the dataset paths in `train_config/*.yaml` before running an experiment.

### Training and evaluation

```bash
# DFC2023 training
python main--.py --encoder1 tinyvim --enc_opt mit --dataset_root ../dataset \
  --dataset_name dfc23-512 --max_height_eval 200 --epochs 150 --batch_size 4 \
  --lr 5e-5 --progress_freq 1000 --de_normal --device cuda:0 \
  --use_DMSN 1 --use_RGHF 1 --use_GSFB 1 --job_name FCHMamba_train

# Evaluation with a downloaded checkpoint
python test.py --dataset_root ../dataset --dataset_name dfc23-512 \
  --ckpt_path /path/to/FCHMamba.ckpt
```

## Model comparison: RMSE on three benchmarks

The table reproduces RMSE from the manuscript's main comparisons, in meters (lower is better). FCHMamba uses the default PCC route. Best values are bold.

| Method | Vaihingen | DFC2019 | DFC2023 |
| --- | ---: | ---: | ---: |
| MAFENet | 1.418 | 1.538 | 4.578 |
| Elevation-CVMamba* | 1.496 | 2.250 | — |
| RAEPHR-Net* | 1.448 | 1.950 | — |
| FusedHE | 1.366 | 1.660 | 3.304 |
| HTC-DC-Net | 1.343 | 2.228 | 4.059 |
| Im2Height | 2.430 | 3.757 | 6.470 |
| SegFormer | 1.631 | 1.853 | 3.790 |
| VMUPerNet | 1.525 | 1.683 | 4.155 |
| ResNet-UPerNet | 1.804 | 1.838 | 4.225 |
| TinyViM-FHE | 1.453 | 1.611 | 3.452 |
| Depth Anything V3 | 1.629 | 1.783 | 5.502 |
| UniDepth V2 | 1.289 | 1.799 | 4.809 |
| MoGe-2 | 1.331 | 1.494 | 4.638 |
| Depth Anything V2 | 1.371 | 1.614 | 4.661 |
| **FCHMamba (Ours)** | **1.258** | **1.457** | **2.853** |

\* Results taken from the original papers; other results are our re-implementations. — indicates a result not reported in the current manuscript.

FCHMamba achieves the lowest RMSE among the evaluated methods on all three benchmarks. Relative reductions over the best evaluated baseline are **2.4%**, **2.5%**, and **13.7%**, respectively.

The model has **108.21 M parameters** and requires **286.20 GFLOPs** under the manuscript's input configuration, balancing prediction accuracy and computational cost.

## Model architecture

- **Frequency-compensated encoder (VM-FC):** Directional frequency decomposition, cross-band mixing, and learned gates construct multiscale high-frequency compensation.
- **Dual-stream adaptive decoder (DSAD):** Directional long-range modeling and gated skip fusion combine regional context with local structural cues during progressive height recovery.
- **Prior-Compatible Compensation (PCC):** The default route preserves the native inter-stage encoder path while supplying compensation to the decoder. Backbone-Coupled Injection (BCI) provides a matched comparison by also passing compensation into subsequent encoder stages.

[![Overall FCHMamba architecture](assets/architecture.png)](assets/architecture.pdf)

[View the original architecture PDF](assets/architecture.pdf)

VM-FC constructs multiscale compensation within one backbone. DSAD progressively combines native deep context with compensated skip features to recover the height map. The legacy label “PCI” in the original figure denotes the route called PCC throughout this project; the supplied PDF is reproduced unchanged.

## Frequency compensation and propagation

[![VM-FC and the matched PCC/BCI propagation routes](assets/vmfc.png)](assets/vmfc.pdf)

[View the original VM-FC PDF](assets/vmfc.pdf)

Both routes use the same compensation construction and compensated decoder skips. Only BCI writes compensation into subsequent encoder states. At the deepest stage, compensation is added after DMSN context modeling and before decoder upsampling.

## Spectral analysis

### Pretraining and task adaptation

Pretrained backbones exhibit low-frequency dominance, while fine-tuning reorganizes their frequency responses differently across architectures and datasets. VMamba's mean HF/LF changes from approximately 0.434 before fine-tuning to 0.470 on Vaihingen and 0.515 on DFC2023.

[![Backbone frequency responses before and after fine-tuning](assets/backbone-spectrum.png)](assets/backbone-spectrum.pdf)

[View the original spectral PDF](assets/backbone-spectrum.pdf)

### Stage-wise effects of compensation routing

PCC concentrates high-frequency enhancement in deeper stages. BCI introduces changes into early and middle stages and propagates them through subsequent states. Routing therefore changes both the strength and hierarchical distribution of frequency responses; stronger high-frequency responses do not necessarily imply lower prediction error.

[![Frequency responses of VMamba and the PCC/BCI routes](assets/route-spectrum.png)](assets/route-spectrum.pdf)

[View the original routing-spectrum PDF](assets/route-spectrum.pdf)

### Robustness to frequency-band thresholds

The table summarizes mean HF/LF from the manuscript's threshold experiment. The low and high thresholds are normalized radial-frequency boundaries. Absolute values vary, but the relative trend remains consistent across configurations, indicating low sensitivity of the main spectral conclusions to the tested boundaries.

| Low / high | Pretrained VMamba | Fine-tuned VMamba | VM-FC |
| --- | ---: | ---: | ---: |
| 0.20 / 0.45 | 0.5889 | 0.7111 | 0.7833 |
| 0.30 / 0.55 | 0.2956 | 0.3510 | 0.3838 |
| 0.35 / 0.60 | 0.2079 | 0.2438 | 0.2675 |

HF/LF describes frequency responses, not prediction accuracy. These statistics use a fixed probe set of 500 ImageNet validation images.

## Qualitative results

The panels below reproduce the manuscript's comparisons. **Ours** denotes FCHMamba. Click an image to inspect it at full resolution.

### Vaihingen

![Height predictions on Vaihingen](assets/vaihingen.png)

### DFC2019

![Height predictions on DFC2019](assets/dfc2019.png)

### DFC2023

![Height predictions on DFC2023](assets/dfc2023.png)

## Availability

- **Source code:** Available in this repository.
- **Model weights:** Trained FCHMamba checkpoints and pretrained backbone weights are available from [Baidu Netdisk](https://pan.baidu.com/s/1Ba6Yo-GnbK30OgYpWCxZtg?pwd=2026), extraction code `2026`.
- **Paper and citation:** Publication information and a citation entry will be added when available.

This repository does not redistribute the benchmark datasets. Dataset access and use remain subject to the respective providers' terms.

## Funding

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62376036 and 52574151), the Doctoral Student Program of the Young S&T Talents Cultivation Project (CAST), and the Fundamental Research Project of Higher Education Institutions by Liaoning Provincial Department of Education (LJ222410146057, LJ212410146040, LJ212410146006).

## Acknowledgements

We thank DGPF for the Vaihingen data and the data providers and organizers of the IEEE GRSS 2019 and 2023 Data Fusion Contests.

## License

See [LICENSE](LICENSE) for the research-use terms and attribution requirements associated with the included FusedHE- and SegFormer-derived components.

## Contact

For research enquiries, contact **Yuanyuan Zhang** ([yuanyuan@ustl.edu.cn](mailto:yuanyuan@ustl.edu.cn)) or **Jun Hu** ([320083700074@ustl.edu.cn](mailto:320083700074@ustl.edu.cn)).
