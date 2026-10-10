import argparse
import datetime
import random
import os
import sys
from pathlib import Path

_NQP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _NQP_ROOT not in sys.path:
    sys.path.insert(0, _NQP_ROOT)
from nqp_utils import profile_model, profile_model_gflops_note

import utils.conda_glibcxx  # noqa: F401  # 须在 torch 之前（GLIBCXX / selective_scan）

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from torch import nn, optim
import utils.misc as utils
from utils.dataset_config import add_gbh_dataset_args, build_train_val_datasets, maybe_apply_ndsm_pickle_max_height
from utils.encoder_presets import add_encoder_args, build_optimizer_param_groups, resolve_encoder_args
from utils.backbone_freeze import (
    apply_groupmamba_pretrained_freeze,
    apply_network_freeze,
    configure_warm_start_network_freeze,
    network_freeze_active,
)
from utils.criterion import DepthSegLoss
from models.model import FHE
from train import train, validate, save_val_vis_from_cache, state_dict_to_cpu

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def get_args_parser():
    parser = argparse.ArgumentParser('FusedHE', add_help=False)
    parser.add_argument(
        '--lr', default=5e-5, type=float,
        help='非主干（decoder/头等）学习率；主干见 configs 各 yaml 的 backbone_lr，未设时回退 --lr_backbone',
    )
    parser.add_argument(
        '--lr_backbone_names', default=["encoder.mit"], type=str, nargs='+',
        help='（兼容旧 enc_opt ）主干参数名匹配；fused+yaml 的 backbone_lr 时通常不依赖本项',
    )
    parser.add_argument(
        '--lr_backbone', default=5e-5, type=float,
        help='configs 中未写 backbone_lr 时主干的回退学习率；与 --lr 可独立',
    )
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int) #batch size
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', '--epoch', dest='epochs', default=70, type=int) # epochs
    parser.add_argument(
        '--cos_open',
        action='store_true',
        help='余弦退火学习率：每 epoch 步进，无周期重启；经 cos_max 个 epoch 从各组当前初始 lr 余弦降至 cos_min 后保持',
    )
    parser.add_argument(
        '--cos_min',
        default=1e-5,
        type=float,
        help='余弦退火最小学习率（需小于各 param_group 的初始 lr；主干与其它组分别用 lr_backbone / lr 作为起点）',
    )
    parser.add_argument(
        '--cos_max',
        default=80,
        type=int,
        help='余弦下降周期 T_max：在此 epoch 数内从初始 lr 降至 cos_min，之后保持 cos_min',
    )
    parser.add_argument('--lr_drop', default=[50], type=list,
                        help='与 cos_open 同用时忽略（未接 StepLR 调度器）')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lambd', default=0.005,type=float) # coeffitiont of Segmentation Loss
    parser.add_argument('--sgd', action='store_true')

    # [depth or full], depth: Only height head, full:height+Segmentation head
    parser.add_argument('--option', default='full', type=str) 

    # [mit, conv, fused], Encoder type, mit: Hierarchical Vision transformers, conv: Resnet , Fused:mit+conv
    parser.add_argument('--enc_opt', default='fused', type=str)
    parser.add_argument('--use_condition', action='store_true',
                        help='启用条件高度分区：bin_num 档 bin + 区间内残差 + 反频率 Focal Loss')
    parser.add_argument('--no_distribution_constraint', action='store_true',
                        help='禁用区间概率分布约束（默认开启，与 use_condition 同时启用时施加高斯分布约束）')
    parser.add_argument('--dist_constraint_weight', default=0.1, type=float,
                        help='distribution_constraint 时 L_prob 权重')
    parser.add_argument('--use_adabin', action='store_true',
                        help='启用自适应分区边界（AdaBins），从 Stage4 全局特征预测 bin edges，开启时忽略 bin_len')
    parser.add_argument('--chamfer_weight', default=0.1, type=float,
                        help='AdaBins 的 Chamfer Loss 权重')
    parser.add_argument('--uniform_weight', default=0.1, type=float,
                        help='AdaBins 的 bin 宽度均匀正则化权重')
    parser.add_argument('--max_perturbation', default=0.5, type=float,
                        help='AdaBins tanh 扰动范围 [0,1)，0.5 表示每个 bin 宽度可偏离均匀的 ±50%%')
    parser.add_argument('--bin_num', default=10, type=int,
                        help='条件分区档数')
    parser.add_argument('--bin_len', default=3, type=float,
                        help='条件分区每档长度（米），仅 use_adabin 关闭时有效')
    parser.add_argument('--de_normal', action='store_true',
                        help='在 loss 和验证中按 max_height_eval 反归一化到米制计算，与非 or 版本一致')
    # parser.add_argument(
    #     '--mask_loss',
    #     default=0.0,
    #     type=float,
    #     help='训练/验证 depth loss 前对预测高度（米）设阈值：<mask_loss 置 0，≥ 保留；'
    #          '0 表示不处理。可与 test.py --mask_test=2 一致地设为 1.0、2.0、3.0 等。',
    # )
    parser.add_argument(
        '--condition_k', default=0, type=int,
        help='条件分区估计高度：每像素 softmax 概率取 top-k（重归一化后）与 centers+残差加权；'
             '0 表示使用全部档位（等价于原版）',
    )
    parser.add_argument('--bin_weight', default=0.15, type=float, help='条件分区时 L_bin 权重')
    parser.add_argument('--height_weight', default=0.85, type=float, help='条件分区时 L_height+L_res 权重')
    parser.add_argument('--scale_height', default=1.0, type=float, help='条件分区时 L_height 缩放')

    # ---- HL 整尾部（HLAttention.HLTail） ----
    parser.add_argument(
        '--use_HL', action='store_true',
        help='启用 HL 整尾部：解耦(box-avg)+边界注意力+区域门控+分割头(去ReLU)+先验注入高度头(+地表软钳)；'
             '会自动把 option 改为 full，并替换原 condition_heads/depth_head + segment_head',
    )
    parser.add_argument(
        '--hl_avg_kernel', default=11, type=int,
        help='HL 解耦的 depthwise 盒平均核大小（作用于 dec_final，不是输入图像；偶数会自动+1）',
    )
    parser.add_argument(
        '--hl_alpha_init', default=0.0, type=float,
        help='F_seg = F + tanh(α)·A·Conv(H) 中 α 的初值（tanh 之前），默认 0：训练初期等价 F_seg=F',
    )
    parser.add_argument(
        '--hl_region_temp_init', default=1.0, type=float,
        help='区域门控温度 τ 初值（log-参数化，保持 >0），更小 → A_region 更接近硬门',
    )
    parser.add_argument(
        '--use_ground_cap', action='store_true', default=True,
        help='在 HLTail 输出端对地表(p_subject 低)区域软钳到 ground_cap_m，并在 loss 中加 L_ground_cap',
    )
    parser.add_argument(
        '--no_ground_cap', dest='use_ground_cap', action='store_false',
        help='关闭 use_ground_cap',
    )
    parser.add_argument(
        '--ground_cap_m', default=1.0, type=float,
        help='地表高度上限（米）。GT 分割==0 的像素显式不准超过该值',
    )
    parser.add_argument(
        '--ground_cap_weight', default=0.1, type=float,
        help='L_ground_cap 损失权重（未启用 use_HL 或 use_ground_cap 时无效）',
    )
    parser.add_argument(
        '--freeze',
        default=0,
        type=int,
        help='前 N epoch 全网络冻结、仅训 warm_start 相对 ckpt 的新增模块（RGHF/DMSN/GSFB/Condition）；'
        '0=关闭；未显式指定且 warm_start 有新增模块时自动 N=5；N+1 起解冻并重建 optimizer',
    )

    parser.add_argument('--pre_trained', default=True, type=bool)
    parser.add_argument('--num_classes', default=1, type=int) #for segmentation head
    parser.add_argument('--in_channel_ffb', default=[2048, 1024, 512, 256], type=list)
                        #only enc_opt for conv, fused. usage = [2048, 1024, 512, 256]
    parser.add_argument('--out_channel_ffb', default=[512, 320, 128, 64], type=list)
                        #only enc_opt for mit, fused. usage = [512, 320, 128, 64] 
    parser.add_argument(
        '--native_proj_ff',
        action='store_true',
        help='GroupMamba 单主干 decoder 使用 ConvTranspose ff（默认旧 GM 为 native_align_ff：1×1+3×3 Conv2d）',
    )
    parser.add_argument(
        '--single_decoder_channels',
        default=[1024, 640, 256, 128],
        type=list,
        help='GroupMamba 旧版单主干 decoder 各级 ff 固定宽度',
    )

    parser.add_argument('--in_channel_decoderb', default=[1024, 640, 256], type=list)
                        #mit,fused = [1024, 640, 256] - conv = [1024, 512, 256]
    parser.add_argument('--out_channel_decoderb', default=[640, 256, 128], type=list)
                        #mit,fused = [640, 256, 128] - conv = [512, 256, 128]
    parser.add_argument('--out_decoder', default=64, type=int)
    parser.add_argument('--job_name', default='fused_with_segment_head', type=str)

    # dataset parameters（数据集统一放在 ../dataset 下，默认 dfc2023no-512）
    _default_dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--dataset_root', default=_default_dataset_root, type=str,
                        help='数据集主目录，默认 ../dataset')
    parser.add_argument('--dataset_name', default='dfc23-512', type=str,
                        help='数据集名字，路径为 dataset_root/dataset_name/{train,validation,test}')
    parser.add_argument('--train_dir', default=None, type=str,
                        help='训练集路径；不传则用 dataset_root/dataset_name/train')
    parser.add_argument('--val_dir', default=None, type=str,
                        help='验证集路径；不传则用 dataset_root/dataset_name/validation')
    add_encoder_args(parser)
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--warm_start',
        default='',
        type=str,
        help='从已有 checkpoint 加载可匹配权重并从 epoch 0 开始新实验；不同于 --resume，不继承输出目录和 epoch',
    )
    parser.add_argument(
        '--grfa_train_phase',
        default='none',
        choices=['none', 'residual', 'finetune'],
        help='GRFA 热启动训练阶段：none=不冻结；residual=只训 band_refine/gate/gamma；finetune=解冻 decoder/head/GRFA，可选冻结 backbone',
    )
    parser.add_argument(
        '--grfa_finetune_backbone',
        action='store_true',
        help='grfa_train_phase=finetune 时同时解冻 encoder/backbone；默认 finetune 阶段仍冻结 backbone',
    )
    parser.add_argument(
        '--grfa_edge_loss_weight',
        default=0.0,
        type=float,
        help='GRFA 空间门边缘辅助监督权重；0 表示关闭。开启时对各 stage edge_gate 加 BCE',
    )
    parser.add_argument(
        '--grfa_edge_threshold_m',
        default=0.75,
        type=float,
        help='GRFA 边缘辅助监督阈值（米/像素），对 GT nDSM 的 Sobel 梯度幅值二值化',
    )
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument(
        '--amp',
        action='store_true',
        help='启用 torch.cuda.amp 混合精度训练，通常可降低显存并提升 4090 吞吐',
    )
    parser.add_argument(
        '--train_bin_metric_freq',
        default=20,
        type=int,
        help='训练阶段 bin top-k 指标计算频率；>1 可减少 CUDA 同步，验证仍每批完整计算',
    )
    parser.add_argument(
        '--disable_bin_metrics',
        action='store_true',
        help='关闭 train/val 的 bin-top 指标计算；不影响 use_condition 的 bin loss',
    )
    parser.add_argument(
        '--progress_freq',
        default=1,
        type=int,
        help='训练/验证进度条刷新间隔；调大可减少终端 IO 与 .item() 同步',
    )
    parser.add_argument(
        '--accum_steps',
        default=1,
        type=int,
        help='梯度累积步数；有效 batch = batch_size × accum_steps，可降低单步显存',
    )
    parser.add_argument(
        '--activation_checkpoint',
        action='store_true',
        help='对 FusedDecoder 的 GRFA/GSFB/DMSN 等模块启用 activation checkpointing，降低显存',
    )
    parser.add_argument(
        '--val_vis_target_idx',
        default=5,
        type=int,
        help='可选：固定 val_vis 的数据集全局下标；不设则每次验证随机一张（与 HTC-DC 一致）',
    )
    parser.add_argument('--exp_name', default=True, type=bool)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_results', default=True, type=bool)
    parser.add_argument(
        '--max_height_eval',
        default=200,
        type=int,
        help='反归一化到米制时的最大高度（米）。仅当设为 0 时从数据集 data_dir/ndsm_stats.pickle 自动读取并打印；否则使用本参数或 checkpoint 中的数值，不从 pickle 覆盖。',
    )
    parser.add_argument('--min_height_eval', default=1e-4, type=float)
    parser.add_argument(
        '--mask_depth_threshold',
        default=1.0,
        type=float,
        help='无数据集 BLG 时 eval_depth 的建筑前景 GT 阈值（米）；有 BLG 时前景以 mask 为准，本阈值仍用于 pred>thr 的 IoU 预测侧',
    )
    parser.add_argument(
        '--paper_building_min_m',
        default=0.0,
        type=float,
        help='无 BLG 时 RMSE-M/NM 建筑像素定义：GT>该值（米）',
    )
    parser.add_argument('--profile_height', default=512, type=int)
    parser.add_argument('--profile_width', default=512, type=int)
    parser.add_argument('--dataset', default="local", type=bool)

    add_gbh_dataset_args(parser)
    return parser


def _split_ckpt_state(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _filter_state_dict_for_model(model, state):
    """仅保留与当前模型键名、形状一致的权重，避免 warm_start 结构不一致时 load 报错。"""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped


def _load_model_weights(model, ckpt_path, device, tag):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = _split_ckpt_state(ckpt)
    state, skipped = _filter_state_dict_for_model(model, state)
    load_ok = model.load_state_dict(state, strict=False)
    missing = list(getattr(load_ok, "missing_keys", []))
    unexpected = list(getattr(load_ok, "unexpected_keys", []))
    print(
        "[{}] 已加载: {} | missing={} unexpected={}".format(
            tag, ckpt_path, len(missing), len(unexpected)
        ),
        flush=True,
    )
    if missing[:8]:
        print("[{}] missing 示例: {}".format(tag, missing[:8]), flush=True)
    if unexpected[:8]:
        print("[{}] unexpected 示例: {}".format(tag, unexpected[:8]), flush=True)
    if skipped:
        print(
            "[{}] 跳过 shape 不匹配的参数: {} 个（示例: {}）".format(
                tag, len(skipped), skipped[:4]
            ),
            flush=True,
        )
    return ckpt


def _is_grfa_residual_param(name: str) -> bool:
    if ".ff" not in name:
        return False
    return any(
        token in name
        for token in (".band_refine.", ".gate.", ".gamma", ".band_mix.")
    )


def _build_optimizer(model, args):
    param_groups = build_optimizer_param_groups(model, args)
    optimizer = optim.Adam(param_groups, lr=args.lr)
    for i, g in enumerate(param_groups):
        print(
            "[lr]  param_group[{}] lr={:.2e}  nparams={}".format(
                i, float(g["lr"]), sum(p.numel() for p in g["params"])
            ),
            flush=True,
        )
    return optimizer


def _apply_grfa_train_phase(model, args) -> None:
    phase = str(getattr(args, "grfa_train_phase", "none")).lower()
    if phase == "none":
        return
    if not bool(getattr(args, "use_GRFA", False)):
        print("[GRFA warm] grfa_train_phase={} 但 use_GRFA=False，跳过冻结设置".format(phase), flush=True)
        return

    if phase == "residual":
        for name, param in model.named_parameters():
            param.requires_grad = _is_grfa_residual_param(name)
    elif phase == "finetune":
        freeze_backbone = not bool(getattr(args, "grfa_finetune_backbone", False))
        for name, param in model.named_parameters():
            if freeze_backbone and name.startswith("encoder."):
                param.requires_grad = False
            else:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        "[GRFA warm] phase={} trainable={}/{} ({:.2f}%)".format(
            phase, trainable, total, 100.0 * trainable / max(total, 1)
        ),
        flush=True,
    )


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    warm_path = getattr(args, "warm_start", "") or ""
    resume_path = getattr(args, "resume", "") or ""
    ckpt_for_structure = ""
    if resume_path and os.path.isfile(resume_path):
        ckpt_for_structure = resume_path
    elif warm_path and os.path.isfile(warm_path):
        ckpt_for_structure = warm_path

    if ckpt_for_structure:
        from test import prepare_fhe_args_from_checkpoint

        runtime_keys = (
            "device",
            "output_dir",
            "run_name",
            "job_name",
            "epochs",
            "resume",
            "warm_start",
            "start_epoch",
            "num_workers",
            "batch_size",
            "lr",
            "lr_backbone",
            "seed",
            "use_RGHF",
            "use_GRFA",
            "use_FAF",
            "use_GSFB",
            "use_DMSN",
            "use_condition",
            "freeze",
            "progress_freq",
        )
        runtime_overrides = {k: getattr(args, k) for k in runtime_keys if hasattr(args, k)}

        ckpt = torch.load(ckpt_for_structure, map_location="cpu")
        state = _split_ckpt_state(ckpt)
        prepare_fhe_args_from_checkpoint(args, state, ckpt, verbose=True)
        for key, value in runtime_overrides.items():
            setattr(args, key, value)
        _tag = "resume" if ckpt_for_structure == resume_path else "warm_start"
        print("[{}] 已根据 checkpoint 对齐模型结构: {}".format(_tag, ckpt_for_structure), flush=True)
    else:
        resolve_encoder_args(args)
    print(
        "encoder1={} encoder2={} enc_opt={} use_adabin={} use_HL={} use_FAF={} use_GRFA={} "
        "use_GSFB={} use_FDSF={} use_DMSN={} use_DPLD={} use_TBWH={} use_FHRM={}".format(
            getattr(args, "encoder1_type", "?"),
            getattr(args, "encoder2_type", "?"),
            getattr(args, "enc_opt", "fused"),
            getattr(args, "use_adabin", False),
            getattr(args, "use_HL", False),
            getattr(args, "use_FAF", False),
            getattr(args, "use_GRFA", False),
            getattr(args, "use_GSFB", 0),
            getattr(args, "use_FDSF", False),
            getattr(args, "use_DMSN", 0),
            getattr(args, "use_DPLD", False),
            getattr(args, "use_TBWH", 0),
            getattr(args, "use_FHRM", 0),
        )
    )
    _accum = max(1, int(getattr(args, "accum_steps", 1)))
    _eff_bs = int(getattr(args, "batch_size", 4)) * _accum
    print(
        "[mem] batch_size={} accum_steps={} => 有效 batch={} | activation_checkpoint={} | amp={} | disable_bin_metrics={}".format(
            getattr(args, "batch_size", 4),
            _accum,
            _eff_bs,
            bool(getattr(args, "activation_checkpoint", False)),
            bool(getattr(args, "amp", False)),
            bool(getattr(args, "disable_bin_metrics", False)),
        ),
        flush=True,
    )
    # _mask_loss = float(getattr(args, "mask_loss", 0.0))
    # if _mask_loss > 0:
    #     print(
    #         "[mask_loss] depth loss 前对预测高度（米）设阈值：<{:.4g} 置 0，≥ 保留".format(_mask_loss),
    #         flush=True,
    #     )
    if getattr(args, "use_HL", False):
        print(
            "[HLTail] 已启用：avg_kernel={} alpha_init={} region_temp={} ground_cap={}m (weight={})".format(
                getattr(args, "hl_avg_kernel", 11),
                getattr(args, "hl_alpha_init", 0.0),
                getattr(args, "hl_region_temp_init", 1.0),
                getattr(args, "ground_cap_m", 1.0),
                getattr(args, "ground_cap_weight", 0.1),
            )
        )
    print(args)

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    maybe_apply_ndsm_pickle_max_height(args)
    train_data, val_data = build_train_val_datasets(args)

    _nw = int(getattr(args, "num_workers", 0))
    _pin = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    _loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=_nw,
        pin_memory=_pin,
    )
    if _nw > 0:
        _loader_kw["persistent_workers"] = True
        _loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_data, shuffle=True, **_loader_kw)
    val_loader = DataLoader(val_data, shuffle=False, **_loader_kw)

    _layout = getattr(args, "dataset_layout", "folder")
    if _layout == "gbh_txt":
        _dd = args.gbh_data_dir or os.path.join(args.dataset_root, args.dataset_name)
        _sd = args.gbh_split_dir or os.path.join(_dd, "data_split_dirs")
        print(
            "[数据集] layout=gbh_txt data_dir={} split_dir={} train={} val={}".format(
                _dd, _sd, len(train_data), len(val_data)
            ),
            flush=True,
        )
    else:
        print(
            "[数据集] layout=folder train={} 张, validation={} 张".format(
                len(train_data), len(val_data)
            ),
            flush=True,
        )
    print(
        "[DataLoader] num_workers={} pin_memory={}（若 GPU 仍空闲，可加大 num_workers）".format(
            _nw, _pin
        )
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    dataset_sizes = {
        "train": len(train_data),
        "val": len(val_data)
    }

    val_vis_target_idx = getattr(args, "val_vis_target_idx", None)
    if val_vis_target_idx is not None and len(val_data) > 0:
        val_vis_target_idx = max(0, min(int(val_vis_target_idx), len(val_data) - 1))
        print(
            "[val_vis] 使用固定数据集下标 idx={}（不设 --val_vis_target_idx 则每轮验证随机一张）".format(
                val_vis_target_idx
            ),
            flush=True,
        )
    else:
        print(
            "[val_vis] 仅在 new best（验证 RMSE 创新低）时保存一张对比图（seed+epoch 随机选样）",
            flush=True,
        )

    if (
        getattr(args, "encoder2_type", "") == "mit"
        and getattr(args, "enc_opt", "") == "fused"
        and getattr(args, "pre_trained", True)
    ):
        wp = (getattr(args, "encoder2_cfg", {}) or {}).get("weights_path")
        if wp:
            mit_abs = os.path.abspath(wp)
            if not os.path.isfile(mit_abs):
                raise FileNotFoundError(
                    "未找到 MiT 预训练权重: {}（当前工作目录: {}）".format(
                        mit_abs, os.getcwd()
                    )
                )
            print("[预训练] 将从此路径加载 MiT (encoder2): {}".format(mit_abs), flush=True)

    print("[模型] 构建 FHE 并加载主干权重 …", flush=True)
    model = FHE(args)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    fh = int(getattr(args, "profile_height", 512))
    fw = int(getattr(args, "profile_width", fh))
    prof = profile_model(model, fh, fw, inner=model)
    args._nqp_profile = prof
    _gf = prof["gflops"]
    print(
        "[模型] params={}  GFLOPs={} (@{}×{}, {})".format(
            prof["params"],
            "{:.4f}".format(_gf) if _gf == _gf else "nan",
            fh,
            fw,
            profile_model_gflops_note(prof),
        ),
        flush=True,
    )

    best_acc = 100.0
    start_epoch = 0
    # 最近 5 次出现 best 时保存的 (epoch, path)，覆盖式更新
    best_save_bag = []
    BEST_KEEP = 15

    if args.resume and os.path.isfile(args.resume):
        _load_model_weights(model, args.resume, device, "resume")
        start_epoch = getattr(args, 'start_epoch', 0)
        print(f"Starting from epoch {start_epoch + 1}")
    elif args.warm_start and os.path.isfile(args.warm_start):
        ckpt_loaded = _load_model_weights(model, args.warm_start, device, "warm_start")
        start_epoch = 0
        print("[warm_start] 从 epoch 1 开始新训练", flush=True)
        configure_warm_start_network_freeze(
            args, _split_ckpt_state(ckpt_loaded), verbose=True
        )
    elif args.warm_start:
        raise FileNotFoundError("warm_start checkpoint 不存在: {}".format(args.warm_start))

    _apply_grfa_train_phase(model, args)
    apply_network_freeze(model, args, epoch=start_epoch + 1, verbose=True)
    apply_groupmamba_pretrained_freeze(model, args, epoch=start_epoch + 1, verbose=False)
    best_model_wts = state_dict_to_cpu(model)

    criterion_d = DepthSegLoss(args=args)

    optimizer_d = _build_optimizer(model, args)

    #schedular = optim.lr_scheduler.StepLR(optimizer_d)

    model.to(device)

    cosine_sched = None
    if getattr(args, "cos_open", False):
        from utils.cosine_anneal import CosineAnnealingHoldEnd

        _tmax = int(max(getattr(args, "cos_max", 70), 1))
        cosine_sched = CosineAnnealingHoldEnd(
            optimizer_d,
            T_max=_tmax,
            eta_min=float(getattr(args, "cos_min", 1e-6)),
            last_step=int(start_epoch),
        )
        print(
            "[lr] 余弦退火: T_max={} epoch, cos_min={}（无周期重启，之后保持为 cos_min）| 恢复步数={}".format(
                _tmax,
                getattr(args, "cos_min", 1e-6),
                int(start_epoch),
            ),
            flush=True,
        )

    log_path = os.path.join(args.output_dir, "train.log")
    if start_epoch > 0:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nResumed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from {args.resume}, start_epoch={start_epoch}\n")
            f.write(f"Command line args:\n{args}\n\n")
    else:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command line args:\n{args}\n\n")

    def _fmt_run_log(v):
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        was_network_frozen = network_freeze_active(args, epoch - 1)
        apply_network_freeze(model, args, epoch=epoch, verbose=(epoch == start_epoch + 1))
        apply_groupmamba_pretrained_freeze(
            model, args, epoch=epoch, verbose=(epoch == start_epoch + 1)
        )
        if was_network_frozen and not network_freeze_active(args, epoch):
            print("[freeze] 解冻后重建 optimizer", flush=True)
            optimizer_d = _build_optimizer(model, args)
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        total_loss, depth_loss, segment_loss, bin_loss, tk_tr, top1_tr, k_bins_cfg, model, fh_tr = train(
                train_loader, model, criterion_d, optimizer_d=optimizer_d,
                                device=device, epoch=epoch, args=args, option=args.option)
        train_log_stats = {"train/total_loss": total_loss,
                          "train/depth_loss": depth_loss,
                          "train/segment_loss": segment_loss,
                          "train/bin_loss": bin_loss}
        if fh_tr:
            train_log_stats.update(fh_tr)
        if k_bins_cfg is not None and tk_tr == tk_tr:
            train_log_stats["train/bin_top{}_hit".format(k_bins_cfg)] = tk_tr
            train_log_stats["train/bin_top1_acc"] = top1_tr

        print(train_log_stats)

        if epoch % args.val_freq == 0:
            results_dict, total_loss_val, depth_loss_val, loss_s_val, bin_val, vk_tr, vk1_tr, vk_bins_cfg, model, fh_val, val_vis_cache = validate(
                val_loader, model, criterion_d, device=device, epoch=epoch, args=args,
                log_dir=args.output_dir, option=args.option, val_vis_target_idx=val_vis_target_idx)

            val_log_stats = {
                "val/total_loss": total_loss_val,
                "val/depth_loss": depth_loss_val,
                "val/segment_loss": loss_s_val,
                "val/rmse": results_dict["rmse"],
                "val/mae": results_dict.get("mae", float("nan")),
                "val/rmse_m": results_dict["rmse_m"],
                "val/rmse_nm": results_dict["rmse_nm"],
                "val/rmse_b": results_dict["rmse_b"],
                "val/d1": results_dict["d1"],
                "val/d2": results_dict["d2"],
                "val/d3": results_dict["d3"],
                "val/b10": results_dict.get("b10", float("nan")),
                "val/b0": results_dict.get("b0", float("nan")),
                "val/m10": results_dict.get("m10", float("nan")),
                "val/m0": results_dict.get("m0", float("nan")),
                "val/iou": results_dict["iou"],
            }
            if fh_val:
                val_log_stats.update(fh_val)
            if isinstance(getattr(args, "_nqp_profile", None), dict):
                val_log_stats["params"] = int(args._nqp_profile["params"])
                val_log_stats["gflops"] = float(args._nqp_profile["gflops"])
            if getattr(args, 'use_condition', False):
                val_log_stats["val/bin_loss"] = bin_val
            if vk_bins_cfg is not None and vk_tr == vk_tr:
                val_log_stats["val/bin_top{}_hit".format(vk_bins_cfg)] = vk_tr
                val_log_stats["val/bin_top1_acc"] = vk1_tr

            print(val_log_stats)

            epoch_rmse = results_dict["rmse"]
            if epoch_rmse < best_acc:
                print('best')
                best_acc = epoch_rmse
                best_model_wts = state_dict_to_cpu(model)
                save_val_vis_from_cache(args.output_dir, epoch, val_vis_cache, args)
                if args.save_model:
                    path = os.path.join(args.output_dir, 'best_epoch_%03d.ckpt' % epoch)
                    torch.save({'state_dict': best_model_wts, 'args': vars(args)}, path)
                    best_save_bag.append((epoch, path))
                    while len(best_save_bag) > BEST_KEEP:
                        old_epoch, old_path = best_save_bag.pop(0)
                        if os.path.isfile(old_path):
                            os.remove(old_path)

            if cosine_sched is not None:
                cosine_sched.step()
            for i, g in enumerate(optimizer_d.param_groups):
                train_log_stats["train/lr_g%d" % i] = g["lr"]

            log_str = f"Epoch {epoch:03d}  [TRAIN]  " + " | ".join(
                f"{k}: {_fmt_run_log(v)}" for k, v in train_log_stats.items()
            )
            log_str += "  [VAL]  " + " | ".join(
                f"{k}: {_fmt_run_log(v)}" for k, v in val_log_stats.items()
            )
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {log_str}\n")
        else:
            if cosine_sched is not None:
                cosine_sched.step()
            for i, g in enumerate(optimizer_d.param_groups):
                train_log_stats["train/lr_g%d" % i] = g["lr"]

            log_str = f"Epoch {epoch:03d}  [TRAIN]  " + " | ".join(
                f"{k}: {_fmt_run_log(v)}" for k, v in train_log_stats.items()
            )
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {log_str}\n")

    torch.save(
        {'state_dict': best_model_wts, 'args': vars(args)},
        os.path.join(args.output_dir, 'best.ckpt'),
    )
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\nTraining finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation RMSE: {best_acc:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FusedHE training script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.distribution_constraint = not getattr(args, 'no_distribution_constraint', False)
    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id+'_'+args.job_name
    if args.resume and os.path.isfile(args.resume):
        args.output_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"Resume: output_dir set to {args.output_dir}")
    else:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)






