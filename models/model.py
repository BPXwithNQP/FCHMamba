import os
import utils.conda_glibcxx  # noqa: F401  # 须在 torch 之前（GLIBCXX / selective_scan）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
#import torhvision
from mmcv.runner import load_checkpoint
from models.mit import mit_b4
from models.resnet101 import Resnet
from models.mambavision_backbone import MambaVisionBackbone
from models.mambavision_seg import MambaVisionTinyBackbone
from models.convnextv2_backbone import ConvNeXtV2Backbone
from models.repvit_backbone import RepViTBackbone
from models.efficient_vmamba_backbone import EfficientVMambaFusedBackbone
from models.groupmamba_backbone import GroupMambaFusedBackbone
from models.fdgm_backbone import FDGMEncoder
from models.Blocks.FAFBlock import FAFBlock
from models.Blocks.nFAF import nFAF
from models.Blocks.liteFAF import liteFAF
from models.Blocks.RFAF import RFAF
from models.Blocks.GRFA import GRFA
from models.Blocks.GSFB import GSFB
from models.Blocks.FDSF import FDSF
from models.Blocks.DMSN import DMSN
from models.Blocks.DPLD import DPLDHead, compute_or_load_dpld_bin_stats
from models.Blocks.TBWH import TBWHHead
from models.Blocks._config import load_block_cfg
from models.Blocks.HLAttention import HLTail
from models.Blocks.FHRM import FHRMRefinement
from utils.encoder_presets import resolve_encoder_args
import random
import numpy as np


def _parse_use_dmsn(args) -> bool:
    raw = getattr(args, "use_DMSN", 0)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def _single_decoder_ff_widths(args):
    sdc = getattr(args, "single_decoder_channels", None)
    if sdc:
        return list(sdc)
    return [1024, 640, 256, 128]


def _mit_ff1_out_channels(args) -> int:
    if bool(getattr(args, "native_proj_ff", False)):
        return int(_single_decoder_ff_widths(args)[0])
    if bool(getattr(args, "native_align_ff", False)):
        sdc = getattr(args, "single_decoder_channels", None)
        if sdc:
            return int(sdc[0])
        return int(args.out_channel_ffb[0])
    return int(args.out_channel_ffb[0]) * 2


def _build_decoder_skip_blocks(args, decoder_in, decoder_out, log_prefix="Decoder"):
    """跳跃融合 block1–3：use_FDSF > use_GSFB > DecoderBlock（mit/conv/fused 共用）。"""
    use_FDSF = bool(getattr(args, "use_FDSF", False))
    use_GSFB = bool(getattr(args, "use_GSFB", False))
    if use_FDSF and use_GSFB:
        print("[{}] use_FDSF=True 覆盖 use_GSFB".format(log_prefix), flush=True)
        use_GSFB = False
    if use_FDSF:
        block1 = FDSF(decoder_in[0], decoder_out[0], stage_idx=3)
        block2 = FDSF(decoder_in[1], decoder_out[1], stage_idx=2)
        block3 = FDSF(decoder_in[2], decoder_out[2], stage_idx=1)
    elif use_GSFB:
        block1 = GSFB(decoder_in[0], decoder_out[0])
        block2 = GSFB(decoder_in[1], decoder_out[1])
        block3 = GSFB(decoder_in[2], decoder_out[2])
    else:
        block1 = DecoderBlock(decoder_in[0], decoder_out[0])
        block2 = DecoderBlock(decoder_in[1], decoder_out[1])
        block3 = DecoderBlock(decoder_in[2], decoder_out[2])
    return use_FDSF, use_GSFB, block1, block2, block3


def build_encoder_branch(name: str, cfg: dict, args):
    """
    根据主干类型与 yaml 合并后的 cfg 构建单支路 encoder（返回 4 级特征 tuple）。
    name: res | mit | convnext | repvit | MambaVision | EfficientVMamba | GroupMamba
    """
    pre = getattr(args, "pre_trained", True)
    wp = cfg.get("weights_path")

    if name in ("res", "res50", "res101"):
        variant = cfg.get("variant", None)
        if name == "res50" or str(variant) in ("50", "res50", "r50"):
            depth = 50
        else:
            depth = 101
        return Resnet(pre_trained=pre, depth=depth, ckpt_path=wp)

    if name == "mit":
        net = mit_b4()
        if pre and wp:
            load_checkpoint(net, wp, logger=None)
            print(
                "[Encoder/mit] 预训练已加载: {}".format(os.path.abspath(wp)),
                flush=True,
            )
        elif pre and not wp:
            raise FileNotFoundError(
                "MiT 需要 weights_path（configs/mit.yaml 或旧版 --mit_ckpt_path / checkpoint args）"
            )
        return net

    if name == "convnext":
        native = bool(cfg.get("native_mode", True))
        return ConvNeXtV2Backbone(
            pretrained=pre,
            ckpt_path=wp,
            no_reshape=native,
        )

    if name == "repvit":
        native = bool(cfg.get("native_mode", True))
        return RepViTBackbone(
            variant=cfg.get("variant", "m1_5"),
            pretrained=pre,
            ckpt_path=wp,
            no_reshape=native,
        )

    if name == "MambaVision":
        variant = cfg.get("variant", "tiny_ade20k")
        nr = bool(cfg.get("native_mode", False))
        if variant == "base_1k":
            return MambaVisionBackbone(
                pretrained=pre, ckpt_path=wp, no_reshape=nr
            )
        return MambaVisionTinyBackbone(
            pretrained=pre, ckpt_path=wp, no_reshape=nr
        )

    if name == "EfficientVMamba":
        return EfficientVMambaFusedBackbone(cfg, args)

    if name == "GroupMamba":
        return GroupMambaFusedBackbone(cfg, args)

    raise ValueError("未知主干类型: {}".format(name))


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

########################## Adaptive Bin Edge（自适应分区边界）############################


class AdaptiveBinEdge(nn.Module):
    """从 Stage4 全局特征预测自适应 bin 宽度，生成 image-level 的 bin_edges 和 centers。
    使用 tanh 有界扰动策略：从均匀分布出发，每个 bin 宽度最多偏离 ±max_perturbation，
    从结构上杜绝 bin 坍缩。"""
    def __init__(self, in_channels, n_bins, max_height=185.0, max_perturbation=0.5):
        super().__init__()
        self.n_bins = n_bins
        self.max_height = max_height
        self.max_perturbation = max_perturbation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, n_bins),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — MambaVision Stage4 特征 (adapter 之后)
        Returns:
            bin_edges:        (B, n_bins+1) — 自适应分区边界（米制），首边=0, 末边=max_height
            centers:          (B, n_bins)   — 各 bin 中心高度（米制）
            bin_widths_normed: (B, n_bins)  — 归一化 bin 宽度，供 loss 做均匀正则化
        """
        feat = x.mean(dim=[2, 3])                                    # GAP → (B, C)
        y = self.mlp(feat)                                            # (B, n_bins)
        delta = torch.tanh(y) * self.max_perturbation                # [-max_pert, +max_pert]
        uniform = 1.0 / self.n_bins
        bin_widths_raw = uniform * (1.0 + delta)                     # 每个 bin ∈ [0.5, 1.5] × uniform
        bin_widths_normed = bin_widths_raw / bin_widths_raw.sum(dim=1, keepdim=True)
        bin_widths = self.max_height * bin_widths_normed             # (B, n_bins), sum=max_height
        bin_widths_pad = F.pad(bin_widths, (1, 0), mode='constant', value=0.0)
        bin_edges = torch.cumsum(bin_widths_pad, dim=1)              # (B, n_bins+1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])      # (B, n_bins)
        return bin_edges, centers, bin_widths_normed


########################## Conditional Cascade Heads（仅 --use_condition 时使用）############################
# bin_num 档、每档 bin_len 米：(0, bin_len], (bin_len, 2*bin_len], ...，中心 (i+0.5)*bin_len


class ConditionalCascadeHeads(nn.Module):
    """条件高度分区头：n_bins 档 bin 概率 + 区间内残差 R。

    pred_d（归一化高度）由 softmax 概率与 (center+R) 加权得到；可选 condition_k>0 时
    每像素仅用概率最大的 k 档（取值仍为原 softmax，再按维度 1 重归一化）参与加权，
    以突出主峰。返回的 bin_prob 仍为全档位 softmax，供损失与其它分支使用。
    """
    def __init__(self, channels_out, n_bins=10, bin_len=6.0, max_height=185.0, condition_k=0):
        super().__init__()
        self.n_bins = n_bins
        self.bin_len = float(bin_len)
        self.max_height = max_height
        self.condition_k = int(condition_k)
        centers = [(i + 0.5) * self.bin_len for i in range(n_bins)]
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.bin_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, n_bins, 1),
        )
        self.res_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, 1, 1),
        )

    def forward(self, x, adaptive_centers=None):
        bin_logits = self.bin_head(x)
        bin_prob = F.softmax(bin_logits, dim=1)
        residual = self.res_head(x)
        if adaptive_centers is not None:
            centers = adaptive_centers.unsqueeze(-1).unsqueeze(-1)    # (B, n_bins, 1, 1)
        else:
            centers = self.centers.view(1, -1, 1, 1)
        prob_h = bin_prob
        k = self.condition_k
        if k > 0:
            k = min(k, int(bin_prob.size(1)))
            topv, topi = torch.topk(bin_prob, k, dim=1)
            prob_h = torch.zeros_like(bin_prob)
            prob_h.scatter_(1, topi, topv)
            prob_h = prob_h / prob_h.sum(dim=1, keepdim=True).clamp(min=1e-8)
        h_meters = (prob_h * (centers + residual)).sum(dim=1, keepdim=True)
        pred_d = (h_meters / self.max_height).clamp(0.0, 1.0)
        return pred_d, bin_logits, bin_prob, residual


class FHE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.option = args.option
        self.enc_opt = args.enc_opt
        self.pre_trained = args.pre_trained
        self.use_condition = getattr(args, 'use_condition', False)
        self.use_adabin = getattr(args, 'use_adabin', False)
        self.use_HL = bool(getattr(args, 'use_HL', False))
        _dpld_mode = getattr(args, 'use_DPLD', 0)
        if isinstance(_dpld_mode, bool):
            _dpld_mode = 1 if _dpld_mode else 0
        else:
            try:
                _dpld_mode = int(_dpld_mode)
            except (TypeError, ValueError):
                _dpld_mode = 0
        if _dpld_mode not in (0, 1, 2):
            _dpld_mode = 1 if _dpld_mode else 0
        self.use_DPLD = _dpld_mode in (1, 2)
        setattr(args, 'use_DPLD', _dpld_mode)

        # ---- TBWH 模式解析（均匀分区无先验，与 DPLD/use_condition 互斥）----
        _tbwh_mode = getattr(args, 'use_TBWH', 0)
        try:
            _tbwh_mode = int(_tbwh_mode)
        except (TypeError, ValueError):
            _tbwh_mode = 0
        self.use_TBWH = bool(_tbwh_mode)
        setattr(args, 'use_TBWH', int(self.use_TBWH))

        _fhrm = getattr(args, "use_FHRM", 0)
        try:
            _fhrm = int(_fhrm)
        except (TypeError, ValueError):
            _fhrm = 0
        self.use_FHRM = _fhrm == 1
        setattr(args, "use_FHRM", int(self.use_FHRM))

        # TBWH 与 DPLD 同坑位；DPLD 优先（同时开时关掉 TBWH）
        if self.use_TBWH and self.use_DPLD:
            print("[FHE] use_DPLD 与 use_TBWH 同时开启，DPLD 优先，自动关闭 TBWH", flush=True)
            self.use_TBWH = False
            args.use_TBWH = 0

        # DPLD 与 use_condition 是同一坑位（高度头）的两种实现；DPLD 优先
        if self.use_DPLD and self.use_condition:
            print(
                "[FHE] 同时开启 use_DPLD 与 use_condition，使用 DPLD 取代 condition_heads "
                "（use_condition 被忽略）",
                flush=True,
            )
            self.use_condition = False
            args.use_condition = False
        # DPLD 与 use_adabin 互斥：bin edges 在 DPLD 内部已是数据集先验，不需要 AdaBins
        if self.use_DPLD and self.use_adabin:
            print("[FHE] use_DPLD 已包含数据集先验 bin，自动关闭 use_adabin", flush=True)
            self.use_adabin = False
            args.use_adabin = False

        # FDGM（FD-RGHF）单分支编码器：最高优先级，忽略 enc_opt / encoder1 / encoder2。
        _fdgm = getattr(args, "use_FDGM", 0)
        if isinstance(_fdgm, bool):
            _fdgm = 1 if _fdgm else 0
        else:
            try:
                _fdgm = 1 if str(_fdgm).strip().lower() not in ("", "0", "false", "no", "off") else 0
            except (TypeError, ValueError):
                _fdgm = 0
        self.use_FDGM = bool(_fdgm)
        setattr(args, "use_FDGM", int(self.use_FDGM))

        resolve_encoder_args(args)
        # resolve_encoder_args 在 use_FDGM 时已将 enc_opt 强制为 'mit'，此处同步本地副本。
        self.enc_opt = args.enc_opt

        if self.use_FDGM:
            print(
                "[FHE] use_FDGM 已启用：忽略 enc_opt/encoder1/encoder2，"
                "以 FD-RGHF 编码器 + MitDecoder 启动",
                flush=True,
            )
            enc = FDGMEncoder(args)
        elif self.enc_opt == "fused":
            enc = FusedEncoder(args)
        elif self.enc_opt == "mit":
            enc = build_encoder_branch(
                args.encoder1_type, args.encoder1_cfg, args
            )
        elif self.enc_opt == "conv":
            enc = build_encoder_branch(
                args.encoder1_type, args.encoder1_cfg, args
            )
        else:
            raise ValueError("enc_opt 应为 mit / conv / fused")

        self.encoder = enc
        channels_out = args.out_decoder
        self.decoder_option = {'mit':MitDecoder(args),
                            'conv': ConvDecoder(args),
                            'fused':FusedDecoder(args)}
        self.decoder = self.decoder_option[self.enc_opt]

        # use_adabin 头与是否走 HL 尾部无关，单独建好
        if self.use_condition and self.use_adabin:
            max_h = getattr(args, 'max_height_eval', 185.0)
            n_bins = getattr(args, 'bin_num', 10)
            ada_in_ch = getattr(args, 'out_channel_ffb', [512, 320, 128, 64])[0]
            max_pert = getattr(args, 'max_perturbation', 0.5)
            self.adabin_head = AdaptiveBinEdge(
                ada_in_ch, n_bins=n_bins, max_height=max_h, max_perturbation=max_pert,
            )

        # ---- 高度头构造 ----
        # 优先级：DPLD > TBWH > use_HL > use_condition > DepthHead
        self.dpld_head = None
        self.tbwh_head = None
        if self.use_DPLD:
            dpld_cfg = load_block_cfg("DPLD")
            max_h = float(getattr(args, 'max_height_eval', 185.0))
            _dpld_nb = getattr(args, 'dpld_n_bins', None)
            n_bins = int(_dpld_nb if _dpld_nb is not None else dpld_cfg.get('n_bins', 128))
            _dpld_gm = getattr(args, 'dpld_ground_bin_max_m', None)
            ground_max_m = float(
                _dpld_gm if _dpld_gm is not None else dpld_cfg.get('ground_bin_max_m', 1.0)
            )
            _dpld_fb = getattr(args, 'dpld_fallback', None)
            fallback = str(_dpld_fb if _dpld_fb is not None else dpld_cfg.get('fallback', 'uniform'))
            cache_name = str(dpld_cfg.get('cache_filename', 'dpld_bin_edges.pt'))
            stats = compute_or_load_dpld_bin_stats(
                args, n_bins=n_bins, ground_bin_max_m=ground_max_m,
                max_height=max_h, fallback=fallback, cache_filename=cache_name,
                online_rescan=(_dpld_mode == 2),
            )
            self.dpld_head = DPLDHead(
                channels_in=channels_out, max_height=max_h,
                bin_edges_m=stats["bin_edges_m"],
                bin_prior_probs=stats["bin_prior_probs"],
                n_bins=n_bins, ground_bin_max_m=ground_max_m,
            )

        if self.use_TBWH:
            tbwh_cfg = load_block_cfg("TBWH")
            # 根据 dataset_name 选对应的 n_bins / max_height
            ds = str(getattr(args, 'dataset_name', '') or '').lower()
            if 'vhg' in ds:
                tbwh_n_bins = int(tbwh_cfg.get('vhg_n_bins', 64))
                tbwh_max_h  = float(tbwh_cfg.get('vhg_max_height', 20.5))
            elif '2019' in ds or '19' in ds:
                tbwh_n_bins = int(tbwh_cfg.get('19_n_bins', 128))
                tbwh_max_h  = float(tbwh_cfg.get('19_max_height', 76.0))
            else:
                # 默认 dfc2023
                tbwh_n_bins = int(tbwh_cfg.get('23_n_bins', 128))
                tbwh_max_h  = float(tbwh_cfg.get('23_max_height', 81.5))
            print(
                "[FHE] TBWH: dataset_name={!r} → n_bins={}, max_height={}m".format(
                    ds, tbwh_n_bins, tbwh_max_h
                ),
                flush=True,
            )
            self.tbwh_head = TBWHHead(
                channels_in=channels_out,
                n_bins=tbwh_n_bins,
                max_height=tbwh_max_h,
            )

        if self.use_HL:
            # 整尾部交给 HLTail 接管：解耦 + 边界注意力 + 区域门控 + 分割头 + 高度头(+ 地表软钳)
            # 强制要求 option=='full'：HLTail 必须同时输出分割与高度
            if self.option != "full":
                print("[FHE] use_HL=True 自动将 option 由 '{}' 切换为 'full'".format(self.option))
                self.option = "full"
            self.hl_tail = HLTail(args, channels_out=channels_out, dpld_head=self.dpld_head)
            # 兼容 state_dict / 旧代码：保留 None 占位，禁止误用
            self.condition_heads = None
            self.depth_head = None
            self.segment_head = None
        else:
            self.hl_tail = None
            if self.use_DPLD or self.use_TBWH:
                # DPLD / TBWH 接管高度头，condition_heads 不再需要
                self.condition_heads = None
                self.depth_head = None
            elif self.use_condition:
                max_h = getattr(args, 'max_height_eval', 185.0)
                n_bins = getattr(args, 'bin_num', 10)
                bin_len = getattr(args, 'bin_len', 6.0)
                _ck = int(getattr(args, 'condition_k', 0))
                self.condition_heads = ConditionalCascadeHeads(
                    channels_out,
                    n_bins=n_bins,
                    bin_len=bin_len,
                    max_height=max_h,
                    condition_k=_ck,
                )
                self.depth_head = None
            else:
                self.condition_heads = None
                self.depth_head = DepthHead(channels_out=channels_out)

            if self.option == "full":
                self.segment_head = SegmentHead(
                    channels_out=channels_out, num_classes=args.num_classes,
                )

        self.fhrm = None
        if getattr(self, "use_FHRM", False):
            if self.use_HL:
                print("[FHE] use_HL 已启用，FHRM 跳过（高度路径由 HLTail 接管）", flush=True)
                self.use_FHRM = False
                args.use_FHRM = 0
            elif self.use_DPLD or self.use_TBWH or self.use_condition:
                print(
                    "[FHE] FHRM 仅支持原版 DepthHead 路径；当前为 DPLD/TBWH/use_condition，已跳过 FHRM",
                    flush=True,
                )
                self.use_FHRM = False
                args.use_FHRM = 0
            else:
                self.fhrm = FHRMRefinement(channels_in=channels_out)

    def forward(self, x, faf_capture=None):
        enc_out = self.encoder(x)

        if self.enc_opt == "fused":
            dec_final = self.decoder(enc_out, faf_capture=faf_capture)
        else:
            dec_final = self.decoder(enc_out)

        # AdaBins 与是否走 HL 尾部正交，独立计算
        adaptive_centers = None
        bin_edges = None
        bin_widths_normed = None
        if self.use_condition and self.use_adabin and hasattr(self, 'adabin_head'):
            mit_feats = enc_out['mit'] if isinstance(enc_out, dict) else enc_out
            stage4_feat = mit_feats[3] if isinstance(mit_feats, (tuple, list)) else mit_feats
            bin_edges, adaptive_centers, bin_widths_normed = self.adabin_head(stage4_feat)

        if self.use_HL and self.hl_tail is not None:
            data = self.hl_tail(dec_final, adaptive_centers=adaptive_centers)
            if bin_edges is not None:
                data['bin_edges'] = bin_edges
                data['adaptive_centers'] = adaptive_centers
                data['bin_widths_normed'] = bin_widths_normed
            return data

        pred_s = None
        if self.option == "full" and getattr(self, "segment_head", None) is not None:
            pred_s = self.segment_head(dec_final)

        # DPLD / TBWH 替换 condition_heads / depth_head 的整条高度路径
        if self.use_DPLD and self.dpld_head is not None:
            out_d, bin_logits, bin_prob = self.dpld_head(dec_final)
            data = {
                'pred_d': out_d,
                'bin_logits': bin_logits,
                'bin_prob': bin_prob,
                'bin_edges_m': self.dpld_head.bin_edges_m,
                'bin_centers_m': self.dpld_head.bin_centers_m,
            }
        elif self.use_TBWH and self.tbwh_head is not None:
            out_d, bin_logits, bin_prob = self.tbwh_head(dec_final)
            data = {
                'pred_d': out_d,
                'bin_logits': bin_logits,
                'bin_prob': bin_prob,
                'bin_edges_m': self.tbwh_head.bin_edges_m,
                'bin_centers_m': self.tbwh_head.bin_centers_m,
            }
        elif self.use_condition and self.condition_heads is not None:
            out_d, bin_logits, bin_prob, residual = self.condition_heads(dec_final, adaptive_centers)
            data = {'pred_d': out_d, 'bin_logits': bin_logits, 'bin_prob': bin_prob, 'residual': residual}
            if bin_edges is not None:
                data['bin_edges'] = bin_edges
                data['adaptive_centers'] = adaptive_centers
                data['bin_widths_normed'] = bin_widths_normed
        else:
            out_d = self.depth_head(dec_final)
            if self.fhrm is not None:
                seg_for_fhrm = pred_s if self.option == "full" else None
                h_base_aux = out_d
                out_d = self.fhrm(dec_final, out_d, seg_for_fhrm)
                data = {'pred_d': out_d, 'pred_d_base': h_base_aux}
            else:
                data = {'pred_d': out_d}
        if self.option == "full":
            data["pred_s"] = pred_s
        return data

        
class DepthHead(nn.Module):
    def __init__(self,channels_out) -> None:
        super().__init__()
        self.depth_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    
    def forward(self,x):
        out = self.depth_head(x)

        return out
    

class SegmentHead(nn.Module):
    def __init__(self,channels_out,num_classes) -> None:
        super().__init__()

        self.segment_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(channels_out, num_classes, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self,x):
        out = self.segment_head(x)

        return out


########################## FUSED MODEL ############################
class FusedEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pre_trained = args.pre_trained
        resolve_encoder_args(args)
        # encoder1 → resnet 支路（旧 state_dict 键名 encoder.resnet）；encoder2 → mit 支路（encoder.mit）
        self.resnet = build_encoder_branch(
            args.encoder1_type, args.encoder1_cfg, args
        )
        self.mit = build_encoder_branch(
            args.encoder2_type, args.encoder2_cfg, args
        )

    def forward(self, x):
        mit = self.mit(x)
        conv = self.resnet(x)
        data = {'mit': mit, 'conv': conv}
        return data


class FusedDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()    
        self.ffb_in = args.in_channel_ffb
        self.ffb_out = args.out_channel_ffb

        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb

        self.out_channel = args.out_decoder

        # 融合模块选择优先级：use_GRFA > use_liteFAF > use_RFAF > use_nFAF > use_FAF > FeatureFusion。
        # 约定 ff1 融合 stage4（最深，stage_idx=4），ff4 融合 stage1（最浅，stage_idx=1）。
        #   - use_GRFA   ：FeatureFusion 同构 base + gated high-frequency residual，参数读 block.yaml.GRFA
        #   - use_liteFAF：显式分频 + 带内残差注入（轻量），参数读 block.yaml.liteFAF
        #   - use_RFAF  ：原版 FeatureFusion + 阶段化 DWT 频域残差，参数读 block.yaml.RFAF
        #   - use_nFAF  ：FAFBlock 改进版 v2（多层 3×3 语义对齐 + 残差门 + cat + 深度 refine），参数读 block.yaml.nFAF
        #   - use_FAF   ：原始 FAFBlock（1×1 对齐 + 加性融合）
        use_GRFA = bool(getattr(args, 'use_GRFA', False))
        use_liteFAF = bool(getattr(args, 'use_liteFAF', False))
        use_RFAF = bool(getattr(args, 'use_RFAF', False))
        use_nFAF = bool(getattr(args, 'use_nFAF', False))
        use_FAF = bool(getattr(args, 'use_FAF', False))
        if use_GRFA and (use_liteFAF or use_RFAF or use_nFAF or use_FAF):
            print("[FusedDecoder] use_GRFA=True 覆盖 use_liteFAF/use_RFAF/use_nFAF/use_FAF", flush=True)
            use_liteFAF = False
            use_RFAF = False
            use_nFAF = False
            use_FAF = False
        if use_liteFAF and (use_RFAF or use_nFAF or use_FAF):
            print("[FusedDecoder] use_liteFAF=True 覆盖 use_RFAF/use_nFAF/use_FAF", flush=True)
            use_RFAF = False
            use_nFAF = False
            use_FAF = False
        if use_RFAF and (use_nFAF or use_FAF):
            print("[FusedDecoder] use_RFAF=True 覆盖 use_nFAF/use_FAF", flush=True)
            use_nFAF = False
            use_FAF = False
        if use_nFAF and use_FAF:
            print("[FusedDecoder] use_nFAF=True 覆盖 use_FAF", flush=True)
            use_FAF = False
        self.use_GRFA = use_GRFA
        self.use_liteFAF = use_liteFAF
        self.use_RFAF = use_RFAF
        self.use_nFAF = use_nFAF
        self.use_FAF = use_FAF
        # 是否为「接受 capture 参数的频率融合模块」（FAFBlock / nFAF / liteFAF / RFAF / GRFA 共用 capture 接口）
        self.faf_like = use_GRFA or use_liteFAF or use_RFAF or use_nFAF or use_FAF
        if use_GRFA:
            self.ff1 = GRFA(self.ffb_in[0], self.ffb_out[0], stage_idx=4)
            self.ff2 = GRFA(self.ffb_in[1], self.ffb_out[1], stage_idx=3)
            self.ff3 = GRFA(self.ffb_in[2], self.ffb_out[2], stage_idx=2)
            self.ff4 = GRFA(self.ffb_in[3], self.ffb_out[3], stage_idx=1)
        elif use_liteFAF:
            # 细粒度参数全部走 block.yaml 的 liteFAF 段，这里只传 stage_idx。
            self.ff1 = liteFAF(self.ffb_in[0], self.ffb_out[0], stage_idx=4)
            self.ff2 = liteFAF(self.ffb_in[1], self.ffb_out[1], stage_idx=3)
            self.ff3 = liteFAF(self.ffb_in[2], self.ffb_out[2], stage_idx=2)
            self.ff4 = liteFAF(self.ffb_in[3], self.ffb_out[3], stage_idx=1)
        elif use_RFAF:
            self.ff1 = RFAF(self.ffb_in[0], self.ffb_out[0], stage_idx=4)
            self.ff2 = RFAF(self.ffb_in[1], self.ffb_out[1], stage_idx=3)
            self.ff3 = RFAF(self.ffb_in[2], self.ffb_out[2], stage_idx=2)
            self.ff4 = RFAF(self.ffb_in[3], self.ffb_out[3], stage_idx=1)
        elif use_nFAF:
            # 细粒度参数全部走 block.yaml 的 nFAF 段，这里只传 stage_idx。
            self.ff1 = nFAF(self.ffb_in[0], self.ffb_out[0], stage_idx=4)
            self.ff2 = nFAF(self.ffb_in[1], self.ffb_out[1], stage_idx=3)
            self.ff3 = nFAF(self.ffb_in[2], self.ffb_out[2], stage_idx=2)
            self.ff4 = nFAF(self.ffb_in[3], self.ffb_out[3], stage_idx=1)
        elif use_FAF:
            faf_kwargs = dict(
                smooth_kernel=int(getattr(args, 'faf_smooth_kernel', 3)),
                lambda_res=float(getattr(args, 'faf_lambda_res', 0.5)),
                learnable_coef=bool(getattr(args, 'faf_learnable_coef', True)),
            )
            self.ff1 = FAFBlock(self.ffb_in[0], self.ffb_out[0], stage_idx=4, **faf_kwargs)
            self.ff2 = FAFBlock(self.ffb_in[1], self.ffb_out[1], stage_idx=3, **faf_kwargs)
            self.ff3 = FAFBlock(self.ffb_in[2], self.ffb_out[2], stage_idx=2, **faf_kwargs)
            self.ff4 = FAFBlock(self.ffb_in[3], self.ffb_out[3], stage_idx=1, **faf_kwargs)
        else:
            self.ff1 = FeatureFusion(self.ffb_in[0], self.ffb_out[0])
            self.ff2 = FeatureFusion(self.ffb_in[1], self.ffb_out[1])
            self.ff3 = FeatureFusion(self.ffb_in[2], self.ffb_out[2])
            self.ff4 = FeatureFusion(self.ffb_in[3], self.ffb_out[3])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        raw_use_dmsn = getattr(args, "use_DMSN", 0)
        if isinstance(raw_use_dmsn, bool):
            use_DMSN = raw_use_dmsn
        else:
            use_DMSN = str(raw_use_dmsn).strip().lower() not in ("", "0", "false", "no", "off")
        self.use_DMSN = use_DMSN
        self.dmsn = DMSN(self.decoder_in[0]) if use_DMSN else nn.Identity()

        # 跳跃融合 block 的选择优先级：use_FDSF > use_GSFB > DecoderBlock。
        self.use_FDSF, self.use_GSFB, self.block1, self.block2, self.block3 = (
            _build_decoder_skip_blocks(
                args, self.decoder_in, self.decoder_out, log_prefix="FusedDecoder"
            )
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.decoder_out[2],
                      out_channels=self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.use_activation_checkpoint = bool(getattr(args, "activation_checkpoint", False))

    def _maybe_checkpoint(self, fn, *args):
        if self.use_activation_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, data, return_intermediates=False, faf_capture=None):
        mit = data['mit']
        conv = data['conv']
        mit_1, mit_2, mit_3, mit_4 = mit
        conv_1, conv_2, conv_3, conv_4 = conv

        if faf_capture is not None:
            if not self.faf_like:
                raise RuntimeError("faf_capture 仅在使用 FAFBlock/nFAF/liteFAF/RFAF/GRFA 时可用")
            faf_capture.clear()
            for name in ("ff1", "ff2", "ff3", "ff4"):
                faf_capture[name] = {}

        c1 = faf_capture["ff1"] if faf_capture else None
        c2 = faf_capture["ff2"] if faf_capture else None
        c3 = faf_capture["ff3"] if faf_capture else None
        c4 = faf_capture["ff4"] if faf_capture else None

        if self.faf_like:
            def _ff1_fwd(m4, c4):
                return self.ff1(m4, c4, capture=c1)
            ff1_ = self._maybe_checkpoint(_ff1_fwd, mit_4, conv_4)
        else:
            ff1_ = self._maybe_checkpoint(self.ff1, mit_4, conv_4)
        if self.use_DMSN:
            ff1_ = self._maybe_checkpoint(self.dmsn, ff1_)
        ff1_ = self.upsample(ff1_)
        if self.faf_like:
            def _ff2_fwd(m3, c3):
                return self.ff2(m3, c3, capture=c2)
            ff2_ = self._maybe_checkpoint(_ff2_fwd, mit_3, conv_3)
        else:
            ff2_ = self._maybe_checkpoint(self.ff2, mit_3, conv_3)
        if self.faf_like:
            def _ff3_fwd(m2, c2_in):
                return self.ff3(m2, c2_in, capture=c3)
            ff3_ = self._maybe_checkpoint(_ff3_fwd, mit_2, conv_2)
        else:
            ff3_ = self._maybe_checkpoint(self.ff3, mit_2, conv_2)
        if self.faf_like:
            def _ff4_fwd(m1, c1_in):
                return self.ff4(m1, c1_in, capture=c4)
            ff4_ = self._maybe_checkpoint(_ff4_fwd, mit_1, conv_1)
        else:
            ff4_ = self._maybe_checkpoint(self.ff4, mit_1, conv_1)

        def _block1_fwd(f1, f2):
            return self.block1(f1, f2)
        def _block2_fwd(x_in, f3):
            return self.block2(x_in, f3)
        def _block3_fwd(x_in, f4):
            return self.block3(x_in, f4)

        x = self._maybe_checkpoint(_block1_fwd, ff1_, ff2_)
        x = self._maybe_checkpoint(_block2_fwd, x, ff3_)
        dec_d2 = x
        x = self._maybe_checkpoint(_block3_fwd, x, ff4_)
        dec_d3 = x
        x = self.conv1(x)
        x = self.upsample(x)

        if return_intermediates:
            return x, [dec_d2, dec_d3]
        return x


class FeatureFusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel/2),
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channel*2),
                      out_channels=int(out_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channel*2)),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channel*2),
                      out_channels=int(out_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channel*2)),
            nn.ReLU())

    def forward(self, x_mit, x_cnn):
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x = torch.cat((x_mit,x_cnn),1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2,
                      out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self,ffi_1,ffi_2):
        ffi_1 = self.conv1(ffi_1)
        x = torch.cat((ffi_1,ffi_2),1)
        x = self.conv2(x)
        x = self.upsample(x)

        return x
    

class Fused_Conv(nn.Module):
    def __init__(self, in_channel ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel)))

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x
    

########################## Conv MODEL ############################
class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel/2), 
                      out_channels=int(in_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel/2)))

        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffb_in = args.in_channel_ffb
        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb
        self.out_channel = args.out_decoder

        self.ff1 = ConvDecoderBlock(self.ffb_in[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ff2 = ConvDecoderBlock(self.ffb_in[1])
        self.ff3 = ConvDecoderBlock(self.ffb_in[2])
        self.ff4 = ConvDecoderBlock(self.ffb_in[3])

        self.use_DMSN = _parse_use_dmsn(args)
        self.dmsn = (
            DMSN(int(args.in_channel_decoderb[0])) if self.use_DMSN else nn.Identity()
        )
        self.use_activation_checkpoint = bool(getattr(args, "activation_checkpoint", False))

        self.use_FDSF, self.use_GSFB, self.block1, self.block2, self.block3 = (
            _build_decoder_skip_blocks(
                args, self.decoder_in, self.decoder_out, log_prefix="ConvDecoder"
            )
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.decoder_out[2],
                      out_channels=self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)



    def _maybe_checkpoint(self, fn, *args):
        if self.use_activation_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = x
        ff1_ = self.ff1(x_4)
        if self.use_DMSN:
            ff1_ = self._maybe_checkpoint(self.dmsn, ff1_)
        ff1_ = self.upsample(ff1_)
        ff2_ = self.ff2(x_3)
        ff3_ = self.ff3(x_2)
        ff4_ = self.ff4(x_1)

        x = self.block1(ff1_, ff2_)
        x = self.block2(x, ff3_)
        x = self.block3(x, ff4_)
        x = self.conv1(x)
        x = self.upsample(x)

        return x
    

################################# MiT Model ##########################################
class MitDecoderBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, 
                      out_channels=int(in_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel*2)),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2), 
                      out_channels=int(in_channel*2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel*2)))

        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x


class NativeProjFFBlock(nn.Module):
    """旧版 GroupMamba 单主干 decoder ff：1×1 投影到固定宽度 + 同通道 ConvTranspose（dfc23-GM 结构）。"""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(in_channel), int(out_channel), kernel_size=1, bias=True),
            nn.BatchNorm2d(int(out_channel)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                int(out_channel), int(out_channel), kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(int(out_channel)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class NativeAlignFFBlock(nn.Module):
    """GroupMamba 等原生通道 ff：1×1 通道对齐 + 3×3 refine（与部分 VHG 单主干权重一致）。"""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(in_channel), int(out_channel), kernel_size=1, bias=True),
            nn.BatchNorm2d(int(out_channel)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_channel), int(out_channel), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(out_channel)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MitDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffb_out = args.out_channel_ffb
        self.decoder_in = args.in_channel_decoderb
        self.decoder_out = args.out_channel_decoderb
        self.native_proj_ff = bool(getattr(args, "native_proj_ff", False))
        self.native_align_ff = bool(getattr(args, "native_align_ff", False))
        if getattr(args, "single_decoder_channels", None) and not (
            self.native_proj_ff or self.native_align_ff
        ):
            # dfc23-GM 旧结构：single_decoder 宽度 + 1×1/3×3 Conv2d ff（非 ConvTranspose）
            self.native_align_ff = True
            args.native_align_ff = True

        if self.native_proj_ff:
            in_ff = args.out_channel_ffb
            out_ff = _single_decoder_ff_widths(args)
            self.ff1 = NativeProjFFBlock(in_ff[0], out_ff[0])
            self.ff2 = NativeProjFFBlock(in_ff[1], out_ff[1])
            self.ff3 = NativeProjFFBlock(in_ff[2], out_ff[2])
            self.ff4 = NativeProjFFBlock(in_ff[3], out_ff[3])
        elif self.native_align_ff:
            in_ff = args.out_channel_ffb
            sdc = getattr(args, "single_decoder_channels", None)
            out_ff = _single_decoder_ff_widths(args) if sdc else args.out_channel_ffb
            self.ff1 = NativeAlignFFBlock(in_ff[0], out_ff[0])
            self.ff2 = NativeAlignFFBlock(in_ff[1], out_ff[1])
            self.ff3 = NativeAlignFFBlock(in_ff[2], out_ff[2])
            self.ff4 = NativeAlignFFBlock(in_ff[3], out_ff[3])
        else:
            self.ff1 = MitDecoderBlock(self.ffb_out[0])
            self.ff2 = MitDecoderBlock(self.ffb_out[1])
            self.ff3 = MitDecoderBlock(self.ffb_out[2])
            self.ff4 = MitDecoderBlock(self.ffb_out[3])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.use_DMSN = _parse_use_dmsn(args)
        self.dmsn = DMSN(_mit_ff1_out_channels(args)) if self.use_DMSN else nn.Identity()
        self.use_activation_checkpoint = bool(getattr(args, "activation_checkpoint", False))

        self.use_FDSF, self.use_GSFB, self.block1, self.block2, self.block3 = (
            _build_decoder_skip_blocks(
                args, self.decoder_in, self.decoder_out, log_prefix="MitDecoder"
            )
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)



    def _maybe_checkpoint(self, fn, *args):
        if self.use_activation_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = x
        ff1_ = self.ff1(x_4)
        if self.use_DMSN:
            ff1_ = self._maybe_checkpoint(self.dmsn, ff1_)
        ff1_ = self.upsample(ff1_)
        ff2_ = self.ff2(x_3)
        ff3_ = self.ff3(x_2)
        ff4_ = self.ff4(x_1)

        x = self.block1(ff1_,ff2_)
        x = self.block2(x,ff3_)
        x = self.block3(x,ff4_)
        x = self.conv1(x)
        x = self.upsample(x)

        return x