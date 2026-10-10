import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence

try:
    from utils.prob_losses import gaussian_loss
except Exception:
    try:
        from .prob_losses import gaussian_loss
    except Exception:
        gaussian_loss = None

try:
    from models.Blocks._config import load_block_cfg as _load_block_cfg
except Exception:
    _load_block_cfg = None

# 条件分区档位由 args.bin_num、args.bin_len 决定，与 model 一致


def _to_bhw(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) -> (B,H,W)；勿用无参 squeeze，否则 batch=1 时会丢掉 B 维。"""
    if x.dim() == 4 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def focal_loss_with_weights(logits, targets, weight_per_class=None, gamma=2.0, reduction='mean', ignore_index=-100):
    """Focal loss，支持 per-class 权重（反频率）。logits (B,C,H,W), targets (B,H,W) long."""
    if logits.dim() == 4:
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
    N, C = logits.shape
    targets = targets.view(-1)
    valid = targets.ne(ignore_index)
    if valid.sum() == 0:
        return logits.new_zeros(1)
    logits = logits[valid]
    targets = targets[valid]
    log_p = F.log_softmax(logits, dim=1)
    pt = log_p.gather(1, targets.unsqueeze(1)).exp().squeeze(1)
    fl = -((1 - pt).pow(gamma) * log_p.gather(1, targets.unsqueeze(1)).squeeze(1))
    if weight_per_class is not None:
        w = weight_per_class.to(fl.device)[targets]
        fl = fl * w
    return fl.mean() if reduction == 'mean' else fl.sum()


class BinsChamferLoss(nn.Module):
    """Chamfer distance between predicted bin centers and GT depth values (AdaBins style)."""
    def __init__(self, min_val=1e-8):
        super().__init__()
        self.min_val = min_val

    def forward(self, bin_edges, target_depth_maps):
        """
        bin_edges: (B, n_bins+1) — adaptive bin edges in meters
        target_depth_maps: (B, H, W) or (B, 1, H, W) — GT depth in meters
        """
        bin_centers = 0.5 * (bin_edges[:, 1:] + bin_edges[:, :-1])   # (B, n_bins)
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)

        target_points = target_depth_maps.flatten(1)
        mask = target_points.ge(self.min_val)
        target_lengths = mask.sum(dim=1).long()
        target_list = [tp[m] for tp, m in zip(target_points, mask)]
        target_padded = pad_sequence(target_list, batch_first=True).unsqueeze(2)

        # pairwise squared distance
        diff = input_points - target_padded.transpose(1, 2)     # (B, p, T)
        dist2 = diff ** 2

        loss_a = dist2.min(dim=2)[0].mean()
        valid_mask = torch.arange(target_padded.size(1), device=dist2.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        dist2_t = dist2.transpose(1, 2)                          # (B, T, p)
        min_over_p = dist2_t.min(dim=2)[0]                       # (B, T)
        min_over_p = min_over_p * valid_mask.float()
        loss_b = min_over_p.sum() / valid_mask.float().sum().clamp(min=1)
        return loss_a + loss_b


class DepthSegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lambd = args.lambd
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.option = args.option
        self.use_condition = getattr(args, 'use_condition', False)
        self.use_adabin = getattr(args, 'use_adabin', False)
        self.distribution_constraint = getattr(args, 'distribution_constraint', True)
        self.dist_constraint_weight = getattr(args, 'dist_constraint_weight', 0.1)
        self.max_height = getattr(args, 'max_height_eval', 185)
        self.min_height_norm = getattr(args, 'min_height_eval', 1e-4) / max(1.0, float(getattr(args, 'max_height_eval', 185)))
        self.bin_weight = getattr(args, 'bin_weight', 0.15)
        self.height_weight = getattr(args, 'height_weight', 0.85)
        self.scale_height = getattr(args, 'scale_height', 1.0)
        self.de_normal = getattr(args, 'de_normal', False)
        # 0：loss 用原始 pred_d；>0：反归一化到米后 <thr 置 0 再参与 depth / L_height（旧 ckpt 无此字段视为 0）
        self.mask_loss = float(getattr(args, 'mask_loss', 0.0))
        self.bin_num = getattr(args, 'bin_num', 10)
        self.bin_len = getattr(args, 'bin_len', 6.0)
        self._centers = [(i + 0.5) * self.bin_len for i in range(self.bin_num)]
        self.chamfer_weight = getattr(args, 'chamfer_weight', 0.1)
        self.uniform_weight = getattr(args, 'uniform_weight', 0.1)
        if self.use_adabin:
            self.chamfer_loss = BinsChamferLoss()

        # ---- HL 尾部地表软钳损失 ----
        # 仅当 args.use_HL=True 且 use_ground_cap=True 时生效；
        # 监督 (1 - segment_gt) 区域里 h_pred ≤ ground_cap_m。
        self.use_HL = bool(getattr(args, 'use_HL', False))
        self.use_ground_cap = bool(getattr(args, 'use_ground_cap', True))
        self.ground_cap_m = float(getattr(args, 'ground_cap_m', 1.0))
        self.ground_cap_weight = float(getattr(args, 'ground_cap_weight', 0.1))

        # ---- DPLD（数据集先验 binEdge + Top-k + 高斯软标签）----
        # 与 use_condition 互斥；FHE 内部会自动关掉 use_condition / use_adabin。
        # 这里参数全部走 block.yaml 的 DPLD 段（命令行未传时）。
        self.use_DPLD = bool(getattr(args, 'use_DPLD', False))
        if self.use_DPLD:
            dpld_cfg = _load_block_cfg("DPLD") if _load_block_cfg is not None else {}
            self.dpld_dist_weight = float(getattr(
                args, 'dpld_dist_weight', dpld_cfg.get('dist_loss_weight', 0.1)
            ))
            self.dpld_bin_weight = float(getattr(
                args, 'dpld_bin_weight', dpld_cfg.get('bin_loss_weight', 0.15)
            ))
            _dpld_nb = getattr(args, 'dpld_n_bins', None)
            self.dpld_n_bins = int(_dpld_nb if _dpld_nb is not None else dpld_cfg.get('n_bins', 128))
            self.dpld_sigma_m = float(getattr(args, 'dpld_sigma_m', dpld_cfg.get('sigma_m', 1.5)))

        # ---- FHRM 辅助监督（H_base 的 MSE，权重从 block.yaml FHRM.aux_base_weight 读）----
        # 仅当 FHE 在使用 FHRM 时才向 preds 写 pred_d_base；这里按 key 存在与否判断
        self.use_FHRM = bool(getattr(args, 'use_FHRM', 0))
        if self.use_FHRM:
            fhrm_cfg = _load_block_cfg("FHRM") if _load_block_cfg is not None else {}
            self.fhrm_aux_base_weight = float(fhrm_cfg.get('aux_base_weight', 0.1))
        else:
            self.fhrm_aux_base_weight = 0.0

        # ---- TBWH（均匀分区 + 可选 KL）----
        # 超参走 block.yaml TBWH；constraint=false 时不加 KL，仅 focal bin。
        self.use_TBWH = bool(getattr(args, 'use_TBWH', False))
        self.tbwh_constraint = True
        if self.use_TBWH:
            tbwh_cfg = _load_block_cfg("TBWH") if _load_block_cfg is not None else {}
            self.dpld_dist_weight = float(tbwh_cfg.get('dist_loss_weight', 0.1))
            self.dpld_bin_weight = float(getattr(
                args, 'dpld_bin_weight', tbwh_cfg.get('bin_loss_weight', 0.15)
            ))
            self.dpld_sigma_m = float(tbwh_cfg.get('sigma_m', 1.5))
            self.tbwh_constraint = bool(tbwh_cfg.get('constraint', True))
            self.dpld_n_bins = 128

    def _apply_mask_loss(self, pred_d: torch.Tensor) -> torch.Tensor:
        """米制阈值：<thr 置 0，≥thr 保留；与 test.py --mask_test=2 一致。pred_d 为 [0,1] 归一化高度。"""
        thr = float(self.mask_loss)
        if thr <= 0:
            return pred_d
        mh = max(float(self.max_height), 1e-6)
        pred_m = pred_d * mh
        pred_m = torch.where(pred_m < thr, torch.zeros_like(pred_m), pred_m)
        return pred_m / mh

    def forward(self, preds, depth_gt, segment_gt=None):
        depth_gt = _to_bhw(depth_gt)
        pred_d = self._apply_mask_loss(_to_bhw(preds["pred_d"]))
        if self.option == "depth":
            pred_s = None
            if self.de_normal:
                pred_d_denorm = pred_d * self.max_height
                depth_gt_denorm = depth_gt * self.max_height
                loss_d = self.mse_loss(pred_d_denorm, depth_gt_denorm)
            else:
                loss_d = self.mse_loss(pred_d, depth_gt)
            loss_s = torch.tensor(0.0, device=pred_d.device)
        else:
            pred_s = _to_bhw(preds["pred_s"])
            if segment_gt is not None:
                segment_gt = _to_bhw(segment_gt)
            if self.de_normal:
                pred_d_denorm = pred_d * self.max_height
                depth_gt_denorm = depth_gt * self.max_height
                loss_d = self.mse_loss(pred_d_denorm, depth_gt_denorm)
            else:
                loss_d = self.mse_loss(pred_d, depth_gt)
            loss_s = self.seg_loss(pred_s, segment_gt)

        loss_bin = preds['pred_d'].new_tensor(0.0)
        loss_prob = preds['pred_d'].new_tensor(0.0)

        if self.use_condition and 'bin_logits' in preds and 'residual' in preds:
            depth_gt_2d = depth_gt.squeeze() if depth_gt.dim() == 3 else depth_gt
            if depth_gt_2d.dim() == 2:
                depth_gt_2d = depth_gt_2d.unsqueeze(0)
            h_gt = depth_gt_2d * self.max_height
            valid = depth_gt_2d.ge(self.min_height_norm)
            if valid.sum() == 0:
                loss_cond = preds['pred_d'].new_tensor(0.0)
            else:
                bin_logits = preds['bin_logits']
                residual = preds['residual'].squeeze(1)
                pred_d_2d = pred_d
                h_pred = pred_d_2d * self.max_height

                if self.use_adabin and 'bin_edges' in preds:
                    bin_edges = preds['bin_edges']                    # (B, n_bins+1) meters
                    adaptive_centers = preds['adaptive_centers']      # (B, n_bins) meters

                    # bin_gt: 根据自适应 edges 确定每个像素属于哪个 bin
                    be_lo = bin_edges[:, :-1, None, None]             # (B, n_bins, 1, 1)
                    be_hi = bin_edges[:, 1:, None, None]
                    h_gt_exp = h_gt.unsqueeze(1)                      # (B, 1, H, W)
                    bin_ind = (h_gt_exp > be_lo) & (h_gt_exp <= be_hi)  # (B, n_bins, H, W)
                    # 对 h_gt==0 且 bin 0 的下界是 0 的情况，补充到 bin 0
                    bin_ind[:, 0] = bin_ind[:, 0] | ((h_gt_exp.squeeze(1) >= 0) & (h_gt_exp.squeeze(1) <= be_hi[:, 0]))
                    bin_gt = bin_ind.float().argmax(dim=1)            # (B, H, W)
                    bin_gt = torch.where(valid, bin_gt, bin_gt.new_full(bin_gt.shape, -100, dtype=torch.long))

                    counts = torch.bincount(bin_gt[valid].view(-1), minlength=self.bin_num).float().clamp(min=1)
                    inv_freq = 1.0 / counts
                    weight_bin = inv_freq / inv_freq.sum() * self.bin_num
                    L_bin = focal_loss_with_weights(bin_logits, bin_gt, weight_per_class=weight_bin, gamma=2.0, ignore_index=-100)

                    # 残差监督：R_gt = h_gt - center_of_gt_bin
                    centers_4d = adaptive_centers.unsqueeze(-1).unsqueeze(-1)   # (B, n_bins, 1, 1)
                    bin_gt_clamped = bin_gt.clamp(0, self.bin_num - 1)
                    bin_gt_onehot = F.one_hot(bin_gt_clamped, self.bin_num).permute(0, 3, 1, 2).float()
                    center_at_gt = (centers_4d * bin_gt_onehot).sum(dim=1)      # (B, H, W)
                    # 自适应 bin 宽度不固定，用当前 bin 宽度做 clamp
                    bin_widths = bin_edges[:, 1:] - bin_edges[:, :-1]            # (B, n_bins)
                    width_at_gt = (bin_widths.unsqueeze(-1).unsqueeze(-1) * bin_gt_onehot).sum(dim=1)
                    R_gt = (h_gt - center_at_gt).clamp(-width_at_gt, width_at_gt)
                    R_gt = torch.where(valid, R_gt, R_gt.new_zeros(1))

                    L_res = (torch.abs(residual - R_gt) * valid.float()).sum() / valid.float().sum().clamp(min=1)
                    L_height = (torch.abs(h_pred - h_gt) * valid.float()).sum() / valid.float().sum().clamp(min=1)
                    loss_cond = self.bin_weight * L_bin + self.height_weight * (self.scale_height * L_height + L_res)
                    loss_bin = L_bin

                    # Chamfer loss: bin centers 贴近 GT 的高度分布
                    L_chamfer = self.chamfer_loss(bin_edges, h_gt)
                    loss_cond = loss_cond + self.chamfer_weight * L_chamfer

                    # 均匀正则化：防止 bin 宽度过度集中
                    if 'bin_widths_normed' in preds:
                        bw = preds['bin_widths_normed']           # (B, n_bins)
                        uniform_target = torch.ones_like(bw) / bw.size(1)
                        L_uniform = F.mse_loss(bw, uniform_target)
                        loss_cond = loss_cond + self.uniform_weight * L_uniform

                    # distribution_constraint with adaptive bin_edges
                    if self.distribution_constraint and gaussian_loss is not None and 'bin_prob' in preds:
                        bin_edges_norm = bin_edges / self.max_height
                        L_prob_per_pix = gaussian_loss(
                            preds['bin_prob'], bin_edges_norm, depth_gt_2d,
                            masked=False, reduction=False
                        )
                        if L_prob_per_pix is not None and valid.sum() > 0:
                            prob_mask = valid.unsqueeze(1).float()
                            L_prob = (L_prob_per_pix * prob_mask).sum() / prob_mask.sum().clamp(min=1)
                            loss_cond = loss_cond + self.dist_constraint_weight * L_prob
                            loss_prob = L_prob
                else:
                    # 固定 bin 的原始逻辑
                    bin_gt = (h_gt / float(self.bin_len)).long().clamp(0, self.bin_num - 1)
                    bin_gt = torch.where(valid, bin_gt, bin_gt.new_full(bin_gt.shape, -100, dtype=torch.long))
                    counts = torch.bincount(bin_gt[valid].view(-1), minlength=self.bin_num).float().clamp(min=1)
                    inv_freq = 1.0 / counts
                    weight_bin = inv_freq / inv_freq.sum() * self.bin_num
                    L_bin = focal_loss_with_weights(bin_logits, bin_gt, weight_per_class=weight_bin, gamma=2.0, ignore_index=-100)
                    centers = preds['pred_d'].new_tensor(self._centers).view(1, -1, 1, 1)
                    bin_gt_onehot = F.one_hot(bin_gt.clamp(0, self.bin_num - 1), self.bin_num).permute(0, 3, 1, 2).float()
                    center_at_gt = (centers * bin_gt_onehot).sum(dim=1)
                    R_gt = (h_gt - center_at_gt).clamp(-float(self.bin_len), float(self.bin_len))
                    R_gt = torch.where(valid, R_gt, R_gt.new_zeros(1))
                    L_res = (torch.abs(residual - R_gt) * valid.float()).sum() / valid.float().sum().clamp(min=1)
                    L_height = (torch.abs(h_pred - h_gt) * valid.float()).sum() / valid.float().sum().clamp(min=1)
                    loss_cond = self.bin_weight * L_bin + self.height_weight * (self.scale_height * L_height + L_res)
                    loss_bin = L_bin

                    if self.distribution_constraint and gaussian_loss is not None and 'bin_prob' in preds:
                        bin_edges_m = [i * self.bin_len for i in range(self.bin_num + 1)]
                        bin_edges_norm = preds['pred_d'].new_tensor(
                            [e / self.max_height for e in bin_edges_m], dtype=torch.float32
                        ).view(1, -1).expand(preds['bin_prob'].size(0), -1)
                        L_prob_per_pix = gaussian_loss(
                            preds['bin_prob'], bin_edges_norm, depth_gt_2d,
                            masked=False, reduction=False
                        )
                        max_bin_edge = self.bin_num * self.bin_len
                        prob_valid = valid & (h_gt <= float(max_bin_edge))
                        if prob_valid.sum() > 0:
                            prob_mask = prob_valid.unsqueeze(1).float()
                            L_prob = (L_prob_per_pix * prob_mask).sum() / prob_mask.sum().clamp(min=1)
                            loss_cond = loss_cond + self.dist_constraint_weight * L_prob
                            loss_prob = L_prob
                        else:
                            loss_prob = preds['pred_d'].new_tensor(0.0)
            loss = loss_d + (self.lambd * loss_s) + loss_cond
        else:
            loss = loss_d + (self.lambd * loss_s)

        # -------- DPLD / TBWH 损失（结构完全一致；dict key 相同，权重来自各自 cfg）--------
        # 公式：
        #   L_bin  = focal_CE(bin_logits, bin_gt; median-frequency balanced)
        #   L_dist = mean_KL(q_gauss ‖ p_pred)   (q 用米制 bin_center 上的高斯)
        #   loss  += dpld_bin_weight * L_bin + dpld_dist_weight * L_dist
        if (
            (self.use_DPLD or self.use_TBWH)
            and 'bin_logits' in preds
            and 'bin_centers_m' in preds
            and 'bin_edges_m' in preds
        ):
            bin_logits = preds['bin_logits']                          # (B, n_bins, H, W)
            bin_prob = preds.get('bin_prob', None)
            if bin_prob is None:
                bin_prob = F.softmax(bin_logits, dim=1)

            depth_gt_2d = depth_gt.squeeze() if depth_gt.dim() == 3 else depth_gt
            if depth_gt_2d.dim() == 2:
                depth_gt_2d = depth_gt_2d.unsqueeze(0)
            h_gt_m = depth_gt_2d * self.max_height                    # (B, H, W) 米制
            valid = depth_gt_2d.ge(self.min_height_norm)               # 与旧逻辑一致

            bin_edges_m = preds['bin_edges_m']                        # (n_bins+1,) 或 (B, n_bins+1)
            bin_centers_m = preds['bin_centers_m']                    # (n_bins,)   或 (B, n_bins)
            if bin_edges_m.dim() == 1:
                inner_edges = bin_edges_m[1:-1]
            else:
                inner_edges = bin_edges_m[0, 1:-1]                    # 所有样本共享 edges
            n_bins = int(bin_logits.size(1))

            # 1) bin_gt：把米制 GT bucketize 到 bin index
            bin_gt = torch.bucketize(
                h_gt_m.contiguous(), inner_edges.to(h_gt_m.device), right=False
            ).clamp(0, n_bins - 1)
            bin_gt = torch.where(valid, bin_gt, bin_gt.new_full(bin_gt.shape, -100, dtype=torch.long))

            # 2) bin 分类权重（median-frequency balanced）
            #    旧 inverse-frequency 在 128 bins、ground 占 ~70% 时会把 bin 0 的权重压成 ≈ 0，
            #    focal CE 几乎完全忽略 ground —— 这是 DPLD 在 dfc23 上 ground 收敛慢的核心原因。
            #    改用 median-frequency 平衡：weight = median(counts) / counts，再 clamp 到 [0.1, 10]，
            #    bin 0 即便像素占主导也能拿到约 0.1 的权重，足够让模型学到 ground。
            if valid.sum() > 0:
                counts = torch.bincount(
                    bin_gt[valid].view(-1), minlength=n_bins
                ).float()
                # 仅在出现过的 bin 上算 median，避免空 bin 把 median 拉成 0
                nz = counts[counts > 0]
                if nz.numel() == 0:
                    weight_bin = counts.new_ones(n_bins)
                else:
                    median_c = nz.median().clamp(min=1.0)
                    counts_safe = counts.clamp(min=1.0)
                    weight_bin = (median_c / counts_safe).clamp(min=0.1, max=10.0)
                L_bin = focal_loss_with_weights(
                    bin_logits, bin_gt, weight_per_class=weight_bin,
                    gamma=2.0, ignore_index=-100,
                )
            else:
                L_bin = preds['pred_d'].new_tensor(0.0)

            # 3) 高斯软标签 KL（TBWH 下可由 block.yaml constraint=false 关闭）
            use_kl = self.use_DPLD or (self.use_TBWH and self.tbwh_constraint)
            if use_kl:
                c1d = bin_centers_m if bin_centers_m.dim() == 1 else bin_centers_m[0]
                c1d = c1d.to(h_gt_m.device)
                c = c1d.view(1, -1, 1, 1)                                  # (1, n_bins, 1, 1)
                h = h_gt_m.unsqueeze(1)                                   # (B, 1, H, W)
                sigma = max(float(self.dpld_sigma_m), 1e-3)
                q_logits = -((c - h) ** 2) / (2.0 * sigma * sigma)
                q = F.softmax(q_logits, dim=1)                            # (B, n_bins, H, W)
                p = bin_prob.clamp(min=1e-6)
                kl_per_pix = (q * (q.clamp(min=1e-6).log() - p.log())).sum(dim=1)  # (B, H, W)
                if valid.sum() > 0:
                    L_dist = (kl_per_pix * valid.float()).sum() / valid.float().sum().clamp(min=1)
                else:
                    L_dist = preds['pred_d'].new_tensor(0.0)
            else:
                L_dist = preds['pred_d'].new_tensor(0.0)

            loss = loss + self.dpld_bin_weight * L_bin
            if use_kl:
                loss = loss + self.dpld_dist_weight * L_dist
            # train/bin_loss 与 total 中 bin 项一致：乘权后的标量（便于调 bin_loss_weight 时看曲线）
            loss_bin = self.dpld_bin_weight * L_bin
            loss_prob = self.dpld_dist_weight * L_dist if use_kl else L_dist

        # -------- HL 尾部：地表高度 ≤ ground_cap_m 的显式软监督 --------
        # 用 GT 分割（更干净）而非 pred 分割来定义地表，避免错误传播
        # 在归一化空间计算，与其他损失量级匹配：
        #   L_cap = mean( (1 - segment_gt) · ReLU(h_pred_norm - cap_norm)^2 )
        loss_reg = preds['pred_d'].new_tensor(0.0)
        if (
            self.use_HL
            and self.use_ground_cap
            and self.option != "depth"
            and segment_gt is not None
            and self.ground_cap_weight > 0
        ):
            # 取「未钳位」的预测做监督；HL 尾部默认会同时返回 pred_d_raw
            pred_for_cap = preds.get('pred_d_raw', preds['pred_d'])
            pred_for_cap = _to_bhw(pred_for_cap)
            seg_gt_2d = _to_bhw(segment_gt).float()
            ground_mask = (1.0 - seg_gt_2d).clamp(0.0, 1.0)
            denom = ground_mask.sum().clamp(min=1.0)
            cap_norm = float(self.ground_cap_m) / max(float(self.max_height), 1e-6)
            over_norm = F.relu(pred_for_cap - cap_norm)
            L_cap = ((over_norm * over_norm) * ground_mask).sum() / denom
            loss = loss + self.ground_cap_weight * L_cap
            loss_reg = L_cap

        # -------- FHRM 辅助监督：H_base 应该独立、稳定地接近 GT --------
        # 不计入 loss_d（保持其语义为最终 pred_d 的 MSE），只加进总 loss
        loss_fhrm_aux = preds['pred_d'].new_tensor(0.0)
        if (
            self.fhrm_aux_base_weight > 0
            and 'pred_d_base' in preds
        ):
            pred_d_base = _to_bhw(preds['pred_d_base'])
            if self.de_normal:
                L_base = self.mse_loss(
                    pred_d_base * self.max_height, depth_gt * self.max_height
                )
            else:
                L_base = self.mse_loss(pred_d_base, depth_gt)
            loss_fhrm_aux = self.fhrm_aux_base_weight * L_base
            loss = loss + loss_fhrm_aux

        return loss, loss_d, loss_s, loss_bin, loss_prob, loss_reg, loss_fhrm_aux

