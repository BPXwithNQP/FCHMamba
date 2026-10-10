import os

import torch
import torch.nn as nn
import torchvision.models as models


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmcv.runner import load_checkpoint


class Resnet(nn.Module):
    def __init__(self, pre_trained=False, depth=101, ckpt_path=None) -> None:
        super(Resnet, self).__init__()

        depth = int(depth)
        if depth == 50:
            self.resnet = models.resnet50(pretrained=False)
        elif depth == 101:
            self.resnet = models.resnet101(pretrained=False)
        else:
            raise ValueError("Resnet depth 仅支持 50 或 101，收到: {}".format(depth))

        if pre_trained:
            if not ckpt_path or not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    "ResNet 预训练权重未找到: {}；请下载后放到 主干网络内容/预训练权重/".format(
                        ckpt_path or "(未配置 weights_path)"
                    )
                )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.resnet.load_state_dict(state_dict, strict=True)
            print(
                "[Encoder/res] 预训练已加载: {}".format(os.path.abspath(ckpt_path)),
                flush=True,
            )
    
    def forward(self, x):
        layer_outs = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        layer_outs.append(x)
        x = self.resnet.layer2(x)
        layer_outs.append(x)
        x = self.resnet.layer3(x)
        layer_outs.append(x)
        x = self.resnet.layer4(x)
        layer_outs.append(x)

        return layer_outs