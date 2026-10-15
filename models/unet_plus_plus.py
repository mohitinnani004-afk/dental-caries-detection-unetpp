"""
UNet++ (Nested U-Net) with Dense Skip Pathways and Deep Supervision
Specially adapted for Dental Caries Detection and Medical Radiograph Segmentation.

Reference:
    Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018).
    "UNet++: A Nested U-Net Architecture for Medical Image Segmentation."
    Deep Learning in Medical Image Analysis (DLMIA), LNCS 11045, pp. 3-11.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> BatchNorm/InstanceNorm -> ReLU] x 2
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout_rate: float = 0.0,
        norm_type: str = "batchnorm"
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        norm_layer = nn.BatchNorm2d if norm_type == "batchnorm" else nn.InstanceNorm2d

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate for suppressing background noise (e.g. bone, metal artifacts)
    and highlighting radiolucent caries regions.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetPlusPlus(nn.Module):
    """
    Nested U-Net (UNet++) Architecture.
    
    Nodes are indexed as X^{i,j}:
      - i: downsampling level along the encoder (0 to 4)
      - j: convolution block along the skip pathway (0 to 4)
    
    Features:
      - Dense skip pathways connecting intermediate nodes
      - Deep supervision heads at X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}
      - Runtime model pruning support (L1, L2, L3, L4)
      - Optional Attention Gating on skip connections
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        deep_supervision: bool = True,
        nb_filter: Optional[List[int]] = None,
        use_attention: bool = True,
        dropout_rate: float = 0.15,
        norm_type: str = "batchnorm"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_attention = use_attention
        self.pruning_level: Optional[int] = None  # None = full model (L4)

        if nb_filter is None:
            nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ----------------- Level 0 (Encoder) -----------------
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0], dropout_rate=dropout_rate, norm_type=norm_type)
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], dropout_rate=dropout_rate, norm_type=norm_type)
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], dropout_rate=dropout_rate, norm_type=norm_type)
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], dropout_rate=dropout_rate, norm_type=norm_type)
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], dropout_rate=dropout_rate, norm_type=norm_type)

        # ----------------- Attention Gates (Optional) -----------------
        if self.use_attention:
            self.att0_1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
            self.att1_1 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)
            self.att2_1 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2] // 2)
            self.att3_1 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3] // 2)

            self.att0_2 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
            self.att1_2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)
            self.att2_2 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2] // 2)

            self.att0_3 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)
            self.att1_3 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1] // 2)

            self.att0_4 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0] // 2)

        # ----------------- Column j = 1 -----------------
        # X^{0,1}: receives X^{0,0} and Up(X^{1,0})
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0], norm_type=norm_type)
        # X^{1,1}: receives X^{1,0} and Up(X^{2,0})
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1], norm_type=norm_type)
        # X^{2,1}: receives X^{2,0} and Up(X^{3,0})
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2], norm_type=norm_type)
        # X^{3,1}: receives X^{3,0} and Up(X^{4,0})
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3], norm_type=norm_type)

        # ----------------- Column j = 2 -----------------
        # X^{0,2}: receives X^{0,0}, X^{0,1}, and Up(X^{1,1})
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], norm_type=norm_type)
        # X^{1,2}: receives X^{1,0}, X^{1,1}, and Up(X^{2,1})
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], norm_type=norm_type)
        # X^{2,2}: receives X^{2,0}, X^{2,1}, and Up(X^{3,1})
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], norm_type=norm_type)

        # ----------------- Column j = 3 -----------------
        # X^{0,3}: receives X^{0,0}, X^{0,1}, X^{0,2}, and Up(X^{1,2})
        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], norm_type=norm_type)
        # X^{1,3}: receives X^{1,0}, X^{1,1}, X^{1,2}, and Up(X^{2,2})
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], norm_type=norm_type)

        # ----------------- Column j = 4 -----------------
        # X^{0,4}: receives X^{0,0}, X^{0,1}, X^{0,2}, X^{0,3}, and Up(X^{1,3})
        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], norm_type=norm_type)

        # ----------------- Output Heads (Deep Supervision) -----------------
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def set_pruning_level(self, level: Optional[int]):
        """
        Set pruning depth:
          - level=1: Evaluates up to X^{0,1} (L1 sub-network, fastest)
          - level=2: Evaluates up to X^{0,2} (L2 sub-network)
          - level=3: Evaluates up to X^{0,3} (L3 sub-network)
          - level=4 or None: Full UNet++ (L4)
        """
        assert level in (None, 1, 2, 3, 4), "Pruning level must be None, 1, 2, 3, or 4"
        self.pruning_level = level

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder Pathway
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))

        # Level 1 skip
        up1_0 = self.up(x1_0)
        skip0_0 = self.att0_1(g=up1_0, x=x0_0) if self.use_attention else x0_0
        x0_1 = self.conv0_1(torch.cat([skip0_0, up1_0], dim=1))

        if self.pruning_level == 1:
            out1 = self.final1(x0_1) if self.deep_supervision else self.final(x0_1)
            return out1

        x2_0 = self.conv2_0(self.pool(x1_0))
        up2_0 = self.up(x2_0)
        skip1_0 = self.att1_1(g=up2_0, x=x1_0) if self.use_attention else x1_0
        x1_1 = self.conv1_1(torch.cat([skip1_0, up2_0], dim=1))

        up1_1 = self.up(x1_1)
        skip0_1 = self.att0_2(g=up1_1, x=x0_1) if self.use_attention else x0_1
        x0_2 = self.conv0_2(torch.cat([x0_0, skip0_1, up1_1], dim=1))

        if self.pruning_level == 2:
            out2 = self.final2(x0_2) if self.deep_supervision else self.final(x0_2)
            return out2

        x3_0 = self.conv3_0(self.pool(x2_0))
        up3_0 = self.up(x3_0)
        skip2_0 = self.att2_1(g=up3_0, x=x2_0) if self.use_attention else x2_0
        x2_1 = self.conv2_1(torch.cat([skip2_0, up3_0], dim=1))

        up2_1 = self.up(x2_1)
        skip1_1 = self.att1_2(g=up2_1, x=x1_1) if self.use_attention else x1_1
        x1_2 = self.conv1_2(torch.cat([x1_0, skip1_1, up2_1], dim=1))

        up1_2 = self.up(x1_2)
        skip0_2 = self.att0_3(g=up1_2, x=x0_2) if self.use_attention else x0_2
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, skip0_2, up1_2], dim=1))

        if self.pruning_level == 3:
            out3 = self.final3(x0_3) if self.deep_supervision else self.final(x0_3)
            return out3

        x4_0 = self.conv4_0(self.pool(x3_0))
        up4_0 = self.up(x4_0)
        skip3_0 = self.att3_1(g=up4_0, x=x3_0) if self.use_attention else x3_0
        x3_1 = self.conv3_1(torch.cat([skip3_0, up4_0], dim=1))

        up3_1 = self.up(x3_1)
        skip2_1 = self.att2_2(g=up3_1, x=x2_1) if self.use_attention else x2_1
        x2_2 = self.conv2_2(torch.cat([x2_0, skip2_1, up3_1], dim=1))

        up2_2 = self.up(x2_2)
        skip1_2 = self.att1_3(g=up2_2, x=x1_2) if self.use_attention else x1_2
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, skip1_2, up2_2], dim=1))

        up1_3 = self.up(x1_3)
        skip0_3 = self.att0_4(g=up1_3, x=x0_3) if self.use_attention else x0_3
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, skip0_3, up1_3], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            if self.training:
                # Return all 4 outputs during training for deep supervision loss
                return [out1, out2, out3, out4]
            else:
                # During inference, return final full-depth output (or average)
                return out4
        else:
            out = self.final(x0_4)
            return out


def build_unet_plus_plus(
    in_channels: int = 1,
    num_classes: int = 1,
    deep_supervision: bool = True,
    use_attention: bool = True,
    dropout_rate: float = 0.15,
    nb_filter: Optional[List[int]] = None
) -> UNetPlusPlus:
    """Helper factory function to construct UNet++ instance."""
    return UNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        nb_filter=nb_filter,
        use_attention=use_attention,
        dropout_rate=dropout_rate
    )


class ColabConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ColabUNetPlusPlus(nn.Module):
    """
    Exact U-Net++ architecture trained in the Colab session with base=16 channels.
    Layers: x00, x10, x20, x30, x40, x01, x11, x21, x31, x02, x12, x22, x03, x13, x04, final.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder
        self.x00 = ColabConvBlock(in_channels, base)
        self.x10 = ColabConvBlock(base, base * 2)
        self.x20 = ColabConvBlock(base * 2, base * 4)
        self.x30 = ColabConvBlock(base * 4, base * 8)
        self.x40 = ColabConvBlock(base * 8, base * 16)

        # First nested decoder level
        self.x01 = ColabConvBlock(base * 3, base)
        self.x11 = ColabConvBlock(base * 6, base * 2)
        self.x21 = ColabConvBlock(base * 12, base * 4)
        self.x31 = ColabConvBlock(base * 24, base * 8)

        # Second nested decoder level
        self.x02 = ColabConvBlock(base * 4, base)
        self.x12 = ColabConvBlock(base * 8, base * 2)
        self.x22 = ColabConvBlock(base * 16, base * 4)

        # Third nested decoder level
        self.x03 = ColabConvBlock(base * 5, base)
        self.x13 = ColabConvBlock(base * 10, base * 2)

        # Fourth nested decoder level
        self.x04 = ColabConvBlock(base * 6, base)

        # Output layer
        self.final = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], dim=1))
        x31 = self.x31(torch.cat([x30, self.up(x40)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13)], dim=1))
        return self.final(x04)


def load_unet_plus_plus_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, dict]:
    """
    Intelligently inspects checkpoint state_dict and instantiates the exact
    matching UNet++ architecture (Colab x00-style vs Full Deep Supervision style).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state

    keys = list(state_dict.keys())
    is_colab_format = any(k.startswith("x00") or k.startswith("x10") for k in keys)

    if is_colab_format:
        base = 16
        for k in keys:
            if "x00.block.0.weight" in k:
                base = state_dict[k].shape[0]
                break
        model = ColabUNetPlusPlus(in_channels=1, out_channels=1, base=base)
        model.load_state_dict(state_dict)
        print(f"[OK] Instantiated and loaded Colab UNet++ architecture (base={base}) from {ckpt_path.name}")
    else:
        model = build_unet_plus_plus(
            in_channels=1,
            num_classes=1,
            deep_supervision=True,
            nb_filter=[32, 64, 128, 256, 512]
        )
        try:
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded Full UNet++ architecture from {ckpt_path.name}")
        except Exception:
            model = build_unet_plus_plus(
                in_channels=1,
                num_classes=1,
                deep_supervision=False,
                nb_filter=[32, 64, 128, 256, 512]
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Loaded UNet++ architecture (non-strict) from {ckpt_path.name}")

    model = model.to(device)
    model.eval()
    return model, state if isinstance(state, dict) else {}

