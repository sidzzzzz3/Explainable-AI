import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

# Imports for FastFlow
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm

#####
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTec(Dataset):
  def __init__(self,
               data_dir,
               category,
               split = "train",
               image_size = 256, # Loads at (3,256,256) image
               good_value = 0,
               anom_value = 1,
               data_type = "float", # "float" or "uint8"
               download = False):
    if download:
      raise ValueError("download not implemented")

    assert category in MVTEC_CATEGORIES
    self.data_dir = data_dir
    self.images_root_dir = os.path.join(data_dir, category, split)
    self.masks_root_dir = os.path.join(data_dir, category, "ground_truth")

    assert os.path.isdir(self.images_root_dir)
    assert os.path.isdir(self.masks_root_dir)

    self.split = split
    if split == "train":
      self.image_files = sorted(glob(os.path.join(self.images_root_dir, "good", "*.png")))
    elif split == "test":
      self.image_files = sorted(glob(os.path.join(self.images_root_dir, "*", "*.png")))
    else:
      raise ValueError(f"invalid split {split} implemented")

    self.image_size = image_size

    self.image_transforms = transforms.Compose([
          transforms.Resize(image_size, antialias=True),
        ])

    self.mask_transforms = transforms.Compose([
          transforms.Resize(image_size, antialias=True),
        ])

    assert data_type in ["float", "uint8"]
    self.data_type = data_type
    self.good_value = good_value
    self.anom_value = anom_value

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    image_file = self.image_files[index]
    image_np = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image_np.transpose(2,0,1))
    image = self.image_transforms(image)

    if self.data_type == "float":
      image = image / 255.0

    if self.split == "train":
      mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
      return image, mask, self.good_value
    else:
      if os.path.dirname(image_file).endswith("good"):
        mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
        y = self.good_value
      else:
        mask_file = image_file
        sps = image_file.split("/")
        mask_file = os.path.join(self.masks_root_dir, sps[-2], sps[-1].replace(".png", "_mask.png"))
        mask_np = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = torch.tensor(mask_np != 0).byte().unsqueeze(0)
        mask = self.mask_transforms(mask)
        y = self.anom_value

      return image, mask, y


# FastFlow model adapted from
# https://github.com/gathierry/FastFlow/blob/master/fastflow.py

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

# Preprocess fast flow images with this; assume input is [0,1]
fastflow_preprocess_transforms = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name = "resnet18",
        flow_steps = 8,
        input_size = 256,
        conv3x3_only = False,
        hidden_ratio = 1.0,
    ):
        super(FastFlow, self).__init__()
        self.backbone_name = backbone_name
        assert (
            backbone_name in SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(SUPPORTED_BACKBONES)

        if backbone_name in [BACKBONE_CAIT, BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    # Assume that x is already normalized by fastflow_preprocess_transforms
    def forward(self, x):
        bsz, nC, nH, nW = x.shape
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        log_prob_maps = []
        log_jac_dets = []
        for i, feature in enumerate(features):
            output, log_jac_det = self.nf_flows[i](feature)
            log_prob = (-0.5) * torch.mean(output**2, dim=1, keepdim=True)
            log_prob_map = F.interpolate(
                log_prob,
                size = [self.input_size, self.input_size],
                mode = "bilinear",
                align_corners = False
            )   # (N,1,256,256)

            log_prob_maps.append(log_prob_map)
            log_jac_dets.append(log_jac_det)

        heatmap = torch.mean(torch.stack(log_prob_maps, dim=-1), dim=-1)
        residual = torch.mean(torch.stack(log_jac_dets, dim=-1), dim=-1)

        # Repeat the number of channels
        heatmap = heatmap.repeat(1,nC,1,1)          # (N,C,H,W)
        residual = residual.view(-1,1).repeat(1,nC) # (N,C)

        reward = heatmap.sum(dim=(1,2,3)) + residual.sum(dim=1)
        loss = -reward

        ret = {
            "loss" : loss,
            "heatmap" : heatmap, #(N,nC,256,256)
            "residual" : residual # (N,nC)
        }

        return ret


