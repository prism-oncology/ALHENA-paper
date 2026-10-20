import timm
import torch
import cv2


# To load pretrained models from timm
MODEL_TO_ARGS = {
    "virchow2": {
        "architecture": "vit_huge_patch14_224",
        "timm_kwargs": {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "reg_tokens": 4,
            "mlp_ratio": 5.3375,
            "global_pool": "",
            "dynamic_img_size": True,
        },
        "pretrained_cfg": {
            "tag": "virchow_v2",
            "custom_load": False,
            "input_size": [3, 224, 224],
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "CC-BY-NC-ND-4.0",
        },
    },
    "uni2h": {
        "architecture": "vit_giant_patch14_224",
        "timm_kwargs": {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        "pretrained_cfg": {
            "tag": "uni2-h",
            "custom_load": True,
            "crop_pct": 1,
            "input_size": [3, 224, 224],
            "fixed_input_size": False,
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "CC BY-NC-ND 4.0",
        },
    },
    "hoptimus0": {
        "architecture": "vit_giant_patch14_reg4_dinov2",
        "timm_kwargs": {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        },
        "pretrained_cfg": {
            "custom_load": True,
            "input_size": [3, 224, 224],
            "fixed_input_size": True,
            "mean": [0.707223, 0.578729, 0.703617],
            "std": [0.211883, 0.230117, 0.177517],
            "num_classes": 0,
            "license": "Apache 2.0",
        },
    },
    "hoptimus1": {
        "architecture": "vit_giant_patch14_reg4_dinov2",
        "timm_kwargs": {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        },
        "pretrained_cfg": {
            "custom_load": True,
            "input_size": [3, 224, 224],
            "fixed_input_size": True,
            "mean": [0.707223, 0.578729, 0.703617],
            "std": [0.211883, 0.230117, 0.177517],
            "num_classes": 0,
            "license": "CC BY-NC-ND 4.0",
        }
    },
}


# Specific transform for each model
BACKBONE_TRANSFORM = {
    "virchow2": {
        "resize_size": 224,            # Resize to 224
        "crop_size": 224,             # Then center-crop 224
        "interpolation": cv2.INTER_CUBIC,  # approximate 'bicubic'
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "uni2h": {
        "resize_size": 224,
        "crop_size": 224,
        "interpolation": cv2.INTER_LINEAR,  # 'bilinear'
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "hoptimus0": {
        "resize_size": 224,
        "crop_size": 224,
        "interpolation": cv2.INTER_CUBIC,  # 'bicubic'
        "mean": [0.707223, 0.578729, 0.703617],
        "std": [0.211883, 0.230117, 0.177517],
    },
    "hoptimus1": {
        "resize_size": 224,
        "crop_size": 224,
        "interpolation": cv2.INTER_CUBIC,  # 'bicubic'
        "mean": [0.707223, 0.578729, 0.703617],
        "std": [0.211883, 0.230117, 0.177517],
    },
}


# Native resolution (magnification / microns per pixel) of each backbone, used by Task B to group encoders.
# Keys must match BACKBONE_TRANSFORM.
BACKBONE_NATIVE_RESOLUTION = {
    "virchow2":  {"nominal_magnification": 20, "nominal_mpp_um_per_px": 0.5},
    "uni2h":     {"nominal_magnification": 20, "nominal_mpp_um_per_px": 0.5},
    "hoptimus0": {"nominal_magnification": 20, "nominal_mpp_um_per_px": 0.5},
    "hoptimus1": {"nominal_magnification": 20, "nominal_mpp_um_per_px": 0.5},
}