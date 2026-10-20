import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import random


# M matrix of each slide: estimated in QuPath (Analyze -> Estimate stain vectors, H&E only), see
# https://qupath.readthedocs.io/en/stable/docs/tutorials/separating_stains.html


# ---------------------------------------------------------------------------
# Stain matrices (computed once at import time)
# ---------------------------------------------------------------------------

# Matrix M: from HED -> RGB (approx). It is inverted here for RGB -> HED.
# See: Quantification of histochemical staining by color deconvolution, Arnout Ruifrok, Dennis A Johnston, January 2001
# Default values from QuPath
M_DEFAULT = torch.tensor([
    [0.651, 0.701, 0.29],   # Hematoxylin
    [0.216, 0.801, 0.558],  # Eosin
    [0.316, -0.598, 0.737]  # Residual
], dtype=torch.float32)
RGB2HED_DEFAULT = torch.inverse(M_DEFAULT)

# M matrices dictionary: maps slide_id -> M matrix (HED -> RGB).
M_MATRICES = {
    # M matrix for breast_s0 slide
    "breast_s0": torch.tensor([
        [0.588, 0.757, 0.284],
        [0.223, 0.946, 0.235],
        [-0.225, -0.185, 0.957]
    ], dtype=torch.float32),

    # M matrix for breast_s1 slide
    "breast_s1": torch.tensor([
        [0.58, 0.758, 0.299],
        [0.219, 0.912, 0.347],
        [-0.025, -0.35, 0.936]
    ], dtype=torch.float32),

    # M matrix for breast_s3 slide
    "breast_s3": torch.tensor([
        [0.567, 0.782, 0.26],
        [0.193, 0.952, 0.237],
        [-0.154, -0.21, 0.966]
    ], dtype=torch.float32),

    # M matrix for breast_s6 slide
    "breast_s6": torch.tensor([
        [0.549, 0.803, 0.231],
        [0.119, 0.95, 0.288],
        [0.025, -0.293, 0.956]
    ], dtype=torch.float32),

    # M matrix for lung_s1 slide
    "lung_s1": torch.tensor([
        [0.396, 0.875, 0.278],
        [0.126, 0.948, 0.293],
        [-0.028, -0.292, 0.956]
    ], dtype=torch.float32),

    # M matrix for lung_s3 slide
    "lung_s3": torch.tensor([
        [0.329, 0.916, 0.229],
        [0.101, 0.951, 0.291],
        [0.205, -0.306, 0.93]
    ], dtype=torch.float32),

    # M matrix for skin_s1 slide
    "skin_s1": torch.tensor([
        [0.455, 0.846, 0.278],
        [0.111, 0.928, 0.356],
        [0.122, -0.368, 0.922]
    ], dtype=torch.float32),

    # M matrix for skin_s2 slide
    "skin_s2": torch.tensor([
        [0.577, 0.757, 0.308],
        [0.164, 0.941, 0.296],
        [-0.15, -0.273, 0.95]
    ], dtype=torch.float32),

    # M matrix for skin_s3 slide
    "skin_s3": torch.tensor([
        [0.449, 0.852, 0.27],
        [0.105, 0.948, 0.3],
        [-0,  -0.302,  0.953]
    ], dtype=torch.float32),

    # M matrix for skin_s4 slide
    "skin_s4": torch.tensor([
        [0.449, 0.86, 0.241],
        [0.114, 0.934, 0.339],
        [0.19,  -0.355,  0.915]
    ], dtype=torch.float32),

    # M matrix for pancreatic_s0 slide
    "pancreatic_s0": torch.tensor([
        [0.393, 0.879, 0.269],
        [0.065, 0.947, 0.316],
        [0.069, -0.32, 0.945]
    ], dtype=torch.float32),

    # M matrix for pancreatic_s1 slide
    "pancreatic_s1": torch.tensor([
        [0.412, 0.867, 0.281],
        [0.113, 0.945, 0.306],
        [-0.002, -0.308, 0.951]
    ], dtype=torch.float32),

    # M matrix for pancreatic_s2 slide
    "pancreatic_s2": torch.tensor([
        [0.448, 0.854, 0.265],
        [0.139, 0.94, 0.312],
        [0.054, -0.321, 0.945]
    ], dtype=torch.float32),

    # M matrix for heart_s0 slide
    "heart_s0": torch.tensor([
        [0.579, 0.764, 0.284],
        [0.143, 0.948, 0.284],
        [-0.114, -0.269, 0.956]
    ], dtype=torch.float32),

    # M matrix for colon_s1 slide
    "colon_s1": torch.tensor([
        [0.261, 0.923, 0.282],
        [0.094, 0.951, 0.294],
        [0.015, -0.296, 0.955]
    ], dtype=torch.float32),

    # M matrix for colon_s2 slide
    "colon_s2": torch.tensor([
        [0.343, 0.894, 0.287],
        [0.091, 0.925, 0.368],
        [0.24, -0.379, 0.894]
    ], dtype=torch.float32),

    # M matrix for kidney_s0 slide
    "kidney_s0": torch.tensor([
        [0.539, 0.804, 0.249],
        [0.18, 0.941, 0.287],
        [-0.01, -0.29, 0.957]
    ], dtype=torch.float32),

    # M matrix for kidney_s1 slide
    "kidney_s1": torch.tensor([
        [0.658, 0.725, 0.202],
        [0.228, 0.932, 0.281],
        [0.034, -0.296, 0.955]
    ], dtype=torch.float32),

    # M matrix for kidney_s2 slide
    "kidney_s2": torch.tensor([
        [0.587, 0.776, 0.231],
        [0.21, 0.95, 0.23],
        [-0.101, -0.213, 0.972]
    ], dtype=torch.float32),

    # M matrix for liver_s0 slide
    "liver_s0": torch.tensor([
        [0.411, 0.871, 0.269],
        [0.11, 0.937, 0.333],
        [0.122, -0.345, 0.931]
    ], dtype=torch.float32),

    # M matrix for liver_s1 slide
    "liver_s1": torch.tensor([
        [0.637, 0.745, 0.2],
        [0.186, 0.928, 0.321],
        [0.11, -0.345, 0.932]
    ], dtype=torch.float32),

    # M matrix for tonsil_s0 slide
    "tonsil_s0": torch.tensor([
        [0.557, 0.774, 0.303],
        [0.111, 0.94, 0.321],
        [-0.079, -0.314, 0.946]
    ], dtype=torch.float32),

    # M matrix for tonsil_s1 slide
    "tonsil_s1": torch.tensor([
        [0.484, 0.828, 0.284],
        [0.102, 0.945, 0.31],
        [-0.03, -0.309, 0.951]
    ], dtype=torch.float32),

    # M matrix for lymph_node_s0 slide
    "lymph_node_s0": torch.tensor([
        [0.475, 0.837, 0.273],
        [0.155, 0.934, 0.321],
        [0.041, -0.33, 0.943]
    ], dtype=torch.float32),

    # M matrix for ovary_s0 slide
    "ovary_s0": torch.tensor([
        [0.451, 0.854, 0.258],
        [0.128, 0.948, 0.291],
        [0.012, -0.295, 0.955]
    ], dtype=torch.float32),

    # M matrix for ovary_s1 slide
    "ovary_s1": torch.tensor([
        [0.557, 0.793, 0.248],
        [0.146, 0.943, 0.298],
        [0.005, -0.302, 0.953]
    ], dtype=torch.float32),

    # M matrix for brain_s0 slide
    "brain_s0": torch.tensor([
        [0.503, 0.816, 0.286],
        [0.139, 0.941, 0.309],
        [-0.043, -0.306, 0.951]
    ], dtype=torch.float32),

    # M matrix for bone_marrow_s0 slide
    "bone_marrow_s0": torch.tensor([
        [0.554, 0.786, 0.274],
        [0.075, 0.958, 0.276],
        [-0.093, -0.269, 0.959]
    ], dtype=torch.float32),

    # M matrix for bone_marrow_s1 slide
    "bone_marrow_s1": torch.tensor([
        [0.462, 0.833, 0.304],
        [0.1, 0.962, 0.254],
        [-0.212, -0.229, 0.95]
    ], dtype=torch.float32),

    # M matrix for bone_s0 slide
    "bone_s0": torch.tensor([
        [0.5, 0.815, 0.293],
        [0.105, 0.959, 0.264],
        [-0.159, -0.246, 0.956]
    ], dtype=torch.float32),

    # M matrix for prostate_s0 slide
    "prostate_s0": torch.tensor([
        [0.484, 0.824, 0.293],
        [0.132, 0.947, 0.291],
        [-0.101, -0.28, 0.955]
    ], dtype=torch.float32),

    # M matrix for cervix_s0 slide
    "cervix_s0": torch.tensor([
        [0.559, 0.785, 0.267],
        [0.15, 0.955, 0.257],
        [-0.124, -0.24, 0.963]
    ], dtype=torch.float32)
}

# Precompute the RGB -> HED inverse for each slide once at import time.
RGB2HEDS = {k: torch.inverse(v) for k, v in M_MATRICES.items()}


def hed_augment_image(
    img: np.ndarray,
    slide_id: str,
    p: float,
    sigma: float
) -> np.ndarray:
    """
    Applies HED-based color augmentation if random.random() < p

    Args:
        img (np.ndarray): Input image of shape (H, W, C) in [0..255]
        slide_id (str): Identifier for the slide. Used to select the M matrix
        p (float): Probability of applying this transform
        sigma (float): Standard deviation for random scaling/offset in HED space

    Inspired from the paper: Faryna, K., Van der Laak, J. and Litjens, G., 2021, February. Tailoring automated data augmentation to H&E-stained histopathology. In Medical imaging with deep learning.
    """
    if random.random() >= p:
        # Augmentation not applied
        return img

    # Choose the M and RGB2HED matrices for the given slide_id
    if slide_id in M_MATRICES:
        M = M_MATRICES[slide_id]
        RGB2HED = RGB2HEDS[slide_id]
    else:
        logger.warning(f"[WARNING] Using default M, slide_id='{slide_id}' not found")
        M = M_DEFAULT
        RGB2HED = RGB2HED_DEFAULT

    # Convert from NumPy [H,W,C] to a float Torch tensor [C,H,W] in [0..1]
    if img.dtype != np.float32 and img.dtype != np.float64:
        img_t = torch.from_numpy(img.astype(np.float32))
    else:
        img_t = torch.from_numpy(img)

    # shape = (H,W,C). Permute to (C,H,W)
    img_t = img_t.permute(2, 0, 1)  # [C,H,W]

    # Scale to [0..1] if needed
    if img_t.max() > 1.0:
        img_t = img_t / 255.0

    eps = 1e-6
    C, H, W = img_t.shape
    flat = img_t.reshape(C, -1).T  # [N,3], N = H*W
    # Get optical density (OD) space
    S = torch.matmul(-torch.log(flat + eps), RGB2HED)

    # Random shift in HED
    alpha = torch.normal(mean=1.0, std=sigma, size=(1,3))
    beta  = torch.normal(mean=0.0, std=sigma, size=(1,3))
    S_hat = alpha * S + beta

    # Convert back to "RGB" space
    out = torch.exp(-torch.matmul(S_hat, M))

    # Reshape back to (C,H,W)
    out = out.T.reshape(C, H, W)

    # Clip to [0..1]
    out = torch.clamp(out, 0.0, 1.0)

    # Convert back to NumPy with shape (H,W,C) in [0..255]
    out = (out * 255.0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    return out
