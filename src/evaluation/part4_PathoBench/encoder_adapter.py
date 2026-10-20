"""Trident patch encoder for an exported H&E encoder bundle (he features), and clip derivation from he."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.evaluation.extract_encoders import load_exported_he_encoder

logger = logging.getLogger(__name__)

# L2 norm clamp (as in Task B)
_L2_EPS = 1e-12


# --------------------------------------------------------------------------- preprocessing


class HEPreprocessTransform:
    """
    Per-patch H&E preprocessing from the exported encoder metadata, as in Task B _preprocess_rgb_to_tensor
    (cv2 square resize, center crop, /255, (x - mean) / std). Returns a CHW float32 tensor.
    """

    def __init__(
        self,
        resize_size: int,
        crop_size: int,
        mean: Any,
        std: Any,
        interpolation_code: int,
    ) -> None:
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.interpolation_code = int(interpolation_code)
        # (1, 1, 3): normalization in HWC
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> "HEPreprocessTransform":
        """Read resize/crop/mean/std/interpolation off an exported encoder module."""
        mean = encoder.mean.detach().cpu().reshape(-1).numpy()  # buffer (1,3,1,1) -> (3,)
        std = encoder.std.detach().cpu().reshape(-1).numpy()
        return cls(
            resize_size=int(getattr(encoder, "resize_size", 0)),
            crop_size=int(getattr(encoder, "crop_size", 0)),
            mean=mean,
            std=std,
            interpolation_code=int(getattr(encoder, "interpolation_code", cv2.INTER_LINEAR)),
        )

    @staticmethod
    def _to_rgb_uint8(image: Any) -> np.ndarray:
        """Convert a PIL / ndarray / tensor patch to a contiguous uint8 RGB HWC array (RGBA composited on white)."""
        if isinstance(image, Image.Image):
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                white = Image.new("RGBA", image.size, (255, 255, 255, 255))
                white.alpha_composite(image)
                image = white
            arr = np.asarray(image.convert("RGB"))
        elif torch.is_tensor(image):
            x = image.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0)
            arr = x.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 4:
                # Composite RGBA on white (uint8), matching Task B.
                rgba = arr.astype(np.float32)
                alpha = rgba[:, :, 3:4] / 255.0
                rgb = rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
                arr = np.clip(rgb, 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected an RGB patch with shape [H, W, 3], got {arr.shape}.")
        return np.ascontiguousarray(arr)

    def __call__(self, image: Any) -> torch.Tensor:
        rgb = self._to_rgb_uint8(image)

        # Square resize (matches _resize_rgb: no-op when already the target size).
        if self.resize_size > 0 and not (rgb.shape[0] == self.resize_size and rgb.shape[1] == self.resize_size):
            rgb = cv2.resize(rgb, (self.resize_size, self.resize_size), interpolation=self.interpolation_code)

        # Center crop (matches _center_crop).
        if self.crop_size > 0:
            h, w = rgb.shape[:2]
            if h < self.crop_size or w < self.crop_size:
                raise RuntimeError(f"Cannot center-crop {self.crop_size} from a patch of shape {rgb.shape}.")
            y0 = (h - self.crop_size) // 2
            x0 = (w - self.crop_size) // 2
            rgb = rgb[y0:y0 + self.crop_size, x0:x0 + self.crop_size, :]

        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        if not np.isfinite(arr).all():
            raise RuntimeError("Non-finite value after H&E preprocessing.")
        arr = np.transpose(arr, (2, 0, 1)).copy()  # HWC -> CHW
        return torch.from_numpy(arr)


# --------------------------------------------------------------------------- encoder loading


def load_bundle_encoder(bundle_path: Union[str, Path], device: Optional[torch.device] = None) -> nn.Module:
    """Load an exported ``HEencoder_t{tn}e{cn}.pt`` bundle as a frozen, eval-mode module."""
    device = device or torch.device("cpu")
    encoder = load_exported_he_encoder(str(bundle_path), device=device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def encoder_has_clip(encoder: nn.Module) -> bool:
    """True iff the bundle exports an ``image_proj`` (finetuned / SEAL); False for ``t-1e-1``."""
    return getattr(encoder, "image_proj", None) is not None


# --------------------------------------------------------------------------- the `he` producer


class HEPatchFeatureModel(nn.Module):
    """nn.Module given to Trident's CustomInferenceEncoder: forward(x) returns the he space (encode_image(project=False))."""

    def __init__(self, encoder: nn.Module, enc_name: str = "histost_he") -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # Attributes Trident's CustomInferenceEncoder / feature writer may read.
        self.enc_name = str(enc_name)
        self.embed_dim = int(getattr(encoder, "image_feature_dim", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # no_grad (not inference_mode) -> plain tensors for Trident's h5 writer
            he = self.encoder.encode_image(x, project=False)
        if he.ndim != 2:
            raise RuntimeError(f"Encoder `he` output must be [B, D], got {tuple(he.shape)}.")
        if torch.any(~torch.isfinite(he)):
            raise RuntimeError("NaN/Inf in encoder `he` output batch.")
        return he.float()


def derive_clip_from_he(
    he: Union[np.ndarray, torch.Tensor],
    encoder: nn.Module,
) -> Union[np.ndarray, torch.Tensor]:
    """clip = L2(image_proj(he)) from a stored he matrix (same array type); raises if the encoder has no image_proj."""
    proj = getattr(encoder, "image_proj", None)
    if proj is None:
        raise RuntimeError(
            "This encoder has no `image_proj` (he-only bundle, e.g. t-1e-1); cannot derive `clip`."
        )
    was_numpy = isinstance(he, np.ndarray)
    proj_device = next(proj.parameters()).device
    t = torch.as_tensor(he, dtype=torch.float32, device=proj_device)
    if t.ndim != 2:
        raise RuntimeError(f"`he` must be [N, D], got {tuple(t.shape)}.")
    with torch.no_grad():
        clip = proj(t)
        norm = torch.linalg.vector_norm(clip, dim=1, keepdim=True)
        if torch.any(norm <= 0) or torch.any(~torch.isfinite(norm)):
            raise RuntimeError("Invalid CLIP embedding norm before L2 normalization.")
        clip = clip / torch.clamp(norm, min=_L2_EPS)
    clip = clip.cpu()
    return clip.numpy() if was_numpy else clip


# --------------------------------------------------------------------------- Trident entry point


def build_trident_custom_encoder(
    bundle_path: Union[str, Path],
    enc_name: str,
    device: Optional[torch.device] = None,
    precision: "torch.dtype" = torch.float32,
) -> Any:
    """
    Wrap an exported bundle as a Trident CustomInferenceEncoder that outputs the he space (fp32 by default).
    Run in the trident env.
    """
    from trident.patch_encoder_models import CustomInferenceEncoder  # lazy: trident env only

    device = device or torch.device("cpu")
    encoder = load_bundle_encoder(bundle_path, device=device)
    model = HEPatchFeatureModel(encoder, enc_name=enc_name).to(device).eval()
    transforms = HEPreprocessTransform.from_encoder(encoder)
    logger.info(
        "Built Trident CustomInferenceEncoder %r (he dim=%d, has_clip=%s, resize=%d, crop=%d).",
        enc_name, model.embed_dim, encoder_has_clip(encoder), encoder.resize_size, encoder.crop_size,
    )
    return CustomInferenceEncoder(
        enc_name=enc_name,
        model=model,
        transforms=transforms,
        precision=precision,
    )


# --------------------------------------------------------------------------- self-check


def _selfcheck() -> None:
    """Synthetic check of the preprocessing (vs Task B), he output, clip derivation and he-only encoders."""
    from src.evaluation.extract_encoders import HEClipEncoderNoTokens

    torch.manual_seed(0)
    size, feat, emb = 16, 24, 12
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    class _DummyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3 * size * size, feat)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x.flatten(1))

    def _make(has_proj: bool) -> nn.Module:
        return HEClipEncoderNoTokens(
            image_encoder=_DummyBackbone(),
            image_feature_dim=feat,
            embed_dim=emb,
            has_image_proj=has_proj,
            image_proj_bias=has_proj,
            backbone="uni2h",
            resize_size=size,
            crop_size=size,
            mean=mean,
            std=std,
            interpolation_code=cv2.INTER_LINEAR,
        ).eval()

    enc = _make(has_proj=True)
    tfm = HEPreprocessTransform.from_encoder(enc)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)

    # (1) preprocessing vs the reference formula
    ref = img.astype(np.float32) / 255.0
    ref = (ref - mean.reshape(1, 1, 3).numpy()) / std.reshape(1, 1, 3).numpy()
    ref = torch.from_numpy(np.transpose(ref, (2, 0, 1)).copy())
    got = tfm(img)
    assert torch.allclose(got, ref, atol=1e-6), "HEPreprocessTransform diverges from Task B reference."

    # (1b) preprocessing vs Task B _preprocess_rgb_to_tensor (its dir must be on sys.path)
    import sys
    _taskb_dir = Path(__file__).resolve().parents[1] / "part2_TCGA" / "taskB_tcga_eval"
    if str(_taskb_dir) not in sys.path:
        sys.path.insert(0, str(_taskb_dir))
    try:
        from src.evaluation.part2_TCGA.taskB_tcga_eval._0_extract_encoder_features import (
            _preprocess_rgb_to_tensor as _taskb_preprocess,
        )
    except Exception as exc:  # pragma: no cover - env-dependent (needs pandas / PIL / tqdm)
        logger.warning("Skipping Task B preprocessing parity check (import failed: %s).", exc)
    else:
        mean_np = mean.reshape(-1).numpy()
        std_np = std.reshape(-1).numpy()
        for rs, cs in ((size, size), (size + 8, size)):  # identity, and a genuine resize→center-crop
            tfm2 = HEPreprocessTransform(rs, cs, mean_np, std_np, cv2.INTER_LINEAR)
            ref2 = _taskb_preprocess(img, {
                "resize_size": rs, "crop_size": cs, "interpolation_code": int(cv2.INTER_LINEAR),
                "mean": mean_np, "std": std_np,
            })
            assert torch.allclose(tfm2(img), ref2, atol=1e-6), \
                f"HEPreprocessTransform != Task B _preprocess_rgb_to_tensor (resize={rs}, crop={cs})."

    x = got.unsqueeze(0)
    # (2) he parity.
    he_model = HEPatchFeatureModel(enc)
    he = he_model(x)
    he_ref = enc.encode_image(x, project=False)
    assert torch.allclose(he, he_ref, atol=1e-6), "HEPatchFeatureModel != encode_image(project=False)."

    # (3) clip derivation parity (offline vs project=True + L2).
    clip = derive_clip_from_he(he, enc)
    clip_ref = enc.encode_image(x, project=True)
    clip_ref = clip_ref / torch.clamp(torch.linalg.vector_norm(clip_ref, dim=1, keepdim=True), min=_L2_EPS)
    assert torch.allclose(clip, clip_ref, atol=1e-6), "derive_clip_from_he != L2(image_proj(he))."
    # numpy round-trips too.
    clip_np = derive_clip_from_he(he.detach().numpy(), enc)
    assert np.allclose(clip_np, clip.detach().numpy(), atol=1e-6), "numpy clip derivation mismatch."

    # (4) he-only encoder.
    enc_he_only = _make(has_proj=False)
    assert not encoder_has_clip(enc_he_only), "he-only encoder should report no clip."
    assert encoder_has_clip(enc), "finetuned encoder should report clip."
    try:
        derive_clip_from_he(he, enc_he_only)
    except RuntimeError:
        pass
    else:
        raise AssertionError("derive_clip_from_he should refuse an he-only encoder.")

    print("encoder_adapter self-check OK: transform parity (spec + real Task B fn), he parity, clip derivation, he-only guard.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _selfcheck()
