"""
Visual sanity-check for H&E augmentations, cell ID masks, and keypoints.

Produces three grids per chunk:
  - he-patches_{chunk}.png           : original + per-augmentation image (RGB)
  - masks_{chunk}.png                : colored cell ID labels
  - coords_{chunk}.png               : image with keypoints (centroids) overlaid

Each grid: 
  - rows = random patches; 
  - columns = ORIGINAL + one column per augmentation (each applied alone).

Results are stored inside the folder <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / checking / HE_augmentations

NB: Keypoints are computed from nucleus centroids mapped to patch coordinates (i.e. nucleus_centroids parquet files), so they exactly match the centroids used at histoST training time.
"""

import sys
import argparse
import json
import math
import re
import random
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import cv2
from scipy.sparse import load_npz

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import str_cell_id
from src.dataset.hed_augmentation import hed_augment_image
from src.utils.models_settings import BACKBONE_TRANSFORM



# context manager when zip is missing
class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False



def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)



def _read_image_from_zip(zip_file: zipfile.ZipFile, img_name: str) -> np.ndarray:
    """Return H&E image (H,W) uint8."""
    data = zip_file.read(img_name)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(im, dtype=np.uint8)



def _read_mask_np_from_zip(zip_file: zipfile.ZipFile, mask_name_npz: str):
    """Return mask (H,W) int32 from sparse .npz containing cell ID segmentation."""
    buf = zip_file.read(mask_name_npz)
    mat = load_npz(io.BytesIO(buf)).toarray().astype(np.int32)
    return mat



def _parse_slide_id(img_name: str) -> str:
    # Match pattern like "something_123" (number at the end)
    m = re.match(r"(.+)_\d+$", img_name)
    if m:
        return m.group(1)
    # Fallback: remove last underscore segment if it exists
    parts = img_name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else img_name



def _build_shape_pipeline(args, resize_size, crop_size, interpolation):
    # Not using normalization and ToTensor because values will be used for visualization purposes
    ops = []
    if args.backbone.lower() in ["virchow2", "uni2h", "hoptimus0", "hoptimus1"]:
        ops.append(A.Resize(height=resize_size, width=resize_size, interpolation=interpolation, mask_interpolation=cv2.INTER_NEAREST))
        ops.append(A.CenterCrop(height=crop_size, width=crop_size))
    else:
        raise ValueError(f"Unknown backbone '{args.backbone}'")
    # keypoint params like in training
    return A.Compose(
        ops,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True, label_fields=["keypoint_ids"])
    )



def _build_single_aug(name: str, args, img_size: int):
    n = name.lower()
    if n == "rotate90":
        return A.RandomRotate90(p=1.0)
    if n == "hflip":
        return A.HorizontalFlip(p=1.0)
    if n == "vflip":
        return A.VerticalFlip(p=1.0)
    if n == "downscale":
        return A.Downscale(scale_range=(args.downscale_min, args.downscale_max), p=1.0)
    if n == "blur":
        return A.Blur(blur_limit=args.blur_limit, p=1.0)
    if n == "gaussnoise":
        return A.GaussNoise(p=1.0)
    if n == "brightnesscontrast":
        return A.RandomBrightnessContrast(
            brightness_limit=args.brightness_limit, contrast_limit=args.contrast_limit, p=1.0
        )
    if n == "superpixels":
        return A.Superpixels(
            p=1.0, p_replace=0.1, n_segments=200, max_size=int(img_size / 2),
        )
    if n == "zoomblur":
        return A.ZoomBlur(max_factor=1.05, p=1.0)
    if n == "hed":
        return None
    raise ValueError(f"Unknown augmentation '{name}'")



def _apply_aug_with_masks(img_uint8, mask_np, keypoints_xy, keypoint_ids,
                          aug_name, aug_obj, shape_pipeline, hed_sigma, slide_id, compose_seed=None):
    """
    Apply one augmentation (alone) + shape ops to image/masks/keypoints.
    Returns: img, mask_np, keypoints_xy (all transformed)
    """
    # Prepare input dict
    data = {"image": img_uint8}
    if mask_np is not None:
        data["mask_np"] = mask_np.astype(np.int32)
    if keypoints_xy is not None and len(keypoints_xy) > 0:
        data["keypoints"] = [tuple(p) for p in keypoints_xy]
        data["keypoint_ids"] = list(keypoint_ids)
    else:
        data["keypoints"] = []
        data["keypoint_ids"] = []

    # Build the overall transform (aug alone + shape ops), with keypoint params
    aug_ops = []
    if aug_name == "hed":
        # HED is image-only, applied before the spatial ops
        data["image"] = hed_augment_image(img=data["image"], slide_id=slide_id, p=1.0, sigma=hed_sigma)
    else:
        if aug_obj is not None:
            aug_ops.append(aug_obj)

    # Shape ops (same as training)
    # Rebuild shape ops with keypoint params here
    shape_ops = shape_pipeline.transforms if hasattr(shape_pipeline, "transforms") else []
    full_ops = A.Compose(
        aug_ops + list(shape_ops),
        additional_targets={"mask_np": "mask"},
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            label_fields=["keypoint_ids"],
        ),
        seed=compose_seed,
    )

    out = full_ops(**data)

    img_t = out["image"]
    mask_np_t = out.get("mask_np", None)
    kpts_t = out.get("keypoints", [])

    return img_t, mask_np_t, np.asarray(kpts_t, dtype=np.float32)



def _label_to_color(u: int):
    if u == 0:
        return (245, 245, 245)
    # simple deterministic hash to color
    v = (u * 2654435761) & 0xFFFFFFFF
    return ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)



def _colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    # vectorize via unique
    uniq = np.unique(label_map)
    lut = {int(u): _label_to_color(int(u)) for u in uniq}
    for u, col in lut.items():
        out[label_map == u] = col
    return out



def _render_mask_np(mask_np: np.ndarray) -> np.ndarray:
    return _colorize_label_map(mask_np)



def _overlay_keypoints_on_image(img: np.ndarray, kpts_xy: np.ndarray, radius: int = 3) -> np.ndarray:
    if kpts_xy is None or len(kpts_xy) == 0:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for (x, y) in kpts_xy:
        x0, y0 = int(round(x - radius)), int(round(y - radius))
        x1, y1 = int(round(x + radius)), int(round(y + radius))
        draw.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
        draw.ellipse([x0+1, y0+1, x1-1, y1-1], outline=(255, 255, 255), width=1)
    return np.array(pil, dtype=np.uint8)



def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Robust text size helper across Pillow versions.
    Uses draw.textbbox when available, falls back to font.getbbox and then to font.getsize.
    Returns (w, h).
    """
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    if hasattr(font, "getsize"):
        return font.getsize(text)
    return (8 * len(text), 12)



def _draw_grid(images_per_row, titles, row_captions, cell_w, cell_h, margin=16, pad=8, title_h=28, caption_w=260):
    
    cols = len(titles) + 1
    rows = len(images_per_row) + 1

    W = margin * 2 + caption_w + (cell_w * len(titles)) + pad * (len(titles) - 1)
    H = margin * 2 + title_h + (cell_h * len(images_per_row)) + pad * (len(images_per_row) - 1)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Header
    x0 = margin + caption_w
    y0 = margin
    for j, t in enumerate(titles):
        tw, th = _text_size(draw, t, font)
        tx = x0 + j * (cell_w + pad) + (cell_w - tw) // 2
        ty = y0 + (title_h - th) // 2
        draw.text((tx, ty), t, fill=(0, 0, 0), font=font)

    # Rows
    for i, row_imgs in enumerate(images_per_row):
        cy = margin + title_h + i * (cell_h + pad)
        # Caption box
        draw.rectangle([margin, cy, margin + caption_w - 1, cy + cell_h - 1],
                       fill=(245, 245, 245), outline=(220, 220, 220), width=1)
        caption = row_captions[i]
        # Wrap simple
        max_chars = max(12, caption_w // 8)
        parts, line = [], []
        for w in caption.split():
            if len(" ".join(line + [w])) <= max_chars:
                line.append(w)
            else:
                parts.append(" ".join(line)); line = [w]
        if line: parts.append(" ".join(line))
        total_th = sum(_text_size(draw, p, font)[1] for p in parts)
        yy = cy + (cell_h - total_th) // 2
        for p in parts:
            tw, th = _text_size(draw, p, font)
            draw.text((margin + 8, yy), p, fill=(0, 0, 0), font=font)
            yy += th

        # Images
        x_base = margin + caption_w
        for j, im_np in enumerate(row_imgs):
            im_pil = Image.fromarray(im_np)
            x = x_base + j * (cell_w + pad)
            canvas.paste(im_pil, (x, cy))
    return canvas



def _compute_keypoints_from_centroids(bbox_row: dict,
                                      centroids_df: pd.DataFrame,
                                      mask_np: np.ndarray,
                                      max_points: int,
                                      rng: random.Random):
    """
    Return keypoints in patch coordinates:
        (centroid_x - x_min_patch, centroid_y - y_min_patch)
    """
    if bbox_row is None or centroids_df is None or mask_np is None:
        return np.empty((0, 2), np.float32), []

    x_min, y_min, x_max, y_max = (bbox_row[k] for k in ("x_min", "y_min", "x_max", "y_max"))

    # Cell ids present in the cell-token mask
    ids_numeric = np.unique(mask_np)
    ids_numeric = ids_numeric[ids_numeric != 0]  # drop background
    if ids_numeric.size == 0:
        return np.empty((0, 2), np.float32), []
    ids_str = {str_cell_id(int(u)) for u in ids_numeric}
    
    # Keep centroids whose cell_id is in those IDs
    df_patch = centroids_df[centroids_df["cell_id"].isin(ids_str)]
    
    # Match histoST training: Ensure centroids really falls inside the patch bbox = removing centroids that are outside the patch
    df_patch = df_patch.query("@x_min <= x < @x_max and @y_min <= y < @y_max")
    
    if df_patch.empty:
        return np.empty((0, 2), np.float32), []

    # Optional subsample for visualization
    if len(df_patch) > max_points:
        df_patch = df_patch.sample(max_points, random_state=rng.randrange(2**32))

    # Convert slide to patch coords then clip to [0, W-1] × [0, H-1]
    H_patch, W_patch = mask_np.shape[:2]
    kx = (df_patch["x"].to_numpy(dtype=np.float32) - float(x_min))
    ky = (df_patch["y"].to_numpy(dtype=np.float32) - float(y_min))
    np.clip(kx, 0, max(0, W_patch - 1), out=kx)
    np.clip(ky, 0, max(0, H_patch - 1), out=ky)
    kpts_xy = np.stack([kx, ky], axis=1)

    # Alphanumeric cell_id values (e.g. "aaablgdi-1") to consecutive integer labels that Albumentations accepts
    kp_ids = pd.factorize(df_patch["cell_id"])[0].tolist()
    
    return kpts_xy, kp_ids






def main(args):

    # Seed
    set_global_seeds(args.seed)
    rng = random.Random(args.seed)

    # Paths
    dataset_path = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id)
    out_dir = dataset_path / "checking" / "HE_augmentations"/ "specific_patches" if args.patch_names_list is not None and len(args.patch_names_list) > 0 else dataset_path / "checking" / "HE_augmentations"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n* Launched code to check H&E augmentations, results will be saved in: {out_dir}")

    # Extract specific backbone pipeline (resize_size, crop_size, interpolation)
    print(f"* Using backbone: {args.backbone}")
    backbone_transform = BACKBONE_TRANSFORM.get(args.backbone.lower())
    if backbone_transform is None:
        raise ValueError(f"Unknown backbone '{args.backbone}'")
    resize_size = backbone_transform["resize_size"]
    crop_size = backbone_transform["crop_size"]
    interpolation = backbone_transform["interpolation"]

    # Save config
    config_to_save = {
        "seed": args.seed, "fold": 'train', 
        "backbone": args.backbone,
        "resize_size": resize_size, "crop_size": crop_size, "interpolation": interpolation,
        "img_size": args.img_size,
        "aug_list": [x.strip().lower() for x in args.aug_list.split(",") if x.strip()],
        "params": {
            "downscale": {"min": args.downscale_min, "max": args.downscale_max},
            "blur": {"blur_limit": args.blur_limit},
            "brightnesscontrast": {"brightness_limit": args.brightness_limit, "contrast_limit": args.contrast_limit},
            "superpixels": {"p_replace":0.1, "n_segments": 200, "max_size": (args.img_size / 2)},
            "zoomblur": {"max_factor": 1.05},
            "hed": {"sigma": args.hed_sigma}
        },
        "max_keypoints": args.max_keypoints
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"* Config:\n{config_to_save}")

    # Read fold list
    fold_csv = dataset_path / f"cell_count_train.csv"
    if not fold_csv.is_file():
        raise FileNotFoundError(f"Cannot find {fold_csv}")
    df_fold = pd.read_csv(fold_csv)
    if "patch_id" not in df_fold.columns:
        raise ValueError(f"'patch_id' column not found in {fold_csv}")
    all_patch_ids = df_fold["patch_id"].tolist()
    if len(all_patch_ids) == 0:
        raise RuntimeError(f"No patch IDs in {fold_csv}")
    print(f"* Found {len(all_patch_ids)} patch IDs in fold train.")

    if args.patch_names_list is not None and len(args.patch_names_list) > 0:
        # Use specific patch names provided by user
        selected = []
        for name in args.patch_names_list:
            if name in all_patch_ids:
                selected.append(name)
            else:
                print(f"[WARN] patch '{name}' not found in fold train")
        if len(selected) == 0:
            raise RuntimeError("No valid patch names found in the provided list.")
        print(f"* Selected {len(selected)} specific patches to visualize.")
    else:
        # Sample patches
        selected = all_patch_ids if args.num_patches >= len(all_patch_ids) else rng.sample(all_patch_ids, args.num_patches)
        print(f"* Selected {len(selected)} patches to visualize.")

    # Zips
    img_zip_path = dataset_path / "images.zip"
    if not img_zip_path.is_file():
        raise FileNotFoundError(f"Cannot find {img_zip_path}")

    masks_zip_path = dataset_path / "masks_cell_ids_nuclei.zip"
    has_masks = masks_zip_path.is_file()
    if not has_masks:
        print("[INFO] masks_cell_ids_nuclei.zip not found -> 'masks_*.png' and 'coords_*.png' will be skipped.")

    # Load patch bounding boxes
    patch_bboxes_path = dataset_path / "patch_bboxes.parquet"
    if not patch_bboxes_path.is_file():
        raise FileNotFoundError(f"Cannot find {patch_bboxes_path}")
    df_patch_bboxes = pd.read_parquet(patch_bboxes_path)
    patch_lookup = {
        row.patch_id: {
            "x_min": row.x_min, "y_min": row.y_min,
            "x_max": row.x_max, "y_max": row.y_max
        }
        for row in df_patch_bboxes.itertuples()
    }

    # Lazy-load centroids, one parquet per slide
    centroid_dir = dataset_path / "nucleus_centroids"
    _centroid_cache: dict[str, pd.DataFrame | None] = {}

    def _load_centroids(slide: str):
        if slide not in _centroid_cache:
            pq = centroid_dir / f"nucleus_centroids_{slide}.parquet"
            if pq.is_file():
                _centroid_cache[slide] = pd.read_parquet(pq)
            else:
                print(f"[WARN] missing centroid file -> {pq}")
                _centroid_cache[slide] = None
        return _centroid_cache[slide]

    # Pipelines and augs
    shape_pipeline = _build_shape_pipeline(args, resize_size, crop_size, interpolation)
    aug_names = [x.strip().lower() for x in args.aug_list.split(",") if x.strip()]
    titles = ["ORIGINAL"] + [n.upper() for n in aug_names]
    aug_objs = {n: _build_single_aug(n, args, args.img_size) for n in aug_names}

    # Determine output cell size
    with zipfile.ZipFile(img_zip_path, "r") as z:
        ref_img = _read_image_from_zip(z, f'{selected[0]}.png')
        cell_h, cell_w = ref_img.shape[:2]  # use the raw patch size for grid cells

    rows_per_png = max(1, args.max_per_png)
    n_chunks = math.ceil(len(selected) / rows_per_png)

    print(f"\n* Processing to get .png using {n_chunks} chunks of {rows_per_png} rows each.")
    with zipfile.ZipFile(img_zip_path, "r") as z_img, \
         zipfile.ZipFile(masks_zip_path, "r") if has_masks else _nullcontext() as z_mask:

        for chunk_idx in range(n_chunks):
            print(f"[Processing] Chunk {chunk_idx + 1}/{n_chunks}")
            
            i0, i1 = chunk_idx * rows_per_png, min((chunk_idx + 1) * rows_per_png, len(selected))
            # Containers for the four outputs
            rows_img, rows_mask, rows_coords = [], [], []
            caps = []
            caps_mask = []
            caps_coords = []

            for patch_id in selected[i0:i1]:
                
                slide_id = _parse_slide_id(patch_id)
                img_name = patch_id + '.png'
                mask_name = patch_id + '.npz'

                # Load inputs
                img_np = _read_image_from_zip(z_img, img_name)

                mask_np = None
                if has_masks and mask_name in z_mask.namelist():
                    mask_np = _read_mask_np_from_zip(z_mask, mask_name)

                # Keypoints (real nucleus centroids -> patch coords)
                bbox_row = patch_lookup.get(patch_id)
                centroids_df = _load_centroids(slide_id)
                kpts_xy, kp_ids = _compute_keypoints_from_centroids(bbox_row, centroids_df, mask_np, args.max_keypoints, rng)

                # ORIGINAL (no shape ops at all — truly raw)
                orig_img = img_np
                orig_masks = mask_np
                orig_kpts = kpts_xy
                # Compose row entries
                img_row = [orig_img]
                mask_row = [_render_mask_np(orig_masks)] if (orig_masks is not None) else []
                coords_row = [_overlay_keypoints_on_image(orig_img, orig_kpts)]

                caption = f"{patch_id}  |  slide={slide_id}"

                # Aug columns
                for aug_n in aug_names:
                    aug_obj = aug_objs[aug_n]
                    aug_seed = rng.randrange(2**32)
                    aug_img, aug_masks, aug_kpts = _apply_aug_with_masks(
                        img_np, mask_np, kpts_xy, kp_ids,
                        aug_name=aug_n, aug_obj=aug_obj, shape_pipeline=shape_pipeline,
                        hed_sigma=args.hed_sigma, slide_id=slide_id,
                        compose_seed=aug_seed,
                    )
                    img_row.append(aug_img)
                    if orig_masks is not None:
                        mask_row.append(_render_mask_np(aug_masks))
                    coords_row.append(_overlay_keypoints_on_image(aug_img, aug_kpts))

                rows_img.append(img_row)
                if mask_np is not None:
                    rows_mask.append(mask_row)
                    caps_mask.append(caption)

                caps.append(caption)

                # Only include rows that truly have keypoints, and keep captions aligned
                if len(orig_kpts) > 0:
                    rows_coords.append(coords_row)
                    caps_coords.append(caption)

            # Save image grid
            grid_img = _draw_grid(rows_img, titles, caps, cell_w, cell_h, margin=20, pad=8, title_h=28, caption_w=260)
            out_png_img = out_dir / f"he-patches_{chunk_idx:03d}.png"
            grid_img.save(out_png_img); print(f"[Saved] {out_png_img}")

            # Save mask grid (if available)
            if rows_mask:
                grid_tok = _draw_grid(rows_mask, titles, caps_mask, cell_w, cell_h, margin=20, pad=8, title_h=28, caption_w=260)
                out_png_t = out_dir / f"masks_{chunk_idx:03d}.png"
                grid_tok.save(out_png_t); print(f"[Saved] {out_png_t}")
            else:
                print(f"[Skip] masks_{chunk_idx:03d}.png (no masks_cell_ids_nuclei.zip or entries missing).")

            # Save coords grid (if available)
            if rows_coords:
                grid_coords = _draw_grid(rows_coords, titles, caps_coords, cell_w, cell_h, margin=20, pad=8, title_h=28, caption_w=260)
                out_png_c = out_dir / f"coords_{chunk_idx:03d}.png"
                grid_coords.save(out_png_c); print(f"[Saved] {out_png_c}")
            else:
                print(f"[Skip] coords_{chunk_idx:03d}.png (no centroid/keypoint data for these patches).")

    print(f"Done.")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visual check of H&E augmentations + masks + nucleus centroids.")
    
    # Data
    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--num_patches", type=int, 
                        default=30, 
                        help="Total number of random patches to visualize.")
    
    parser.add_argument("--patch_names_list", nargs="+", type=str, 
                        default=None, 
                        help="Space-separated list of specific patch names to visualize (overrides num_patches). "
                             "Example: --patch_names_list heart_s0_1089 colon_s1_7291 skin_s1_1905 breast_s1_19951 tonsil_s0_5116 "
                             "Saved PNGs will be in 'specific_patches' subfolder.")
    
    parser.add_argument("--max_per_png", type=int, 
                        default=5, 
                        help="Rows (patches) per PNG.")
    

    # Backbone / shape pipeline
    parser.add_argument("--backbone", type=str, default="uni2h", choices=["virchow2", "uni2h", "hoptimus0", "hoptimus1"], help="Backbone to mirror training shape ops.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size for augmentations.")

    # Augmentation selection
    parser.add_argument("--aug_list", type=str,
                        default="rotate90,hflip,vflip,downscale,blur,gaussnoise,brightnesscontrast,superpixels,zoomblur,hed",
                        help="Comma-separated list of augmentations to show (each alone).")

    # Magnitudes
    parser.add_argument("--downscale_min", type=float, default=0.88, help="Min downscale factor.")
    parser.add_argument("--downscale_max", type=float, default=0.97, help="Max downscale factor.")
    parser.add_argument("--blur_limit", nargs=2, type=int, default=[3, 5], metavar=("MIN", "MAX"), help="Blur limit (min max) for Blur augmentation. Only odd kernel sizes are used to ensure the blur has a clear center pixel.")
    parser.add_argument("--brightness_limit", type=float, default=0.12, help="Brightness limit for RandomBrightnessContrast augmentation.")
    parser.add_argument("--contrast_limit", type=float, default=0.12, help="Contrast limit for RandomBrightnessContrast augmentation.")
    parser.add_argument("--hed_sigma", type=float, default=0.2, help="Sigma for HED augmentation")

    # Keypoints
    parser.add_argument("--max_keypoints", type=int, default=10000, help="Max points to display per patch.")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    main(args)