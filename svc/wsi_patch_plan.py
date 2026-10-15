"""Choose the TRIDENT sampling level while keeping a 512 px / 40x field of view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SUPPORTED_MAGNIFICATIONS = (5, 10, 20, 40)
TARGET_PATCH_SIZE = 512


def _number(properties: dict[str, Any], names: tuple[str, ...]) -> float | None:
    lower_properties = {str(key).lower(): value for key, value in properties.items()}
    for name in names:
        value = lower_properties.get(name.lower())
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if result > 0:
            return result
    return None


def _inspect_properties(wsi_path: Path) -> dict[str, Any]:
    errors = []
    for module_name, class_name in (("tiffslide", "TiffSlide"), ("openslide", "OpenSlide")):
        try:
            module = __import__(module_name, fromlist=[class_name])
            slide = getattr(module, class_name)(str(wsi_path))
            try:
                return dict(slide.properties)
            finally:
                slide.close()
        except Exception as error:
            errors.append(f"{module_name}: {error}")
    raise RuntimeError("Could not read WSI magnification metadata (" + "; ".join(errors) + ")")


def patch_plan(
    properties: dict[str, Any],
    overlap_percentage: int,
    *,
    native_magnification: float | None = None,
    mpp_x: float | None = None,
    mpp_y: float | None = None,
) -> dict[str, float | int]:
    """Plan a 128 µm patch field and a 512 px UNI encoder input."""
    if overlap_percentage not in (0, 50, 75):
        raise ValueError("overlap percentage must be 0, 50, or 75")

    if native_magnification is None:
        native_magnification = _number(properties, (
            "openslide.objective-power", "aperio.appmag", "hamamatsu.sourcelens",
            "tiffslide.objective-power", "objective-power",
        ))
    if mpp_x is None:
        mpp_x = _number(properties, (
            "openslide.mpp-x", "tiffslide.mpp-x", "aperio.mpp", "mpp-x", "mpp",
        ))
    if mpp_y is None:
        mpp_y = _number(properties, (
            "openslide.mpp-y", "tiffslide.mpp-y", "mpp-y",
        ))
    if native_magnification is None and mpp_x is not None:
        # Common scanner calibration is about 10 µm per pixel at 1x.
        native_magnification = 10.0 / mpp_x
    if native_magnification is None and mpp_x is None:
        raise ValueError(
            "The WSI has neither objective-power nor MPP metadata. Set "
            "WSI_NATIVE_MAGNIFICATION and WSI_MPP_X (and WSI_MPP_Y if needed) "
            "to provide the missing calibration."
        )
    if native_magnification is None:
        raise ValueError("Could not estimate native magnification; provide WSI_NATIVE_MAGNIFICATION")
    if native_magnification <= 0:
        raise ValueError(f"native magnification must be positive, got {native_magnification}")
    if mpp_x is None:
        # The slide has objective power but no explicit calibration. Record
        # this conventional estimate in the run summary and validate the H5
        # tile geometry against it before inference.
        mpp_x = 10.0 / native_magnification
    if mpp_x <= 0:
        raise ValueError(f"mpp_x must be positive, got {mpp_x}")
    if mpp_y is None:
        mpp_y = mpp_x
    if mpp_y <= 0:
        raise ValueError(f"mpp_y must be positive, got {mpp_y}")

    # Use the highest standard TRIDENT level supported by the slide. This
    # avoids inventing a 40x image from a lower-resolution scan.
    mag = max((candidate for candidate in SUPPORTED_MAGNIFICATIONS
               if candidate <= native_magnification + 1e-6), default=SUPPORTED_MAGNIFICATIONS[0])
    # Convert the desired 512 px @ 40x (128 µm) field to the slide's actual
    # MPP and chosen TRIDENT level. At 20x / 0.5 MPP this is 256 px.
    mpp_x_at_mag = mpp_x * native_magnification / mag
    mpp_y_at_mag = mpp_y * native_magnification / mag
    patch_size = int(round((TARGET_PATCH_SIZE * 0.25) / ((mpp_x_at_mag + mpp_y_at_mag) / 2.0)))
    if patch_size <= 0:
        raise ValueError(f"Computed invalid TRIDENT patch size: {patch_size}")
    overlap_px = int(round(patch_size * overlap_percentage / 100.0))
    return {
        "magnification": int(mag),
        "patch_size": patch_size,
        "overlap_px": overlap_px,
        "encoder_img_size": TARGET_PATCH_SIZE,
        "mpp_x": float(mpp_x),
        "mpp_y": float(mpp_y),
        "native_magnification": float(native_magnification),
        "expected_field_of_view_um": TARGET_PATCH_SIZE * 0.25,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wsi", type=Path)
    parser.add_argument("--overlap-percentage", type=int, default=0)
    parser.add_argument("--native-magnification", type=float, default=None)
    parser.add_argument("--mpp-x", type=float, default=None)
    parser.add_argument("--mpp-y", type=float, default=None)
    args = parser.parse_args(argv)
    if not args.wsi.is_file():
        raise FileNotFoundError(args.wsi)
    properties = _inspect_properties(args.wsi)
    override = args.native_magnification
    mpp_x = args.mpp_x
    mpp_y = args.mpp_y
    env = __import__("os").environ
    if override is None:
        override_text = env.get("WSI_NATIVE_MAGNIFICATION")
        if override_text:
            override = float(override_text)
    if mpp_x is None and env.get("WSI_MPP_X"):
        mpp_x = float(env["WSI_MPP_X"])
    if mpp_y is None and env.get("WSI_MPP_Y"):
        mpp_y = float(env["WSI_MPP_Y"])
    plan = patch_plan(
        properties, args.overlap_percentage,
        native_magnification=override, mpp_x=mpp_x, mpp_y=mpp_y,
    )
    print("\t".join(str(plan[key]) for key in (
        "magnification", "patch_size", "overlap_px", "encoder_img_size",
        "mpp_x", "mpp_y", "native_magnification", "expected_field_of_view_um",
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
