# takes a predicted binary mask and turns it into actual engineering measurements
# main idea: skeletonize the crack to get the centerline, then use EDT to get width at each point
# width at skeleton pixel = 2 * distance to nearest background pixel
# then scale pixels to mm using the camera's ground sampling distance

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects


class CrackQuantifier:
    # gsd = mm per pixel, 0.15 is roughly what you get from a DSLR at close range on a bridge
    def __init__(self, gsd_mm_per_pixel: float = 0.15, min_component_size: int = 50):
        self.gsd = gsd_mm_per_pixel
        self.min_component_size = min_component_size

    def clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        # remove tiny noise blobs that aren't real cracks
        mask_bool = binary_mask > 0.5
        cleaned = remove_small_objects(mask_bool, min_size=self.min_component_size)
        cleaned_uint8 = (cleaned * 255).astype(np.uint8)

        # small closing to reconnect crack segments that got broken during prediction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(cleaned_uint8, cv2.MORPH_CLOSE, kernel)
        return closed > 0

    def analyze(self, binary_mask: np.ndarray) -> dict:
        cleaned_mask = self.clean_mask(binary_mask)
        total_pixels = cleaned_mask.size
        crack_pixel_count = int(np.sum(cleaned_mask))

        if crack_pixel_count == 0:
            return {
                "detected": False,
                "crack_area_px": 0,
                "crack_density_pct": 0.0,
                "max_width_mm": 0.0,
                "mean_width_mm": 0.0,
                "total_length_mm": 0.0,
                "aci_severity": "Sound / Intact",
                "recommended_action": "None required."
            }

        # get the centerline of the crack
        skeleton = skeletonize(cleaned_mask)

        # EDT gives distance from each foreground pixel to the nearest edge
        # at skeleton points this equals the local crack radius, so width = 2x
        edt = distance_transform_edt(cleaned_mask)

        skeleton_indices = np.where(skeleton)
        skeleton_radii = edt[skeleton_indices]
        orthogonal_widths_px = skeleton_radii * 2.0

        if len(orthogonal_widths_px) == 0:
            max_width_px = 0.0
            mean_width_px = 0.0
        else:
            max_width_px = float(np.max(orthogonal_widths_px))
            mean_width_px = float(np.mean(orthogonal_widths_px))

        # convert to real-world mm
        max_width_mm = max_width_px * self.gsd
        mean_width_mm = mean_width_px * self.gsd
        total_length_px = float(np.sum(skeleton))
        total_length_mm = total_length_px * self.gsd
        crack_density_pct = (crack_pixel_count / total_pixels) * 100.0

        # ACI 224R-01 thresholds for bridge decks (Table 4.1)
        # tolerable limit under deicing salts is 0.18mm
        if max_width_mm < 0.10:
            aci_severity = "Minor Micro-cracking"
            action = "Routine visual monitoring; within acceptable structural tolerance."
        elif max_width_mm <= 0.18:
            aci_severity = "Tolerable Bridge Deck Limit (ACI 224R)"
            action = "Acceptable upper limit under deicing salts; schedule periodic non-destructive re-inspection."
        elif max_width_mm <= 0.40:
            aci_severity = "Moderate Damage"
            action = "Exceeds environmental threshold; surface hydrophobic sealing/grouting required to prevent rebar corrosion."
        else:
            aci_severity = "Critical Structural Damage"
            action = "Structural integrity compromised; immediate epoxy injection, load limitation, or core drilling assessment mandatory."

        return {
            "detected": True,
            "crack_area_px": crack_pixel_count,
            "crack_density_pct": crack_density_pct,
            "max_width_px": round(max_width_px, 2),
            "max_width_mm": round(max_width_mm, 3),
            "mean_width_px": round(mean_width_px, 2),
            "mean_width_mm": round(mean_width_mm, 3),
            "total_length_mm": round(total_length_mm, 2),
            "aci_severity": aci_severity,
            "recommended_action": action,
            "skeleton": skeleton,
            "distance_map": edt
        }

    def render_overlay(self, rgb_image: np.ndarray, analysis: dict) -> np.ndarray:
        # draws cyan over the crack region, red on the centerline, and a small info box
        overlay = rgb_image.copy()
        if not analysis.get("detected", False):
            return overlay

        skeleton = analysis["skeleton"]
        distance_map = analysis["distance_map"]
        mask = distance_map > 0

        # semi-transparent cyan blend over crack pixels
        overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)

        # skeleton in red on top
        overlay[skeleton] = [255, 0, 0]

        hud_text = [
            f"Max Width: {analysis['max_width_mm']:.2f} mm",
            f"Mean Width: {analysis['mean_width_mm']:.2f} mm",
            f"Total Length: {analysis['total_length_mm']:.1f} mm",
            f"ACI 224R: {analysis['aci_severity']}"
        ]

        cv2.rectangle(overlay, (10, 10), (360, 110), (0, 0, 0), -1)
        for i, line in enumerate(hud_text):
            color = (0, 0, 255) if "Critical" in line or "Moderate" in line else (0, 255, 0)
            if i < 3:
                color = (255, 255, 255)
            cv2.putText(overlay, line, (18, 32 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        return overlay
