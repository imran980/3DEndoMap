# Step 3: Real-Time Camera Localization

## Background

We have:
- **Registered organ model** with endoscopic mesh aligned inside
- **Registration transform** mapping endo coordinates → organ coordinates
- **Training views** with known camera poses

**Goal:** Given a new endoscopic frame, determine *where* the camera is on the organ model.

## Approach: Render-and-Compare Localization

Since Endo-4DGS can render from any viewpoint, we localize by:

1. **Input:** A new endoscopic frame (image)
2. **Compare:** Render from all known training viewpoints, compute image similarity (SSIM/MSE) to the input
3. **Localize:** The best-matching viewpoint gives us the camera pose
4. **Map:** Apply registration transform to get position on the organ
5. **Display:** Show the position on the 3D organ model

> [!NOTE]
> This is a **proof-of-concept** localization. Production systems would use feature matching (ORB/SuperPoint) or learned pose regression. But render-and-compare validates the full pipeline end-to-end.

## Proposed Changes

### [NEW] [localize_camera.py](file:///home/mi3dr/Endo-4DGS/localize_camera.py)

Takes a query image, compares against all rendered training views, returns the best-matching camera pose mapped to organ coordinates.

- Load trained Endo-4DGS model
- Load registration transform from Step 2
- For a query image: compute similarity against all training renderings
- Return the best-matching pose, mapped to organ coordinates
- Visualize: show query image, best match, and position on organ

---

### [NEW] [run_navigation_demo.py](file:///home/mi3dr/Endo-4DGS/run_navigation_demo.py)

End-to-end demo that simulates the full GPS pipeline:
- Picks test frames as "live" query images
- Localizes each one
- Shows the camera moving along the organ model
- Saves a summary figure

## Verification Plan

### Automated Tests
```bash
# Run localization on test frames
python localize_camera.py --model_path output/endonerf/pulling --configs arguments/endonerf.py

# Full navigation demo
python run_navigation_demo.py --model_path output/endonerf/pulling --configs arguments/endonerf.py
```
