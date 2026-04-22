# Task: Surgical Navigation Pipeline

## Step 1: Inner Surface Reconstruction ✅
- [x] `extract_surface.py` — TSDF fusion → mesh + camera trajectory

## Step 2: CT Registration & GPS Visualization ✅
- [x] `create_synthetic_organ.py` — synthetic organ mesh enclosing reconstruction
- [x] `register_to_organ.py` — ICP alignment between endo mesh and organ
- [x] `visualize_navigation.py` — interactive 3D GPS view
- [x] Test on pulling model ✅
- [x] Test on cutting model ✅
