"""
Monocular pose estimation for "drop in any RGB video" use cases.

Implements a simple ORB + RANSAC essential-matrix visual odometry
that produces a 4x4 c2w trajectory from a list of RGB frames. No
external SLAM dependency.

Limitations of this VO (be honest about them):
- **Scale ambiguity.** recoverPose() normalizes the translation
  vector to unit length. For tubular endoscopy where the camera
  moves smoothly, we anchor the absolute scale by setting each
  segment's translation magnitude to a per-frame median depth
  estimate (`scale_per_frame_mm`).
- **Drift.** No loop closure. On a short clip (< ~30 s of motion)
  this is usually tolerable; for longer clips, swap in proper
  SLAM (Endo-2DTAM / ORB-SLAM3 / DROID-SLAM) — same output format.
- **Failure handling.** If a frame can't be matched (low texture,
  motion blur, saturation), we copy the previous pose and continue
  rather than crash.

This module is the "Phase 1" deliverable from the README: dashboard
runs from RGB-only video. Treat it as a baseline to be replaced
with Endo-2DTAM once integrated.
"""

from typing import List, Optional, Sequence  # noqa: F401
import numpy as np
import cv2


def _intrinsic_from_hfov(hfov_deg: float, image_hw):
    H, W = image_hw
    fx = fy = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    K = np.array([[fx, 0.0, W / 2.0],
                  [0.0, fy, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


class MonocularVO:
    """ORB + RANSAC essential-matrix VO.

    Args:
      hfov_deg: horizontal FOV of the camera in degrees.
      image_hw: (H, W) of the input frames (assumed constant).
      n_features: ORB target keypoint count.
      min_matches: minimum inlier matches required to trust a pose update.
    """

    def __init__(self, hfov_deg: float, image_hw, n_features: int = 2000,
                 min_matches: int = 30):
        self.K = _intrinsic_from_hfov(hfov_deg, image_hw)
        self.image_hw = image_hw
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches

    def estimate(self, frames: Sequence[np.ndarray],
                 scale_per_frame_mm: Optional[Sequence[float]] = None) -> List[np.ndarray]:
        """Run VO across a list of RGB frames (HxWx3 uint8).

        Returns a list of 4x4 c2w matrices, the same length as `frames`.
        Frame 0's pose is the identity (defines world origin).

        scale_per_frame_mm: optional per-frame absolute-scale anchor in mm
            (e.g. each frame's median depth). When given, the unit-length
            t from recoverPose is rescaled to this value, anchoring the
            trajectory to metric units.
        """
        n = len(frames)
        if n == 0:
            return []
        poses = [np.eye(4, dtype=np.float64)]
        prev_kp, prev_des = None, None
        n_failed = 0

        for i in range(n):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)

            if i == 0:
                prev_kp, prev_des = kp, des
                continue

            relative_c2w = self._pair_pose(
                prev_kp, prev_des, kp, des,
                scale_mm=(scale_per_frame_mm[i]
                          if scale_per_frame_mm is not None else None),
            )
            if relative_c2w is None:
                n_failed += 1
                poses.append(poses[-1].copy())
            else:
                poses.append(poses[-1] @ relative_c2w)

            prev_kp, prev_des = kp, des

        if n_failed:
            print(f"[VO] {n_failed}/{n-1} frames had insufficient matches "
                  f"(pose carried over from previous frame).")
        return poses

    def _pair_pose(self, kp1, des1, kp2, des2, scale_mm: Optional[float]):
        """Relative c2w of camera 2 in camera 1's frame, or None on failure."""
        if des1 is None or des2 is None:
            return None
        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return None

        matches = self.matcher.match(des1, des2)
        if len(matches) < self.min_matches:
            return None

        pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=1.0,
        )
        if E is None or mask is None or int(mask.sum()) < self.min_matches:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        # cv2.recoverPose returns R, t such that x2 = R x1 + t. The pose
        # of camera 2 in camera 1's frame (c2w_2_in_1) is the inverse of
        # this rigid transform: rotation R^T, translation -R^T t. Easier
        # path: invert the 4x4.
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = t.flatten()
        T_c2w_rel = np.linalg.inv(T_w2c)

        if scale_mm is not None and scale_mm > 1e-3:
            tr = T_c2w_rel[:3, 3]
            n = np.linalg.norm(tr)
            if n > 1e-6:
                T_c2w_rel[:3, 3] = tr * (float(scale_mm) / n)
        return T_c2w_rel


def write_pose_txt(poses: Sequence[np.ndarray], path: str) -> None:
    """Save 4x4 c2w matrices as 16 floats per line.

    The matrix is *transposed* before flattening so that
    parse_pose_txt() (which reshapes 16 floats then .T) round-trips.
    """
    with open(path, "w") as f:
        for p in poses:
            flat = p.T.reshape(-1).tolist()
            f.write(" ".join(f"{v:.8f}" for v in flat) + "\n")


def parse_pose_txt(path: str) -> List[np.ndarray]:
    """Inverse of write_pose_txt — read 4x4 c2w matrices from a text file.

    Each line: 16 floats. Lines with 12 floats are also accepted (the
    last row is assumed to be [0,0,0,1]).
    """
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4).T)
            elif len(vals) == 12:
                m = np.eye(4)
                m[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(m)
    return poses
