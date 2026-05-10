Two-paper research plan
Paper 1 — Calibrated SLAM uncertainty for endoscopic Gaussian maps
Problem. Endo-2DTAM produces a 2D-Gaussian reconstruction, but every Gaussian comes with no calibrated error budget. A surgeon can't tell which regions of the GPS map are reliable vs noise. Stock confidence scores from depth nets and SLAM optimizers are systematically overconfident.

Contribution. First application of split conformal prediction to endoscopic Gaussian SLAM, producing per-region error intervals with distribution-free finite-sample coverage guarantees.

Method (concrete).

Run the existing pipeline (EndoDAC depth → Endo-2DTAM map).
Test-time augmentation: re-run EndoDAC N=4× per frame under {hflip, brightness ±10%, small crop}; per-pixel uncertainty σ_p = std-dev across the 4 predictions.
Per-Gaussian aggregation: for each Gaussian g, sum σ_p over the pixels that contributed to g (Endo-2DTAM's covisibility list), weighted by alpha-compositing weight → σ_g.
Calibration on a held-out C3VD subset with GT depth: compute residual r_g = |predicted depth at g − GT depth at g|, score s_g = r_g / σ_g.
Split conformal: take the ⌈(n+1)(1−α)⌉-th order statistic q of {s_g} on the calibration split.
Test time: any new Gaussian's calibrated error interval is ±q · σ_g, with marginal coverage ≥ 1−α.
Dashboard: alpha-blend each rendered region by 1 − (q · σ_g)/max; render high-uncertainty regions hatched.
Datasets.

C3VD — calibration + in-distribution test (has GT depth + poses).
EndoMapper-Bronchus — out-of-distribution real-OR validation (no GT, qualitative + cross-coverage).
BREAD phantom — cross-domain test.
Baselines.

Endo-2DTAM observation count (raw).
Per-Gaussian opacity (raw).
Uncalibrated TTA std-dev.
Vanilla split conformal on per-pixel depth (no per-Gaussian aggregation).
Metrics.

Empirical coverage at α∈{0.1, 0.05} (target: within ±2% of nominal).
Mean interval width at fixed coverage.
Calibration curve (predicted vs observed coverage).
AUROC: "high-uncertainty region ↔ actually-high-error region".
Ablation: TTA count, aggregation rule (max / mean / weighted), augmentation set.
Milestones (~12 weeks).

Wk 1–2: TTA harness + per-Gaussian aggregation.
Wk 3–4: Conformal calibration + diagnostic plots.
Wk 5–6: C3VD evaluation + ablations.
Wk 7–8: Cross-domain (BREAD / EndoMapper).
Wk 9–10: Dashboard integration + qualitative figures.
Wk 11–12: Paper write-up.
Risks. (1) TTA σ may not correlate with true error in textureless regions → coverage holds but intervals widen unhelpfully. Report it. (2) Cross-domain coverage drift — fall back to weighted / adaptive conformal (Tibshirani 2019) and disclose. (3) Endo-2DTAM internals may need deeper instrumentation for per-Gaussian covisibility — instrument early.

Venue. MICCAI 2026 main, IPCAI 2026, or ISBI.

Paper 2 — Conformal anomaly detection via a population airway prior
Problem. Idea 1 says "this region is uncertain" — the surgeon's actual question is "is this region abnormal?" That requires a learned model of normal anatomy.

Contribution. A normal-airway prior + a calibrated deviation score with distribution-free upper bound on the false-positive rate for region-level anomaly flagging from monocular bronchoscopic video.

Method.

Normal prior trained on ATM'22 (~300 healthy chest-CT airway segmentations):
A. PCA on registered airway meshes — classical Statistical Shape Model, smooth, deterministic, our default.
B. PointNet++ autoencoder — ablation, more expressive.
C. DeepSDF — research direction.
Mesh-to-prior registration: coarse alignment via carina/branch endpoints → non-rigid ICP for fine fit.
Deviation score per region: d_i = ‖observed_vertex_i − prior_match_i‖ / local_radius_i (dimensionless).
Conformal calibration on a held-out normal cohort: take (1−α)-quantile of d_i. Regions exceeding it are flagged "abnormal at FPR ≤ α".
Dashboard: existing trust-channel rendering from paper 1 → reused for the anomaly heatmap.
Datasets.

ATM'22 (split: 70% prior-train, 15% conformal-calibration, 15% normal-test).
LIDC-IDRI airway labels — out-of-cohort normal calibration check.
Pathology test set (TCIA Lung-CT-Diagnosis or hospital-partnered abnormal cases).
BREAD phantom with deliberately-introduced narrowings — synthetic sanity.
C3VD pipeline + Endo-2DTAM map (inherited).
Baselines.

Raw distance-to-prior (no conformal threshold).
One-class SVM in the prior's latent space.
Reconstruction error from prior autoencoder.
Idea-1 uncertainty alone (does high uncertainty already flag abnormality? hopefully not — and that's a useful negative result).
Metrics.

Empirical FPR vs nominal α (the conformal guarantee).
TPR at fixed FPR=10%, 5%.
Region-level AUROC on labeled abnormal cases.
Lesion-localization F1 (overlap with clinician-marked lesion).
Cross-patient generalization (leave-one-cohort-out).
Milestones (~22 weeks, assuming paper 1 done).

Wk 1–3: ATM'22 data pipeline + mesh normalization.
Wk 4–7: Train PCA + autoencoder priors; reconstruction-on-holdout evaluation.
Wk 8–10: Mesh-to-prior registration (the hardest engineering part).
Wk 11–13: Conformal threshold + abnormal test set evaluation.
Wk 14–16: Dashboard integration + small clinician user study.
Wk 17–22: Paper write-up + revisions.
Risks. (1) ATM'22 not diverse enough → train + cross-cohort calibrate. (2) Non-rigid registration brittle on real OR data — use centerline-only as fallback. (3) "Abnormal" is heterogeneous (tumor / stenosis / anatomical variant) — report metrics broken down by category. (4) OR data not exchangeable with ATM'22 — disclose, use weighted conformal.

Venue. MICCAI 2026/27 main, TMI / Medical Image Analysis (journal extension).

How they fit together
Component	Built in P1	Reused in P2	New in P2
Endo-2DTAM map	✓	✓	—
TTA + per-Gaussian uncertainty	✓	✓ (used as a weighting in the deviation score)	—
Split conformal calibration code	✓	✓ (different residual definition)	—
Dashboard trust-channel rendering	✓	✓	—
Population airway prior	—	—	✓ (PCA / PointNet++ / DeepSDF)
Mesh-to-prior registration	—	—	✓
Deviation score	—	—	✓
Anomaly visualization	—	—	✓ (built on top of P1's rendering)
Paper 1's intro positions calibrated SLAM uncertainty as the foundation for downstream clinical decisions, with anomaly detection as the obvious application — a forward reference that doesn't need to be delivered in P1 but makes P2 a clean continuation.

If you can deliver only one: P2 is the higher-impact paper but harder. P1 is the safer first publication and de-risks the conformal pipeline against real GT depth before the OOD generalization battle in P2.
