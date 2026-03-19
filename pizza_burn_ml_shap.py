# -*- coding: utf-8 -*-
"""
Pizza Oven Burn Pattern — ML Part Impact Analysis (XGBoost + SHAP)
===================================================================
Stage 2 Analysis — Applied extension of pizza-oven-burn-analysis

Pipeline:
  [Stage 1] pizza-oven-burn-analysis
              ↓ pizza_burn_measurements.csv
              ↓ oven_maintenance_log.csv
              ↓ maintenance_schedule.csv
  [Stage 2] THIS SCRIPT  ← you are here
              ↓ XGBoost: cycle MAD ← part binary features
              ↓ SHAP TreeExplainer: per-part contribution
              ↓ SHAP Interaction Values: part×part synergy
              ↓ Excel (8 sheets) + 3 plots

Analogy Mapping (Pizza ↔ Semiconductor):
  hearth_stone       ← ESC (Electrostatic Chuck)   largest MAD effect
  dome_lining        ← C-Shroud
  upper_heater_cal   ← Inner Electrode
  lower_heater_cal   ← Outer Electrode
  stone_ring_clean   ← Edge Ring
  ash_removal        ← Regular Cleaning
  door_seal          ← VAT / Seal
  thermocouple_check ← TC / Sensor Check

References:
  Lundberg & Lee (2017) NeurIPS
    "A Unified Approach to Interpreting Model Predictions"
  Lundberg et al. (2020) Nature Machine Intelligence
    "From local explanations to global understanding with explainable AI for trees"
  Lundberg et al. (2018) arXiv:1802.03888
    "Consistent Individualized Feature Attribution for Tree Ensembles"
  Chen & Guestrin (2016) KDD
    "XGBoost: A Scalable Tree Boosting System"
  Strobl et al. (2007) BMC Bioinformatics
    "Bias in random forest variable importance measures"
  Zhao et al. (2025) arXiv:2512.01205
    "Milling Machine Predictive Maintenance Based on ML and SHAP"
  Varoquaux et al. (2017) NeuroImage
    "Assessing and tuning brain decoders, via the Neuroimage Decoding Benchmark"
  Cochrane Handbook Ch.6
    "Choosing effect measures for dichotomous outcomes"

Author: Go
Version: Ver1
"""

import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Optional ML libraries ──────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed.  pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed.  pip install shap")

try:
    from sklearn.model_selection import (cross_val_score,
                                         LeaveOneOut, KFold)
    from sklearn.metrics import r2_score, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed.  pip install scikit-learn")

VERSION    = "Ver1"
TOOL_TITLE = f"Pizza Oven Burn ML Part Impact Analysis (XGBoost+SHAP) {VERSION}"

# ── Stage 1 output filenames (default) ────────────────────────
STAGE1_BURN_CSV  = "pizza_burn_measurements.csv"
STAGE1_MAINT_CSV = "oven_maintenance_log.csv"
STAGE1_SCHED_CSV = "maintenance_schedule.csv"


# ==============================================================
# Font Setup  (Hiragino Sans は使用しない)
# ==============================================================
def setup_fonts():
    import matplotlib.font_manager as fm
    candidates = [
        'Yu Gothic', 'YuGothic', 'Meiryo', 'MS Gothic',
        'Noto Sans CJK JP', 'IPAexGothic', 'DejaVu Sans',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    found = [f for f in candidates if f in available]
    plt.rcParams['font.family'] = 'sans-serif'
    if found:
        plt.rcParams['font.sans-serif'] = (
            found + plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['axes.unicode_minus'] = False

setup_fonts()


# ==============================================================
# Stage 1 CSV Loader
# ==============================================================
def load_stage1_data(data_dir: str):
    """Load Stage 1 output CSVs produced by pizza-oven-burn-analysis.

    Expected files (placed in data_dir):
      pizza_burn_measurements.csv   — 16-point BurnDeviation per pizza
      oven_maintenance_log.csv      — maintenance events per oven/shelf
      maintenance_schedule.csv      — part schedule reference

    If files are absent, a minimal mock dataset is auto-generated
    so the script can run standalone for demonstration.
    """
    bp = os.path.join(data_dir, STAGE1_BURN_CSV)
    mp = os.path.join(data_dir, STAGE1_MAINT_CSV)
    sp = os.path.join(data_dir, STAGE1_SCHED_CSV)

    if not all(os.path.exists(p) for p in [bp, mp, sp]):
        print(f"[STAGE2] Stage 1 CSVs not found in '{data_dir}'.")
        print( "         Auto-generating demo data for standalone run.")
        print( "         For real analysis: copy Stage 1 outputs here first.")
        burn_df, maint_df, sched_df = _generate_demo_data(data_dir)
        return burn_df, maint_df, sched_df

    print(f"[STAGE2] Loading Stage 1 outputs from '{data_dir}' ...")
    burn_df  = pd.read_csv(bp)
    maint_df = pd.read_csv(mp)
    sched_df = pd.read_csv(sp)
    print(f"  pizza_burn_measurements : {len(burn_df):,} rows")
    print(f"  oven_maintenance_log    : {len(maint_df):,} rows")
    print(f"  maintenance_schedule    : {len(sched_df)} rows")
    return burn_df, maint_df, sched_df


# ==============================================================
# Demo Data Generator  (standalone fallback only)
# ==============================================================
def _generate_demo_data(output_dir: str,
                        n_ovens=5, n_shelves=2, n_firings=400,
                        maint_interval=40, seed=42):
    """Generate synthetic Stage 1 CSVs for standalone demo.

    Design principle:
      Each oven/shelf draws a RANDOM maintenance group sequence
      (groups A–E have different part combinations).
      noise_level (= cycle BurnDeviation SD) drops after maintenance
      in proportion to which parts were replaced.
      → Creates a genuine XGBoost-learnable MAD signal.

    PART_MAD_EFFECT values mirror the semiconductor part hierarchy:
      hearth_stone (ESC) >> dome_lining (C-Shroud) > heaters > ring > cleaning
    """
    PART_MAD_EFFECT = {
        'hearth_stone':      2.0,
        'dome_lining':       1.4,
        'upper_heater_cal':  0.8,
        'lower_heater_cal':  0.7,
        'stone_ring_clean':  0.4,
        'ash_removal':       0.25,
        'door_seal':         0.2,
        'thermocouple_check':0.15,
    }
    MAINT_GROUPS = {
        'A': ['ash_removal', 'stone_ring_clean'],
        'B': ['ash_removal', 'stone_ring_clean',
              'upper_heater_cal', 'lower_heater_cal', 'thermocouple_check'],
        'C': ['ash_removal', 'stone_ring_clean', 'dome_lining'],
        'D': ['hearth_stone', 'stone_ring_clean', 'ash_removal'],
        'E': ['hearth_stone', 'dome_lining', 'upper_heater_cal',
              'lower_heater_cal', 'stone_ring_clean',
              'ash_removal', 'door_seal'],
    }
    ANGLES  = [0, 45, 90, 135, 180, 225, 270, 315]
    ZONES   = ['center', 'edge']
    SHELVES = ['Upper', 'Lower']
    RECIPES = ['Margherita', 'Marinara', 'Diavola', 'Quattro_Formaggi']
    os.makedirs(output_dir, exist_ok=True)

    all_burn, all_maint = [], []
    from datetime import timedelta

    for oi in range(1, n_ovens + 1):
        oven_id = f'OVEN_{oi:02d}'
        for si, shelf in enumerate(SHELVES[:n_shelves]):
            rng = np.random.default_rng(seed + oi * 100 + si)
            bias_angle = rng.choice(ANGLES)
            bias_mag   = rng.uniform(0.3, 1.5)
            s_offset   = 0.3 if shelf == 'Upper' else -0.2
            noise      = rng.uniform(1.2, 1.8)
            maint_cnt  = 0
            cycle_id   = 0
            cyc_start  = 0
            t0 = datetime(2024, 1, 1) + timedelta(hours=int(rng.integers(0, 720)))
            grp_seq = list(rng.choice(list(MAINT_GROUPS.keys()),
                                      size=n_firings // maint_interval + 2))

            for firing in range(1, n_firings + 1):
                pid  = f'{oven_id}_{shelf[0]}_{firing:04d}'
                dt   = t0 + timedelta(hours=firing * 4 + int(rng.integers(-1, 3)))
                recipe = rng.choice(RECIPES)
                fsm    = firing - cyc_start
                noise += rng.uniform(0.003, 0.012)

                mtype, mparts, is_m = '', [], False
                if fsm >= maint_interval and firing > 1:
                    maint_cnt += 1; cycle_id += 1; cyc_start = firing; is_m = True
                    grp   = grp_seq[maint_cnt - 1]
                    parts = MAINT_GROUPS[grp][:]
                    mtype = grp
                    eff   = sum(PART_MAD_EFFECT.get(p, 0.1) for p in parts)
                    noise = max(0.3, noise - min(eff * 0.25, noise * 0.85))
                    mparts = parts
                    all_maint.append({
                        'OvenID': oven_id, 'Shelf': shelf,
                        'FiringNumber': firing, 'Date': dt,
                        'MaintenanceCycle': cycle_id,
                        'MaintenanceType': mtype,
                        'Parts': ', '.join(sorted(set(mparts))),
                        'NoiseLevel_After': round(noise, 4),
                    })

                for angle in ANGLES:
                    for zone in ZONES:
                        r   = 0.3 if zone == 'center' else 0.8
                        rad = np.radians(angle)
                        xc  = round(r * np.sin(rad), 4)
                        yc  = round(r * np.cos(rad), 4)
                        ad  = abs(angle - bias_angle)
                        if ad > 180: ad = 360 - ad
                        burn = (float(rng.normal(0, noise))
                                + bias_mag * np.cos(np.radians(ad)) * 0.4
                                + (0.6 if zone == 'edge' else -0.2)
                                + s_offset + 0.3 * yc)
                        all_burn.append({
                            'PizzaID': pid, 'OvenID': oven_id, 'Shelf': shelf,
                            'FiringNumber': firing, 'DateTime': dt,
                            'TotalFirings': firing, 'FiringsSinceMaint': fsm,
                            'Recipe': recipe, 'Angle_deg': angle,
                            'Zone': zone, 'X': xc, 'Y': yc,
                            'BurnDeviation': round(burn, 4),
                            'MaintenanceCycle': cycle_id,
                            'MaintenanceType': mtype if is_m else '',
                            'Parts': (', '.join(sorted(set(mparts)))
                                      if is_m else ''),
                        })

    burn_df  = pd.DataFrame(all_burn)
    maint_df = pd.DataFrame(all_maint)
    sched_rows = []
    for grp, parts in MAINT_GROUPS.items():
        for p in parts:
            sched_rows.append({'MaintenanceGroup': grp, 'Part': p,
                               'MAD_Reduction': PART_MAD_EFFECT.get(p, 0.1)})
    sched_df = pd.DataFrame(sched_rows)

    burn_df.to_csv(os.path.join(output_dir, STAGE1_BURN_CSV), index=False)
    maint_df.to_csv(os.path.join(output_dir, STAGE1_MAINT_CSV), index=False)
    sched_df.to_csv(os.path.join(output_dir, STAGE1_SCHED_CSV), index=False)
    print(f"  Demo CSVs saved to '{output_dir}/'")
    mad_vals = (burn_df[burn_df['MaintenanceCycle'] > 0]
                .groupby(['OvenID','Shelf','MaintenanceCycle'])['BurnDeviation']
                .apply(lambda s: (s - s.mean()).abs().mean()))
    print(f"  Cycle MAD [{mad_vals.min():.3f}, {mad_vals.max():.3f}]  "
          f"mean={mad_vals.mean():.3f}")
    return burn_df, maint_df, sched_df


# ==============================================================
# Step 1: Cycle Summary  (Stage 2 preprocessing)
# ==============================================================
def build_cycle_summary(burn_df: pd.DataFrame,
                        maint_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stage 1 measurements into per-cycle statistics.

    Each row = 1 maintenance cycle for one (OvenID, Shelf).

    Metrics:
      Mean : signed average BurnDeviation (directional bias)
             Ref: Wikipedia "Deviation statistics"
      MAD  : mean(|BurnDev_i − mean(BurnDev)|)
             Robust to outliers; captures burn non-uniformity
             Ref: Wikipedia "Mean absolute deviation"
      SD   : standard deviation (additive; useful for decomposition)

    The ML target is cycle MAD — it quantifies how uneven the burn
    pattern was across all 16 measurement points in that cycle.
    Parts come from the maintenance event that *started* the cycle.
    """
    print("\n[STAGE2] Step 1 — Building cycle summary from Stage 1 data ...")
    burn_df = burn_df.copy()
    burn_df['MaintenanceCycle'] = (
        pd.to_numeric(burn_df['MaintenanceCycle'], errors='coerce')
        .fillna(0).astype(int))

    # Lookup: (OvenID, Shelf, cycle) → parts string
    maint_lookup = {}
    for _, row in maint_df.iterrows():
        key = (row['OvenID'], row['Shelf'], int(row['MaintenanceCycle']))
        maint_lookup[key] = str(row.get('Parts', ''))

    rows = []
    for (oven, shelf, cycle), grp in burn_df.groupby(
            ['OvenID', 'Shelf', 'MaintenanceCycle']):
        if cycle == 0:
            continue
        bd = grp['BurnDeviation'].dropna()
        if len(bd) < 5:
            continue
        mean_v = float(bd.mean())
        mad_v  = float((bd - mean_v).abs().mean())
        sd_v   = float(bd.std())
        rows.append({
            'CycleID': f'{oven}_{shelf}_{cycle:03d}',
            'OvenID': oven, 'Shelf': shelf, 'Cycle': cycle,
            'N': len(bd),
            'Mean': round(mean_v, 5),
            'MAD':  round(mad_v,  5),
            'SD':   round(sd_v,   5),
            'Parts': maint_lookup.get((oven, shelf, cycle), ''),
        })

    df = pd.DataFrame(rows)
    print(f"  Cycles: {len(df)}  "
          f"MAD [{df['MAD'].min():.3f}, {df['MAD'].max():.3f}]  "
          f"mean={df['MAD'].mean():.3f}  SD={df['MAD'].std():.3f}")
    return df


# ==============================================================
# Step 2: ML Feature Matrix
# ==============================================================
def build_feature_matrix(cycle_df: pd.DataFrame):
    """Construct binary part feature matrix for XGBoost.

    X columns:
      Part_{part_name}  0/1 — was this part replaced at cycle start?
      N_in_Cycle        integer — measurement point count (control var)

    Rationale:
      Lundberg et al. (2020) Nature MI:
        TreeExplainer achieves exact Shapley values on tree ensembles
        and handles binary features with high fidelity.
      Strobl et al. (2007) BMC Bioinformatics:
        Same-scale (binary) features minimise MDI bias →
        use SHAP instead of MDI for safety.

    Returns:
      X          pd.DataFrame — feature matrix
      y_mad      pd.Series   — cycle MAD (primary ML target)
      y_mean     pd.Series   — cycle Mean
      y_sd       pd.Series   — cycle SD
      part_names list[str]   — feature names that start with 'Part_'
    """
    print("\n[STAGE2] Step 2 — Building ML feature matrix ...")

    if cycle_df.empty:
        return None, None, None, None, []

    # Collect all part names
    all_parts: set = set()
    for ps in cycle_df['Parts'].dropna():
        for p in str(ps).split(','):
            p = p.strip()
            if p and p != 'nan':
                all_parts.add(p)

    # Keep parts that appear in at least 2 cycles
    counts: dict = {}
    for _, row in cycle_df.iterrows():
        seen: set = set()
        for p in str(row.get('Parts', '')).split(','):
            p = p.strip()
            if p and p != 'nan':
                seen.add(p)
        for p in seen:
            counts[p] = counts.get(p, 0) + 1

    min_cyc   = max(2, int(len(cycle_df) * 0.02))
    valid_prt = sorted([p for p in all_parts if counts.get(p, 0) >= min_cyc])
    print(f"  Parts (>={min_cyc} cycles): {len(valid_prt)}  →  {valid_prt}")

    if len(valid_prt) < 2:
        print("  ERROR: insufficient valid parts (<2). Cannot proceed.")
        return None, None, None, None, []

    rows_X, y_m, y_mn, y_s = [], [], [], []
    for _, row in cycle_df.iterrows():
        here: set = set()
        for p in str(row.get('Parts', '')).split(','):
            p = p.strip()
            if p and p != 'nan':
                here.add(p)
        feat = {f'Part_{p}': (1 if p in here else 0) for p in valid_prt}
        feat['N_in_Cycle'] = int(row.get('N', 0))
        rows_X.append(feat)
        y_m.append(row['MAD']); y_mn.append(row['Mean']); y_s.append(row['SD'])

    X      = pd.DataFrame(rows_X)
    y_mad  = pd.Series(y_m,  name='MAD')
    y_mean = pd.Series(y_mn, name='Mean')
    y_sd   = pd.Series(y_s,  name='SD')
    part_names = [c for c in X.columns if c.startswith('Part_')]
    print(f"  Feature matrix : {X.shape[0]} cycles × {X.shape[1]} features")
    print(f"  Part features  : {len(part_names)}")
    print(f"  Target MAD     : mean={y_mad.mean():.4f}  "
          f"SD={y_mad.std():.4f}  "
          f"range=[{y_mad.min():.4f}, {y_mad.max():.4f}]")
    return X, y_mad, y_mean, y_sd, part_names


# ==============================================================
# Step 3: XGBoost + SHAP Analysis
# ==============================================================
def run_ml_shap(cycle_df: pd.DataFrame, burn_df: pd.DataFrame = None) -> dict:
    """XGBoost + SHAP part impact analysis.

    Target:  Cycle BurnDeviation MAD
    Purpose: Quantify which part replacements most reduce burn non-uniformity

    XGBoost overfitting controls
    (XGBoost official docs "Notes on Parameter Tuning"):
      max_depth      3 (n<50) / 4 (n≥50)
      learning_rate  0.03 (n<50) / 0.05 (n≥50)
      subsample      0.8  — row subsampling
      colsample_bytree 0.8
      reg_alpha      1.0  — L1 regularisation
      reg_lambda     2.0  — L2 regularisation

    Cross-validation
    (Varoquaux et al. 2017 NeuroImage):
      LOO-CV  if n_cycles < 30
      5-Fold  if n_cycles ≥ 30

    SHAP
    (Lundberg & Lee 2017 NeurIPS; Lundberg et al. 2020 Nature MI):
      TreeExplainer — exact Shapley values for tree ensembles
      shap_interaction_values — pairwise synergy between parts
      (Lundberg et al. 2018 arXiv:1802.03888)
    """
    if not all([HAS_XGB, HAS_SHAP, HAS_SKLEARN]):
        miss = [l for l, ok in [('xgboost', HAS_XGB),
                                 ('shap', HAS_SHAP),
                                 ('scikit-learn', HAS_SKLEARN)] if not ok]
        print(f"  SKIP ML: missing libraries {miss}")
        return {}

    print("\n[STAGE2] Step 3 — XGBoost + SHAP Analysis")
    print("  Ref: Lundberg et al. (2020) Nature Machine Intelligence")
    print("  Ref: Chen & Guestrin (2016) KDD — XGBoost")

    X, y_mad, y_mean, y_sd, part_names = build_feature_matrix(cycle_df)
    if X is None or len(X) < 10:
        print("  SKIP: <10 cycles")
        return {}

    result = {'X': X, 'y_mad': y_mad, 'y_mean': y_mean,
              'y_sd': y_sd, 'part_names': part_names}
    n = len(X)

    # ── XGBoost params ────────────────────────────────────────
    params = dict(
        n_estimators   = 300,
        max_depth      = 3 if n < 50 else 4,
        learning_rate  = 0.03 if n < 50 else 0.05,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        reg_alpha      = 1.0,
        reg_lambda     = 2.0,
        min_child_weight = max(3, n // 20),
        gamma          = 0.1,
        random_state   = 42,
        verbosity      = 0,
    )
    print(f"  XGBoost params: max_depth={params['max_depth']}, "
          f"lr={params['learning_rate']}, n_est={params['n_estimators']}, "
          f"mcw={params['min_child_weight']}")

    model = xgb.XGBRegressor(**params)

    # ── Cross-validation ──────────────────────────────────────
    cv = LeaveOneOut() if n < 30 else KFold(n_splits=5, shuffle=True,
                                             random_state=42)
    cv_name = "LOO-CV" if n < 30 else "5-Fold CV"
    print(f"  CV: {cv_name}  (n={n})")
    try:
        cv_sc = cross_val_score(model, X, y_mad, cv=cv,
                                scoring='neg_mean_absolute_error')
        base  = np.abs(y_mad - y_mad.mean()).mean()
        imprv = (base - (-cv_sc.mean())) / base * 100 if base > 0 else 0
        result['cv_scores'] = cv_sc
        print(f"  CV MAE  : {-cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
        print(f"  Baseline: {base:.4f}  Improvement: {imprv:+.1f}%")
    except Exception as e:
        print(f"  CV error: {e}")
        result['cv_scores'] = np.array([])

    # ── Full-data fit (for SHAP) ──────────────────────────────
    model.fit(X, y_mad)
    pred   = model.predict(X)
    tr_r2  = r2_score(y_mad, pred)
    tr_mae = mean_absolute_error(y_mad, pred)
    result.update({'model_r2': tr_r2, 'model_mae': tr_mae, 'model': model})
    print(f"  Train R²={tr_r2:.4f}  MAE={tr_mae:.4f}")

    # ── SHAP TreeExplainer ────────────────────────────────────
    print("  Computing SHAP values (TreeExplainer) ...")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        result.update({'shap_values': sv, 'explainer': explainer})

        fi = pd.DataFrame({
            'Feature':       list(X.columns),
            'Part_Name':     [c.replace('Part_', '') for c in X.columns],
            'Mean_Abs_SHAP': np.abs(sv).mean(axis=0),
            'Mean_SHAP':     sv.mean(axis=0),
            'SD_SHAP':       sv.std(axis=0),
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        result['feature_importance'] = fi

        print("  SHAP Top10:")
        for _, r in fi.head(10).iterrows():
            d = '+' if r['Mean_SHAP'] > 0 else '-'
            print(f"    {r['Part_Name'][:30]:30s}  "
                  f"|SHAP|={r['Mean_Abs_SHAP']:.4f}  "
                  f"Mean={r['Mean_SHAP']:+.4f}({d})")
    except Exception as e:
        print(f"  SHAP error: {e}")
        traceback.print_exc()
        result.update({'shap_values': None,
                       'feature_importance': pd.DataFrame()})

    # ── SHAP Interaction Values ───────────────────────────────
    n_feat = len(X.columns)
    if n_feat <= 30:
        print(f"  Computing SHAP interaction values ({n_feat} features) ...")
        try:
            sv_int = explainer.shap_interaction_values(X)
            imat   = np.abs(sv_int).mean(axis=0)
            np.fill_diagonal(imat, 0)
            result.update({'interaction_matrix': imat,
                           'shap_interaction_values': sv_int})

            cols   = list(X.columns)
            irows  = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    v = float(imat[i, j])
                    if v > 1e-6:
                        irows.append({
                            'Feature_A': cols[i], 'Feature_B': cols[j],
                            'Part_A':    cols[i].replace('Part_', ''),
                            'Part_B':    cols[j].replace('Part_', ''),
                            'Mean_Abs_Interaction': v,
                        })
            int_pairs = (pd.DataFrame(irows)
                         .sort_values('Mean_Abs_Interaction', ascending=False)
                         .reset_index(drop=True) if irows else pd.DataFrame())
            result['interaction_pairs'] = int_pairs
            if not int_pairs.empty:
                print("  Top Interactions:")
                for _, r in int_pairs.head(5).iterrows():
                    print(f"    {r['Part_A'][:20]:20s} × "
                          f"{r['Part_B'][:20]:20s}  "
                          f"|Int|={r['Mean_Abs_Interaction']:.4f}")
        except Exception as e:
            print(f"  Interaction error: {e}")
            result.update({'interaction_matrix': None,
                           'interaction_pairs': pd.DataFrame()})
    else:
        print(f"  SKIP interactions (>{30} features)")
        result.update({'interaction_matrix': None,
                       'interaction_pairs': pd.DataFrame()})

    return result


# ==============================================================
# Step 4: Plot Generation
# ==============================================================
def generate_plots(ml_result: dict, cycle_df: pd.DataFrame,
                   save_dir: str) -> list:
    """Generate Plot11 / Plot12 / Plot13.

    Plot11: SHAP Summary — Bar (mean |SHAP|) + Beeswarm
      Ref: Lundberg & Lee (2017) NeurIPS Fig.1
    Plot12: SHAP Interaction Heatmap
      Ref: Lundberg et al. (2018) arXiv:1802.03888
    Plot13: Top Interaction Pairs — Dependence scatter
      Ref: Lundberg et al. (2018) arXiv:1802.03888
    """
    os.makedirs(save_dir, exist_ok=True)
    if not ml_result:
        return []

    X    = ml_result.get('X')
    sv   = ml_result.get('shap_values')
    fi   = ml_result.get('feature_importance', pd.DataFrame())
    rng  = np.random.default_rng(99)
    plots = []

    if X is None or sv is None:
        return []

    cv_sc  = ml_result.get('cv_scores', np.array([]))
    cv_str = (f"CV MAE={-cv_sc.mean():.4f}±{cv_sc.std():.4f}"
              if len(cv_sc) > 0 else "")
    r2_str = f"Train R²={ml_result.get('model_r2', 0):.3f}"

    # ── Plot11: SHAP Summary (Bar + Beeswarm) ─────────────────
    print("\n[STAGE2] Step 4a — Plot11: SHAP Summary")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        # Left: Bar
        if not fi.empty:
            top_n  = min(20, len(fi))
            fi_top = fi.head(top_n).iloc[::-1]
            c_bar  = ['#e74c3c' if v > 0 else '#3498db'
                      for v in fi_top['Mean_SHAP'].values]
            axes[0].barh(range(top_n), fi_top['Mean_Abs_SHAP'].values,
                         color=c_bar, alpha=0.85, edgecolor='white')
            axes[0].set_yticks(range(top_n))
            axes[0].set_yticklabels(
                [n[:30] for n in fi_top['Part_Name'].values], fontsize=9)
            axes[0].set_xlabel('Mean |SHAP Value|  (impact on Burn MAD)',
                               fontsize=10)
            axes[0].set_title(
                'SHAP Feature Importance\n'
                'Red=raises MAD (worsens)  |  Blue=lowers MAD (improves)',
                fontsize=11)
            axes[0].grid(True, alpha=0.3, axis='x')
            axes[0].legend(handles=[
                mpatches.Patch(color='#e74c3c',
                               label='Raises MAD (↑ burn variability)'),
                mpatches.Patch(color='#3498db',
                               label='Lowers MAD (↓ burn variability)'),
            ], fontsize=8, loc='lower right')

        # Right: Beeswarm
        top_n2 = min(15, len(fi))
        fi_bee = fi.head(top_n2)
        n_show = len(fi_bee)
        if n_show > 0:
            for i, (_, fr) in enumerate(fi_bee.iterrows()):
                fname = fr['Feature']
                if fname not in X.columns:
                    continue
                ci  = list(X.columns).index(fname)
                svc = sv[:, ci]
                fv  = X[fname].values
                yp  = n_show - 1 - i
                j   = rng.normal(0, 0.15, len(svc))
                axes[1].scatter(svc, yp + j,
                                c=['#e74c3c' if v > 0.5 else '#3498db'
                                   for v in fv],
                                s=12, alpha=0.55)
            axes[1].set_yticks(range(n_show))
            axes[1].set_yticklabels(
                [n[:30] for n in fi_bee['Part_Name'].values[::-1]], fontsize=9)
            axes[1].axvline(0, color='gray', lw=1.0, ls='--', alpha=0.7)
            axes[1].set_xlabel('SHAP Value  (per-cycle impact on Burn MAD)',
                               fontsize=10)
            axes[1].set_title(
                'SHAP Beeswarm Plot\n'
                'Red=part present in cycle  |  Blue=part absent',
                fontsize=11)
            axes[1].grid(True, alpha=0.3, axis='x')
            axes[1].legend(handles=[
                Line2D([0],[0], marker='o', color='w',
                       markerfacecolor='#e74c3c', ms=9,
                       label='Part present in cycle'),
                Line2D([0],[0], marker='o', color='w',
                       markerfacecolor='#3498db', ms=9,
                       label='Part absent in cycle'),
            ], fontsize=8, loc='upper right')

        fig.suptitle(
            f'Pizza Oven — ML Part Impact (XGBoost + SHAP) {VERSION}\n'
            f'Stage 2 of pizza-oven-burn-analysis  |  '
            f'Target: Cycle BurnDeviation MAD  |  {r2_str}  {cv_str}',
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p11 = os.path.join(save_dir, f'Plot11_SHAP_Summary_{VERSION}.png')
        fig.savefig(p11, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots.append(('SHAP Summary', p11))
        print(f"  Saved: {p11}")
    except Exception as e:
        print(f"  Plot11 error: {e}"); traceback.print_exc()

    # ── Plot12: Interaction Heatmap ────────────────────────────
    imat = ml_result.get('interaction_matrix')
    if imat is not None and not fi.empty:
        print("[STAGE2] Step 4b — Plot12: SHAP Interaction Heatmap")
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            n_feat = min(15, imat.shape[0])
            fi12   = fi.head(n_feat)
            idx_l, labels = [], []
            for _, fr in fi12.iterrows():
                fname = fr['Feature']
                if fname in X.columns:
                    idx_l.append(list(X.columns).index(fname))
                    labels.append(fr['Part_Name'][:22])

            if len(idx_l) >= 2:
                sub = imat[np.ix_(idx_l, idx_l)].copy()
                np.fill_diagonal(sub, 0)
                im  = ax.imshow(sub, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45,
                                   ha='right', fontsize=9)
                ax.set_yticklabels(labels, fontsize=9)
                plt.colorbar(im, ax=ax,
                             label='Mean |SHAP Interaction Value|')
                ax.set_title(
                    f'SHAP Part Interaction Heatmap {VERSION}\n'
                    f'Off-diagonal = synergistic effect between parts\n'
                    f'(Analogy: simultaneous ESC + C-Shroud replacement)',
                    fontsize=11)
                mx = sub.max() if sub.max() > 0 else 1.0
                for i in range(len(idx_l)):
                    for j in range(len(idx_l)):
                        if i != j and sub[i, j] > 0.001:
                            ax.text(j, i, f'{sub[i, j]:.3f}',
                                    ha='center', va='center', fontsize=7,
                                    color=('white'
                                           if sub[i, j] > mx * 0.6
                                           else 'black'))
            fig.tight_layout()
            p12 = os.path.join(
                save_dir, f'Plot12_SHAP_Interaction_Heatmap_{VERSION}.png')
            fig.savefig(p12, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plots.append(('SHAP Interaction Heatmap', p12))
            print(f"  Saved: {p12}")
        except Exception as e:
            print(f"  Plot12 error: {e}"); traceback.print_exc()

    # ── Plot13: Dependence Plots ───────────────────────────────
    int_pairs = ml_result.get('interaction_pairs', pd.DataFrame())
    if (not int_pairs.empty) and sv is not None:
        print("[STAGE2] Step 4c — Plot13: SHAP Dependence Plots")
        try:
            top4 = int_pairs.head(4)
            np4  = len(top4)
            if np4 > 0:
                fig, ax13 = plt.subplots(1, np4, figsize=(5 * np4, 5))
                if np4 == 1:
                    ax13 = [ax13]
                for idx, (_, pair) in enumerate(top4.iterrows()):
                    if idx >= len(ax13):
                        break
                    ax  = ax13[idx]
                    fa, fb = pair['Feature_A'], pair['Feature_B']
                    if fa in X.columns and fb in X.columns:
                        ca   = list(X.columns).index(fa)
                        sva  = sv[:, ca]
                        fva  = X[fa].values
                        fvb  = X[fb].values
                        ax.scatter(
                            fva + rng.normal(0, 0.02, len(fva)), sva,
                            c=['#e74c3c' if v > 0.5 else '#3498db'
                               for v in fvb],
                            s=25, alpha=0.65, edgecolors='none')
                        ax.set_xlabel(
                            f'{pair["Part_A"][:25]}\n(0=absent, 1=present)',
                            fontsize=9)
                        ax.set_ylabel(
                            f'SHAP value for\n{pair["Part_A"][:20]}',
                            fontsize=9)
                        ax.set_title(
                            f'Interaction:\n{pair["Part_A"][:16]}\n'
                            f'× {pair["Part_B"][:16]}\n'
                            f'|Int|={pair["Mean_Abs_Interaction"]:.4f}',
                            fontsize=9)
                        ax.axhline(0, color='gray', lw=0.7, ls='--')
                        ax.grid(True, alpha=0.3)
                        ax.legend(handles=[
                            Line2D([0],[0], marker='o', color='w',
                                   markerfacecolor='#e74c3c', ms=8,
                                   label=f'{pair["Part_B"][:15]}=Present'),
                            Line2D([0],[0], marker='o', color='w',
                                   markerfacecolor='#3498db', ms=8,
                                   label=f'{pair["Part_B"][:15]}=Absent'),
                        ], fontsize=7, loc='best')

                fig.suptitle(
                    f'Top Part Interaction Pairs — SHAP Dependence {VERSION}\n'
                    "How Part B's presence changes Part A's SHAP contribution",
                    fontsize=11)
                fig.tight_layout()
                p13 = os.path.join(
                    save_dir, f'Plot13_SHAP_Dependence_{VERSION}.png')
                fig.savefig(p13, dpi=150, bbox_inches='tight')
                plt.close(fig)
                plots.append(('SHAP Dependence Plots', p13))
                print(f"  Saved: {p13}")
        except Exception as e:
            print(f"  Plot13 error: {e}"); traceback.print_exc()

    return plots


# ==============================================================
# Step 5: Excel Output
# ==============================================================
def save_excel(burn_df, maint_df, cycle_df, ml_result, output_path):
    """Save all Stage 2 results to multi-sheet Excel.

    Sheets:
      Stage1_Burn_Sample      sampled Stage 1 measurements (≤5,000 rows)
      Stage1_Maintenance_Log  all maintenance events from Stage 1
      Cycle_Summary           per-cycle MAD / Mean / SD + Parts
      ML_Feature_Importance   SHAP-ranked part importance table
      SHAP_Values             per-cycle SHAP value matrix
      SHAP_Interactions       pairwise part interaction scores
      ML_Model_Info           XGBoost params + CV scores + analogy mapping
      Analysis_Info           full reference list
    """
    print(f"\n[STAGE2] Step 5 — Saving Excel → {output_path}")

    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # Stage1_Burn_Sample
    samp = (burn_df.sample(min(5000, len(burn_df)), random_state=42)
            .sort_values(['OvenID', 'Shelf', 'FiringNumber'])
            .reset_index(drop=True))
    samp.to_excel(writer, sheet_name='Stage1_Burn_Sample', index=False)

    maint_df.to_excel(writer, sheet_name='Stage1_Maintenance_Log', index=False)
    cycle_df.to_excel(writer, sheet_name='Cycle_Summary', index=False)

    fi = ml_result.get('feature_importance', pd.DataFrame())
    if not fi.empty:
        fi.to_excel(writer, sheet_name='ML_Feature_Importance', index=False)

    X  = ml_result.get('X')
    sv = ml_result.get('shap_values')
    if X is not None and sv is not None:
        shap_df = pd.DataFrame(sv, columns=[f'SHAP_{c}' for c in X.columns])
        shap_df.insert(0, 'CycleIndex', range(len(shap_df)))
        shap_df.to_excel(writer, sheet_name='SHAP_Values', index=False)

    int_pairs = ml_result.get('interaction_pairs', pd.DataFrame())
    if not int_pairs.empty:
        int_pairs.to_excel(writer, sheet_name='SHAP_Interactions', index=False)

    cv_sc  = ml_result.get('cv_scores', np.array([]))
    n_samp = len(X) if X is not None else 0
    cv_nm  = "LOO-CV" if n_samp < 30 else "5-Fold CV"
    info   = [
        ['Item', 'Value'],
        ['Tool',   TOOL_TITLE],
        ['Version', VERSION],
        ['Date',   datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Stage',  'Stage 2 — applied on top of pizza-oven-burn-analysis'],
        ['Target Variable', 'Cycle BurnDeviation MAD'],
        ['Algorithm', 'XGBRegressor (Chen & Guestrin 2016 KDD)'],
        ['SHAP Method', 'TreeExplainer (Lundberg et al. 2020 Nature MI)'],
        ['CV Method', cv_nm],
        ['n_cycles', n_samp],
        ['Train R2',  f"{ml_result.get('model_r2', 0):.4f}"],
        ['Train MAE', f"{ml_result.get('model_mae', 0):.4f}"],
        ['CV MAE',    f"{-cv_sc.mean():.4f}" if len(cv_sc) > 0 else 'N/A'],
        ['CV MAE SD', f"{cv_sc.std():.4f}"   if len(cv_sc) > 0 else 'N/A'],
        ['max_depth', '3(n<50) / 4(n≥50)'],
        ['learning_rate', '0.03(n<50) / 0.05(n≥50)'],
        ['subsample', '0.8'],
        ['reg_alpha', '1.0 (L1)'],
        ['reg_lambda', '2.0 (L2)'],
        [''],
        ['--- Analogy Mapping (Pizza ↔ Semiconductor) ---', ''],
        ['hearth_stone',       '← ESC (Electrostatic Chuck)'],
        ['dome_lining',        '← C-Shroud'],
        ['upper_heater_cal',   '← Inner Electrode'],
        ['lower_heater_cal',   '← Outer Electrode'],
        ['stone_ring_clean',   '← Edge Ring'],
        ['ash_removal',        '← Regular Cleaning'],
        ['door_seal',          '← VAT / Seal'],
        ['thermocouple_check', '← TC / Sensor Check'],
    ]
    pd.DataFrame(info).to_excel(
        writer, sheet_name='ML_Model_Info', index=False, header=False)

    refs = [['Category', 'Reference'],
        ['SHAP Framework',
         'Lundberg & Lee (2017) NeurIPS '
         '"A Unified Approach to Interpreting Model Predictions"'],
        ['SHAP TreeExplainer',
         'Lundberg et al. (2020) Nature Machine Intelligence '
         '"From local explanations to global understanding with explainable AI for trees"'],
        ['SHAP Interactions',
         'Lundberg et al. (2018) arXiv:1802.03888 '
         '"Consistent Individualized Feature Attribution for Tree Ensembles"'],
        ['XGBoost',
         'Chen & Guestrin (2016) KDD '
         '"XGBoost: A Scalable Tree Boosting System"'],
        ['Feature Importance Bias',
         'Strobl et al. (2007) BMC Bioinformatics '
         '"Bias in random forest variable importance measures"'],
        ['Cross-Validation',
         'Varoquaux et al. (2017) NeuroImage '
         '"Assessing and tuning brain decoders, via the Neuroimage Decoding Benchmark"'],
        ['Predictive Maintenance + SHAP',
         'Zhao et al. (2025) arXiv:2512.01205 '
         '"Milling Machine Predictive Maintenance Based on ML and SHAP"'],
        ['MAD as target',
         'Cochrane Handbook Ch.6: '
         '"Choosing effect measures — direction and magnitude"'],
        ['M2M Variation methodology',
         'Jonathan Pinon: ΔMean + ΔMAD + ΔSD across consecutive cycles'],
    ]
    pd.DataFrame(refs[1:], columns=refs[0]).to_excel(
        writer, sheet_name='Analysis_Info', index=False)

    writer.close()
    print(f"  ✅ Excel saved: {output_path}")


# ==============================================================
# Summary Console Output
# ==============================================================
def print_summary(ml_result: dict, cycle_df: pd.DataFrame):
    fi        = ml_result.get('feature_importance', pd.DataFrame())
    cv_sc     = ml_result.get('cv_scores', np.array([]))
    int_pairs = ml_result.get('interaction_pairs', pd.DataFrame())
    print("\n" + "=" * 62)
    print("  STAGE 2 ANALYSIS SUMMARY")
    print("=" * 62)
    print(f"  Maintenance cycles : {len(cycle_df)}")
    print(f"  Ovens              : {cycle_df['OvenID'].nunique()}")
    print(f"  Train R²           : {ml_result.get('model_r2', 0):.4f}")
    print(f"  Train MAE          : {ml_result.get('model_mae', 0):.4f}")
    if len(cv_sc) > 0:
        print(f"  CV MAE             : "
              f"{-cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

    if not fi.empty:
        mx = fi['Mean_Abs_SHAP'].max()
        if mx > 0:
            print(f"\n  SHAP Part Ranking (Target: Burn MAD):")
            for _, r in fi.head(10).iterrows():
                d   = '+' if r['Mean_SHAP'] > 0 else '-'
                bar = '█' * max(1, int(r['Mean_Abs_SHAP'] / mx * 20))
                print(f"    {r['Part_Name'][:26]:26s} {bar:<20s} "
                      f"|SHAP|={r['Mean_Abs_SHAP']:.4f} ({d})")

    if not int_pairs.empty:
        print(f"\n  Top Part Interactions:")
        for _, r in int_pairs.head(5).iterrows():
            print(f"    {r['Part_A'][:20]:20s} × "
                  f"{r['Part_B'][:20]:20s}  "
                  f"|Int|={r['Mean_Abs_Interaction']:.4f}")
    print("=" * 62)


# ==============================================================
# Main
# ==============================================================
def main():
    print("=" * 62)
    print(f"  {TOOL_TITLE}")
    print()
    print("  Stage 2 — ML applied extension of:")
    print("  https://github.com/shoheimustgoon/pizza-oven-burn-analysis")
    print("=" * 62)

    DATA_DIR   = 'data'       # Stage 1 CSV outputs go here
    OUTPUT_DIR = 'output'     # plots + Excel saved here
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 0: Load Stage 1 outputs ──────────────────────────
    burn_df, maint_df, sched_df = load_stage1_data(DATA_DIR)

    # ── Step 1: Cycle summary ──────────────────────────────────
    cycle_df = build_cycle_summary(burn_df, maint_df)
    if cycle_df.empty:
        print("ERROR: No cycles found in data.")
        sys.exit(1)

    # ── Step 2-3: XGBoost + SHAP ──────────────────────────────
    ml_result = run_ml_shap(cycle_df, burn_df)
    if not ml_result:
        print("ERROR: ML analysis returned no result.")
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────
    print_summary(ml_result, cycle_df)

    # ── Step 4: Plots ──────────────────────────────────────────
    print("\n[STAGE2] Step 4 — Generating plots ...")
    plots = generate_plots(ml_result, cycle_df, save_dir=OUTPUT_DIR)

    # ── Step 5: Excel ──────────────────────────────────────────
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    xls   = os.path.join(OUTPUT_DIR,
                         f'pizza_burn_ml_shap_{VERSION}_{ts}.xlsx')
    save_excel(burn_df, maint_df, cycle_df, ml_result, xls)

    # ── Done ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  COMPLETE")
    print(f"  Excel  : {xls}")
    for name, path in plots:
        print(f"  [{name}] {os.path.basename(path)}")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
