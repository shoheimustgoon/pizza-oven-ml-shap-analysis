# 🍕 Pizza Oven Burn — ML Part Impact Analysis (XGBoost + SHAP)

> **Applied extension of [pizza-oven-burn-analysis](https://github.com/shoheimustgoon/pizza-oven-burn-analysis)**

> **[🇯🇵 日本語の説明はこちら (Click here for Japanese Description)](#japanese-description)**

---

## 📖 Overview

This project is **Stage 2** of the Pizza Oven Burn Pattern Analysis series.

It takes the CSV outputs from Stage 1 and applies **machine learning (XGBoost) and SHAP explainability** to answer the question:

> *"Which oven parts, when replaced during maintenance, have the greatest statistical impact on burn uniformity?"*

The analytical techniques — **SHAP TreeExplainer, SHAP Interaction Values, and XGBoost with cross-validated feature attribution** — are directly applicable to any **manufacturing quality control** scenario involving periodic maintenance and measurable outcomes.

---

## 🔗 Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: pizza-oven-burn-analysis                      │
│  https://github.com/shoheimustgoon/pizza-oven-burn-analysis │
│                                                         │
│  generate_pizza_data.py  →  pizza_burn_measurements.csv │
│                          →  oven_maintenance_log.csv    │
│                          →  maintenance_schedule.csv    │
│  burn_analysis.py        →  Cycle stats, Cohen's d,     │
│                             M2M variation, Direction    │
└──────────────────┬──────────────────────────────────────┘
                   │ CSV outputs
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: THIS REPOSITORY  ← you are here              │
│                                                         │
│  pizza_burn_ml_shap.py                                  │
│    Step 1: Build cycle-level MAD / Mean / SD            │
│    Step 2: Binary part feature matrix (XGBoost input)   │
│    Step 3: XGBoost → SHAP TreeExplainer                 │
│            → SHAP Interaction Values                    │
│    Step 4: Plot11 SHAP Summary (Bar + Beeswarm)         │
│            Plot12 SHAP Interaction Heatmap              │
│            Plot13 SHAP Dependence Plots                 │
│    Step 5: Excel output (8 sheets)                      │
└─────────────────────────────────────────────────────────┘
```

---

## 🍕 Analogy: Semiconductor ↔ Pizza Oven

| Semiconductor (Actual) | Pizza Oven (Analogy) | SHAP target |
|---|---|---|
| Tilt-R cycle MAD | Burn Deviation cycle MAD | ✅ ML target |
| ESC (Electrostatic Chuck) | Hearth Stone | `hearth_stone` |
| C-Shroud | Dome Lining | `dome_lining` |
| Inner Electrode | Upper Heater Calibration | `upper_heater_cal` |
| Outer Electrode | Lower Heater Calibration | `lower_heater_cal` |
| Edge Ring | Stone Ring Clean | `stone_ring_clean` |
| Regular Cleaning | Ash Removal | `ash_removal` |
| VAT / Seal | Door Seal | `door_seal` |
| TC / Sensor Check | Thermocouple Check | `thermocouple_check` |

---

## 🔬 Methodology

### Why XGBoost + SHAP?

Stage 1 uses descriptive statistics (Mean, MAD, SD, Cohen's d) and linear regression to quantify part impact. Stage 2 adds a **non-linear machine learning layer** to:

1. **Capture non-linear part effects** that linear regression misses
2. **Attribute causality at the individual cycle level** via SHAP values
3. **Detect part interaction effects** (e.g. "hearth_stone × dome_lining synergy")

### Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `Part_{name}` | Binary 0/1 | Was this part replaced at cycle start? |
| `N_in_Cycle` | Integer | Number of measurement points (control) |

### Analysis Table

| Step | Method | Literature |
|---|---|---|
| Cycle MAD | mean\|BurnDev_i − mean\| | Wikipedia: Mean absolute deviation |
| XGBoost | Gradient boosting regressor | Chen & Guestrin (2016) KDD |
| SHAP TreeExplainer | Exact Shapley values for tree ensembles | Lundberg et al. (2020) Nature MI |
| SHAP Interactions | Pairwise synergy between parts | Lundberg et al. (2018) arXiv:1802.03888 |
| Cross-Validation | LOO-CV (n<30) / 5-Fold (n≥30) | Varoquaux et al. (2017) NeuroImage |
| Overfitting controls | max_depth≤4, L1/L2 reg, subsample | XGBoost official docs |

---

## 📊 Outputs

### Plots

| File | Description |
|---|---|
| `Plot11_SHAP_Summary_Ver1.png` | Bar: mean \|SHAP\| per part + Beeswarm: per-cycle SHAP distribution |
| `Plot12_SHAP_Interaction_Heatmap_Ver1.png` | Part × Part interaction matrix |
| `Plot13_SHAP_Dependence_Ver1.png` | Top interaction pairs: dependence scatter |

### Excel Sheets

| Sheet | Contents |
|---|---|
| `Stage1_Burn_Sample` | Sampled Stage 1 measurements (≤5,000 rows) |
| `Stage1_Maintenance_Log` | All maintenance events |
| `Cycle_Summary` | Per-cycle MAD / Mean / SD + parts |
| `ML_Feature_Importance` | SHAP-ranked part importance |
| `SHAP_Values` | Per-cycle SHAP value matrix |
| `SHAP_Interactions` | Pairwise part interaction scores |
| `ML_Model_Info` | XGBoost params, CV results, analogy mapping |
| `Analysis_Info` | Full reference list |

---

## 💻 Usage

### Step 1: Run Stage 1 first (or skip for demo)

```bash
# Clone and run Stage 1 to generate CSV files
git clone https://github.com/shoheimustgoon/pizza-oven-burn-analysis
cd pizza-oven-burn-analysis
python generate_pizza_data.py
# → Creates: data/pizza_burn_measurements.csv
#            data/oven_maintenance_log.csv
#            data/maintenance_schedule.csv
```

### Step 2: Copy Stage 1 CSVs here

```bash
cd pizza-oven-ml-shap  # this repository
mkdir -p data
cp ../pizza-oven-burn-analysis/data/*.csv data/
```

### Step 3: Run Stage 2

```bash
python pizza_burn_ml_shap.py
```

> **Note:** If Stage 1 CSVs are not found in `data/`, the script auto-generates demo data and runs standalone. No manual setup needed to try it out.

### Output location

```
output/
  Plot11_SHAP_Summary_Ver1.png
  Plot12_SHAP_Interaction_Heatmap_Ver1.png
  Plot13_SHAP_Dependence_Ver1.png
  pizza_burn_ml_shap_Ver1_YYYYMMDD_HHMMSS.xlsx
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
openpyxl
xgboost
shap
scikit-learn
```

Install:

```bash
pip install numpy pandas matplotlib openpyxl xgboost shap scikit-learn
```

---

## 📚 References

1. Lundberg & Lee (2017) NeurIPS — "A Unified Approach to Interpreting Model Predictions"
2. Lundberg et al. (2020) Nature Machine Intelligence — "From local explanations to global understanding with explainable AI for trees"
3. Lundberg et al. (2018) arXiv:1802.03888 — "Consistent Individualized Feature Attribution for Tree Ensembles"
4. Chen & Guestrin (2016) KDD — "XGBoost: A Scalable Tree Boosting System"
5. Strobl et al. (2007) BMC Bioinformatics — "Bias in random forest variable importance measures"
6. Zhao et al. (2025) arXiv:2512.01205 — "Milling Machine Predictive Maintenance Based on ML and SHAP"
7. Varoquaux et al. (2017) NeuroImage — "Assessing and tuning brain decoders, via the Neuroimage Decoding Benchmark"
8. Cochrane Handbook Ch.6 — "Choosing effect measures"

---

## 👨‍💻 Author

**Go Sato**
Data Scientist | Causal Inference, Reliability Engineering, Machine Learning Explainability

---

---

# 🍕 ピザ窯の焦げムラ分析 — ML パーツ影響度解析（XGBoost + SHAP）

> **[pizza-oven-burn-analysis](https://github.com/shoheimustgoon/pizza-oven-burn-analysis) の応用編（Stage 2）**

## 📖 概要

本プロジェクトはピザ窯焦げムラ分析シリーズの**Stage 2**です。

Stage 1 が出力したCSVデータを受け取り、**機械学習（XGBoost）と SHAP 説明可能AI** を適用します。

> *「どのパーツを交換したメンテナンスが、焼きムラの改善に最も寄与するか？」*

この手法 — **SHAP TreeExplainer・SHAP Interaction Values・交差検証付き XGBoost** — は、定期メンテナンスと計測アウトカムを持つ**製造品質管理**全般に応用可能です。

---

## 🔗 2段階パイプライン

```
Stage 1: pizza-oven-burn-analysis
  generate_pizza_data.py  →  CSV 3ファイル出力
  burn_analysis.py        →  統計解析（Cohen's d, M2M変動, 方向性分析）
         ↓ CSV
Stage 2: このリポジトリ  ← ここ
  pizza_burn_ml_shap.py   →  XGBoost + SHAP → Excel + 3プロット
```

---

## 🔬 なぜ XGBoost + SHAP を追加するのか

Stage 1 の記述統計・線形回帰では捉えにくい以下の点を Stage 2 が補完します：

1. **非線形なパーツ効果** の検出
2. **個別サイクル単位**での貢献度の属性分解（SHAP値）
3. **パーツ交互作用**の検出（例：石床交換 × ドーム内壁交換の相乗効果）

---

## 💻 使い方

```bash
# Stage 1 CSV を data/ に配置してから実行
python pizza_burn_ml_shap.py

# data/ に CSV がない場合はデモデータを自動生成してそのまま動作確認可能
```

---

## 👨‍💻 Author

**佐藤 剛 (Go Sato)**
データサイエンティスト | 因果推論・信頼性工学・機械学習説明可能性
