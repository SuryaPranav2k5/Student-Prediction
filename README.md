# Student At-Risk Prediction System (OULAD)

End-to-end ML engineering project predicting whether an Open University Learning Analytics Dataset (OULAD) student is **at-risk** (`final_result ∈ {Withdrawn, Fail}`) at three checkpoints per course presentation — week 4, week 8, and week 12 — using only data available on or before each checkpoint.

## Data

- **Source:** OULAD (7 relational tables: studentInfo, studentRegistration, studentAssessment, assessments, vle, studentVle, courses), downloaded from the UCI mirror into `data/raw/`.
- **Train split:** presentations `2013J` + `2014J` (20,105 students).
- **Test split:** presentation `2014B` (7,804 students) — held out, untouched until final evaluation.
- **Label:** `at_risk = 1` if `final_result ∈ {Withdrawn, Fail}`, else `0`.

| Split | Students | at_risk = 1 | at_risk rate |
|-------|----------|-------------|--------------|
| Train (2013J+2014J) | 20,105 | 10,173 | 50.60% |
| Test (2014B) | 7,804 | 4,446 | 56.97% |

> **Data-vs-spec deviation flagged:** the spec expected `at_risk` to be the minority class. In OULAD it is **near-balanced / majority** (~51–57%). This is reported honestly; it materially affects the operating-point analysis (see below).

## Problem framing

This is a **temporal / checkpoint-based prediction problem**, not an end-of-course classifier. At each checkpoint the model must predict risk using only information observed up to that point, so an intervention can still act. The label (`final_result`) is only known at the end of the course and is never used as a feature — it is joined in strictly for training/evaluation after features are frozen at the checkpoint cutoff.

## Checkpoint dates (Step 1)

OULAD dates are **day-offsets from the start of the module presentation** (day 0 = first day); there are no absolute calendar dates. After inspecting the actual assessment due-date distribution per course, fixed calendar cutoffs were adopted (agreed decision: fixed cutoffs + an explicit missingness indicator — **not** dropping affected rows):

- week 4 → **day 28**, week 8 → **day 56**, week 12 → **day 84**

A cutoff counts as valid for a course only if **at least one assessment has closed by that day**. Where zero assessments have closed, the students are **kept** and flagged with an explicit `no_assessment_closed` indicator (missingness is informative):

| Checkpoint | Students with no assessment closed | Courses affected |
|------------|------------------------------------|------------------|
| Week 4 | 5,468 | EEE (first due day 33), GGG (first due day 61) |
| Week 8 | 2,534 | GGG (first due day 61) |
| Week 12 | 0 | none |

Per the spec's "flag and stop" instruction, this boundary condition was surfaced before implementation rather than silently worked around. The full cohort is retained at every checkpoint (83,727 feature rows = 27,909 students × 3 checkpoints); week-4/week-8 PR-AUC is therefore computed over students who may have no assessment signal yet, which is an explicit scope caveat.

## Leakage prevention (Steps 3 + 4)

This is the core design constraint. Every feature is derived exclusively from records with a day-offset ≤ the row's checkpoint cutoff:

- VLE clicks: `studentVle.date <= cutoff`
- Assessments: only `due_date <= cutoff` assessments enter the "due by cutoff" universe; only `date_submitted <= cutoff` submissions count as taken/missed.
- Scores / earliness / first-two-assessment delta computed strictly from on/before-cutoff submissions.
- Registration date is static pre-course information.

`features/test_no_leakage.py` independently re-derives every date-sensitive feature from the raw tables (strictly filtered to ≤ cutoff) and asserts element-wise equality with the stored feature matrix for the **full** feature set (all 83,727 rows — not a sample). It **raises** (not warns) on any violation and runs as part of the feature-build pipeline (`python features/build_features.py`). Current result: **passed**.

## Feature engineering (Step 3)

Per student, per checkpoint, using only pre-cutoff data:

- Cumulative VLE clicks by `activity_type`: all-time-to-date AND last-7-days-only.
- Engagement lag: days between material release and the student's first click (where release week is known).
- Submission timeliness: days early/late per assessment due by the checkpoint; count & fraction of missed submissions.
- Early assessment scores + score delta between the first two assessments taken.
- `num_of_prev_attempts`, `studied_credits`, `date_registration` relative to course start.

**Excluded from model features entirely** (routed to `fairness_eval.parquet`, used only in Step 6): `gender`, `disability`, `age_band`, `imd_band`.

Missing demographic fields are encoded as an explicit `"unknown"` category — **never imputed** with mean/mode (e.g., `imd_band` contains `?` values).

Output: `data/processed/features.parquet` (83,727 rows). It holds 84 columns total: **74 model features + 10 metadata columns** (`id_student`, `module_presentation`, `code_module`, `code_presentation`, `checkpoint_label`, `cutoff_day`, `week_elapsed`, `at_risk`, `final_result`, `split`). And `data/processed/fairness_eval.parquet`.

> **Verified correction:** an earlier build carried `gender`, `imd_band`, `age_band` and `disability` in the model feature list (74 → 78 columns), which broke the exclusion rule above. They were silently coerced to constant-zero by `pd.to_numeric(..., errors="coerce")`, so the reported metrics were **not** contaminated — LightGBM PR-AUC is bit-identical before and after the fix (0.8903 / 0.9217 / 0.9416). This is exactly the kind of failure that only shows up as an exception in an unrelated script, so the columns are now dropped physically in `features/build_features.py` (with an assert), re-asserted in `models/train.py`, and `serving/make_sample_input.py` will fail loudly if one ever reappears. See "What didn't work".

## Metric choice

**PR-AUC (area under the precision-recall curve)**, not accuracy. `at_risk` is near-balanced (~51–57%), not strongly imbalanced in either direction, so accuracy is not "misleading" in the classic minority-class sense. The real justification is **cost-sensitivity and class-specific performance**: the system's job is to surface students who need intervention, so we care specifically about how well the model ranks / identifies the at-risk (positive) class. PR-AUC isolates that performance and is robust to the operating threshold; accuracy would collapse both classes into a single number and hide whether at-risk students are actually being caught. PR-AUC is reported with a **bootstrap 95% confidence interval** (~1,000 resamples of the test set).

## Modeling (Step 5)

Mandatory baseline and main model, both on the identical feature set, **no SMOTE/synthetic oversampling**:

- **Baseline:** `LogisticRegression(class_weight='balanced')`
- **Main:** `LGBMClassifier(objective='binary', is_unbalance=True)`

Two LightGBM strategies compared on the held-out test set:

| Model | week 4 PR-AUC | week 8 PR-AUC | week 12 PR-AUC |
|-------|---------------|---------------|----------------|
| Pooled LightGBM (+ `week_elapsed`) | 0.8903 | 0.9217 | 0.9416 |
| Separate LightGBM (one per checkpoint) | 0.8901 | 0.9235 | 0.9422 |
| LogisticRegression baseline | 0.8727 | 0.9052 | 0.9251 |

Pooled and per-checkpoint LightGBM are **statistically indistinguishable** (differences ≤ 0.002 PR-AUC, within bootstrap CI overlap). The **pooled model is kept as the production artifact** (single artifact, `week_elapsed` lets one model serve all checkpoints). Full numbers: `artifacts/comparison.json`, `artifacts/metrics.json`.

No LSTM/deep sequence model was built (explicitly out of scope for this version).

**PR-AUC is fairly flat across checkpoints** (0.8903 → 0.9217 → 0.9416, only ~0.05 gain from week 4 to week 12 despite tripling the observation window). This is explained by the SHAP drivers: the most predictive features — missed assessments (`n_missed`, `frac_missed`), first/second assessment scores, and early VLE engagement (`homepage_clicks`) — are all available already at week 4, so the early signal is strong and later weeks add relatively little new discrimination.

## Evaluation (Step 6)

Headline PR-AUC with bootstrap 95% CI on the held-out 2014B test set:

| Checkpoint | PR-AUC | 95% CI | n |
|------------|--------|--------|---|
| Week 4 | 0.8903 | 0.8833 – 0.8975 | 7,804 |
| Week 8 | 0.9217 | 0.9161 – 0.9277 | 7,804 |
| Week 12 | 0.9416 | 0.9371 – 0.9463 | 7,804 |

Overall PR-AUC (pooling all checkpoint-level predictions into a single precision-recall curve): **0.92**.

**Operating point.** Because `at_risk` is the majority class (~57%), a precision=0.5 target is cleared trivially by flagging everyone (recall = 100%, precision ≈ prevalence). That is not a useful actionable threshold for triggering interventions, so we do **not** use it. The primary operating point is **precision = 0.75** (above prevalence, so it requires real model signal), with recall reported at that precision:

| Checkpoint | Precision target | Recall at threshold | Threshold |
|------------|------------------|---------------------|-----------|
| Week 4 | 0.75 | 0.8266 | 0.3819 |
| Week 8 | 0.75 | 0.8799 | 0.2932 |
| Week 12 | 0.75 | 0.9163 | 0.2495 |

Plain language at week 8: "the model catches about 88% of at-risk students while 3 in 4 flags are correct." The F1-optimal threshold is also reported as a secondary reference in `metrics.json`.

This threshold choice is a deviation from the spec's precision=0.5 suggestion, made because precision=0.5 is degenerate at this prevalence.

> **Scope caveat (full coverage):** all 7,804 held-out test students are scored at every checkpoint, including EEE/GGG students who have no assessment closed yet at week 4 (and GGG at week 8). The week-4/week-8 PR-AUC and operating point therefore include students with no assessment signal, which makes those checkpoints harder and shaves PR-AUC relative to a fully-assessed cohort.

- **Calibration curve:** one plot, `artifacts/plots/calibration_curve.png` (`sklearn.calibration.calibration_curve`).
- **Subgroup diagnostic:** recall within the predicted at-risk class, split by true label — Withdrawn recall (0.88–0.95) consistently exceeds Fail recall (0.76–0.88). The model does not train on Withdrawn vs Fail separately, but at the prediction threshold it is better at catching Withdrawn students.

### Fairness slice (reporting only — no mitigation implemented)

Actual breakdown at the week-8 operating point (threshold 0.2932), recall / FPR / flag-rate by group:

| Attribute | Group | n | Recall | FPR | Flag-rate |
|-----------|-------|---|--------|-----|-----------|
| gender | F | 3368 | 0.871 | 0.423 | 0.668 |
| gender | M | 4436 | 0.886 | 0.358 | 0.669 |
| imd_band | 0–10% | 818 | 0.912 | 0.523 | 0.778 |
| imd_band | 10–20 | 841 | 0.912 | 0.478 | 0.761 |
| imd_band | 20–30% | 909 | 0.884 | 0.457 | 0.733 |
| imd_band | 30–40% | 855 | 0.885 | 0.409 | 0.681 |
| imd_band | 40–50% | 747 | 0.865 | 0.329 | 0.636 |
| imd_band | 50–60% | 708 | 0.908 | 0.402 | 0.689 |
| imd_band | 60–70% | 686 | 0.860 | 0.338 | 0.621 |
| imd_band | 70–80% | 662 | 0.829 | 0.383 | 0.627 |
| imd_band | 80–90% | 697 | 0.840 | 0.329 | 0.581 |
| imd_band | 90–100% | 609 | 0.866 | 0.304 | 0.560 |
| imd_band | ? (unknown) | 272 | 0.864 | 0.308 | 0.518 |

(week-4 and week-12 show the same qualitative pattern; full per-checkpoint tables are produced by `eval/fairness_breakdown.py` and stored in `artifacts/metrics.json` under `fairness`.)

**Reading the ~23-point FPR spread (0.304 → 0.523) — it is not a single-group outlier:**

- **Gender is a non-issue.** FPR is near-parity (F 0.423 vs M 0.358 at week 8 — women are actually flagged *more*, not less), and true at-risk prevalence is near-identical (F 0.545 vs M 0.589). No disparate-impact signal on gender.
- **The spread is an IMD deprivation gradient, and it largely tracks a real base-rate difference.** FPR falls monotonically as deprivation decreases: the most-deprived band (0–10%) has both the highest false-positive rate (~0.49–0.52) and the highest *true* at-risk prevalence (0.654, vs 0.499 for the least-deprived bands 70–100% — a ~15-point real gap). Recall is also *higher* in deprived bands (0.87–0.93) than affluent ones (0.82–0.90), so on the metric that matters for intervention the model serves deprived students at least as well.

**Recall is the other half of the fairness picture, and it points the same way — so the correct reading is a named trade-off, not two disconnected facts.** Week-8 recall / FPR / mean-predicted-probability-among-non-at-risk students:

| Group | Recall | FPR | Flag-rate | Mean P(at-risk) \| not at-risk |
|-------|--------|-----|-----------|----------------|
| IMD 0–10% (most deprived) | 0.912 | 0.523 | 0.778 | 0.3457 |
| IMD 10–20 | 0.912 | 0.478 | 0.761 | 0.3302 |
| IMD 20–30 | 0.884 | 0.457 | 0.733 | 0.3130 |
| IMD 30–40 | 0.885 | 0.409 | 0.681 | 0.3018 |
| IMD 40–50 | 0.865 | 0.329 | 0.636 | 0.2645 |
| IMD 50–60 | 0.908 | 0.402 | 0.689 | 0.2931 |
| IMD 60–70 | 0.860 | 0.338 | 0.621 | 0.2716 |
| IMD 70–80 | 0.829 | 0.383 | 0.627 | 0.2980 |
| IMD 80–90 | 0.840 | 0.329 | 0.581 | 0.2538 |
| IMD 90–100% (least deprived) | 0.866 | 0.304 | 0.560 | 0.2534 |
| Gender F | 0.871 | 0.423 | 0.668 | 0.3033 |
| Gender M | 0.886 | 0.358 | 0.669 | 0.2784 |

Deprived-vs-affluent aggregate at week 8: recall 0.902 vs 0.843, FPR 0.484 vs 0.337, mean P given not-at-risk 0.329 vs 0.267.

**Both recall and FPR are elevated for deprived students, and the mean-score diagnostic shows why: deprived students who are *not* at-risk still score higher on average (0.329 vs 0.267). That is a lower effective risk threshold for that group — the model flags them more readily in both directions.** Concretely this is a pro-deprived trade on the metric that matters for intervention (it catches 6 more of every 100 true at-risk students in the deprived bands) bought with roughly 15 extra false positives per 100 affluent-band-equivalent negatives. (Those are rates on different denominators — recall is per 100 *actual at-risk* students, FPR is per 100 *actual not-at-risk* students — so they are not directly subtractable into a single net figure; converting to absolute counts would require the group sizes and prevalence.) This is a real, directional equity/efficiency choice the learned model made on its own, not neutral behavior - and it is the specific thing that should be examined first if this system were ever deployed to allocate a scarce resource like advisor time.

**The residual, honest caveat:** even among students who are *not* at-risk, deprived students are flagged as false positives roughly 1.6–1.7× more often than affluent students. That gap is only partly explained by base rate — it is a genuine disparate-impact signal on socioeconomic status, smooth rather than concentrated in any single category, and it would be the first thing to revisit with reweighting or group-aware thresholds. **Mitigation was out of scope for this iteration and is not implemented** — but given the magnitude, it should not remain out of scope in a follow-up.

True at-risk prevalence by group (the base rate the FPR gradient tracks):

| Group | n | at-risk prevalence |
|-------|---|--------------------|
| IMD 0–10% (most deprived) | 2,454 | 0.654 |
| IMD 10–20 | 2,523 | 0.652 |
| IMD 20–30 | 2,727 | 0.646 |
| IMD 30–40 | 2,565 | 0.571 |
| IMD 40–50 | 2,241 | 0.573 |
| IMD 50–60 | 2,124 | 0.568 |
| IMD 60–70 | 2,058 | 0.542 |
| IMD 70–80 | 1,986 | 0.547 |
| IMD 80–90 | 2,091 | 0.494 |
| IMD 90–100% (least deprived) | 1,827 | 0.455 |
| Gender F | 10,104 | 0.545 |
| Gender M | 13,308 | 0.589 |

Deprived-band (IMD 0–20%) at-risk prevalence 0.650 vs affluent-band (IMD 70–100%) 0.499 — a **15.1-point real base-rate gap**; gender shows a 4.4-point gap.

## Explainability (Step 7)

SHAP `TreeExplainer` on the winning pooled LightGBM model:

- Global feature-importance bar chart: `artifacts/plots/shap_global_importance.png`
- Top global drivers: `n_missed`, `frac_missed`, `first_assessment_score`, `homepage_clicks_last7`, `mean_earliness_days` — all available at week 4 (consistent with the flat PR-AUC curve above).
- `no_assessment_closed` sits at **SHAP rank 62 of 74 with mean|SHAP| = 0.00007** - effectively zero importance (12 features rank below it). This is a verified, mechanistic result, not a top-5 coincidence. It is **exactly, symmetrically collinear** with `n_assessments_due`: `no_assessment_closed == 1` iff `n_assessments_due == 0` holds for 100% of rows (it is defined as that comparison, verified across all 83,727 rows). Since `n_assessments_due` also encodes the *magnitude* (1, 2, 3. assessments due), any tree split on `n_assessments_due` recovers the flag perfectly, so the flag contributes strictly no extra information. `n_assessments_due` itself ranks 23 (mean|SHAP| = 0.023) - the tree splits there and the indicator is simply never needed. This is a verified, mechanistic result, not a top-5 coincidence. It is **exactly, symmetrically collinear** with `n_assessments_due`: `no_assessment_closed == 1` iff `n_assessments_due == 0` holds for 100% of rows (it is defined as that comparison, verified across all 83,727 rows). Since `n_assessments_due` also encodes the *magnitude* (1, 2, 3… assessments due), any tree split on `n_assessments_due` recovers the flag perfectly, so the flag contributes strictly no extra information. `n_assessments_due` itself ranks 23 (mean|SHAP| = 0.023) — the tree splits there and the indicator is simply never needed.
  - Worth noting the tempting-but-wrong explanation, since it is a real trap: the flag does *not* become redundant via `n_missed`/`frac_missed`/`first_assessment_score`. That triple signature is genuinely ambiguous — 167 rows (0.2%) have `n_assessments_due > 0`, missed nothing, and recorded a literal first-assessment score of 0, and those rows are 84% at-risk versus 41% for the true no-assessment-yet rows. The redundancy is really through `n_assessments_due`, and that is the clean, correct account.
  - The flag is retained as an explicit interpretability handle (it makes "no assessment signal yet" self-documenting), not because it carries predictive signal — it carries none by construction. Verified by `eval/collinearity_check.py`.
- 4 worked individual examples with per-student SHAP breakdowns: `artifacts/plots/shap_student_*.png`

No LLM-based natural-language explanation layer was built (out of scope until steps 1–6 are verified).

## Testing the model with input

Start the API, then send a feature dict. **You do not need all 74 features** — the endpoint fills anything you omit with 0, so a small hand-edited payload is enough to move the prediction.

```bash
uvicorn serving.api:app --reload     # terminal 1
python serving/make_sample_input.py  # terminal 2 — exports a real student's features to JSON
python serving/test_client.py        # hits /health and /predict
```

`artifacts/sample_input.json` is one real held-out student's full 74-feature vector; `artifacts/sample_input_min.json` is a ~15-feature subset you can edit by hand. `python serving/test_client.py --full` uses the full payload.

Equivalent raw calls:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"n_missed": 0, "frac_missed": 0.0, "first_assessment_score": 92.0,
                    "homepage_clicks_last7": 40.0, "studied_credits": 60.0,
                    "n_assessments_taken": 2, "n_assessments_due": 2,
                    "no_assessment_closed": 0, "week_elapsed": 12},
       "checkpoint_label": "week_12"}'
```

**Or use the dashboard form** — the "Try it yourself" section at the bottom of `serving/dashboard.py` exposes the 12 highest-leverage fields as number inputs with three presets, so you can drag values and watch P(at_risk) plus the SHAP waterfall update. Verified:

| Input | P(at_risk) |
|-------|-----------|
| Low-risk preset (missed nothing, strong scores, active) | **0.079** |
| Low-risk preset, then `n_missed` = 5 typed in | **0.344** |
| ...then `first_assessment_score` = 100 typed in | **0.310** |
| High-risk preset (missed all 5, no scores, 240 credits) | **0.997** |

Verified responses (both run against the pinned production model):

| Payload | P(at_risk) | Top SHAP driver |
|---------|-----------|-----------------|
| Real held-out student, missing all 5 assessments (true label: Withdrawn) | **0.9960** | `n_missed = +1.93` |
| Synthetic engaged student, missed nothing, strong scores | **0.0735** | `first_assessment_score = −0.52` |

Note that `gender`, `imd_band`, `age_band` and `disability` are **not accepted as model inputs** — they live only in `fairness_eval.parquet`. Sending them is harmless (they are ignored), but they cannot influence the prediction.

## Serving (Step 8)

- **FastAPI:** `serving/api.py` — `POST /predict` accepts student features and returns `at_risk` probability + SHAP explanation; `GET /health`.
- **Streamlit:** `serving/dashboard.py` - four sections: (1) the per-checkpoint PR-AUC comparison chart from Step 5 as the headline visual, (2) cohort-level risk distribution, (3) per-student drill-down with SHAP explanation, and (4) a **live what-if form** where you type feature values and watch P(at_risk) plus the SHAP breakdown update. Run with `streamlit run serving/dashboard.py`.
- **Dockerfile:** `Dockerfile` for the FastAPI service.
- No MLflow. Model artifacts are written twice per run: **versioned** (`artifacts/models/lgb_pooled_<timestamp>.joblib`, the history) and **pinned** (`artifacts/models/lgb_pooled_production.joblib`, a fixed filename that serving, evaluation, SHAP and the dashboard all load). That removes the ambiguity of "whatever timestamp sorts last." A `metrics.json` is written per training run (`artifacts/metrics.json`) with model name, checkpoint, PR-AUC + CI, date, plus the subgroup and fairness tables.

## What didn't work / honest dead ends

- **The frontend had never been run until it was tested.** `serving/dashboard.py` was compile-checked and then left alone for most of the project's life. Headlessly rendering it surfaced two real bugs that would each have bitten a user, and one of which my own test initially hid:
  1. **Presets didn't change the prediction.** All three preset buttons returned P(at_risk) = 0.997. Cause: the number inputs were re-seeded from the preset's defaults, but Streamlit keeps widget state keyed by `key`, so on the rerun triggered by the radio the fields silently retained their old values. Fixed by scoping the widget keys to the preset (`in_{preset}_{feature}`), which is also the correct behaviour in a real browser, not just a test artifact.
  2. **A dict/Series type error** in the same block (`base.index` on a `dict`) that only fired for the two example presets, never the default one. Fixed with an explicit `isinstance` branch.
  Both were invisible to a syntax check and would have looked like "the dashboard works" until someone clicked a button. **Lesson: "it compiles" is not "it runs" — the two bugs above lived behind a working default render.**
- **Protected attributes leaked into the model feature list.** `gender`, `imd_band`, `age_band` and `disability` were being carried as model features, violating the Step 3 exclusion rule and making the README's claim that they were excluded false. They were silently coerced to constant-zero by `pd.to_numeric(..., errors="coerce")`, so the effect was benign in the strict sense — LightGBM PR-AUC is **bit-identical** before and after the fix (0.8903 / 0.9217 / 0.9416) — but it was a real defect that no test caught. It surfaced only because `serving/make_sample_input.py` crashed on `float('F')`. Now prevented three ways: the columns are dropped physically in `features/build_features.py` (with an assert), re-asserted in `models/train.py::feature_columns`, and the API ignores the four attributes entirely. **Lesson: a spec-level constraint ("exclude these from model input") was enforced by convention, not by code, and convention failed silently.**
- **The mandated baseline was under-converged and under-reported.** `LogisticRegression` hit the `max_iter=1000` limit (unscaled lbfgs on features spanning click counts in the thousands vs scores 0-100), which understated it. With `StandardScaler` + `max_iter=5000` it goes from 0.8440/0.8929/0.9142 to **0.8727/0.9052/0.9251** — and at week 12 it comes within 1.6 points of LightGBM (0.9251 vs 0.9416). A knowingly hobbled baseline would have inflated the main model's apparent advantage, so the scaler is not optional. **Notable takeaway: on this feature set a well-converged linear model is a much stronger baseline than the headline numbers first suggested.**
- **A hardcoded baseline figure in the dashboard went stale.** The caption quoting the LR PR-AUC was written once and never updated, so it drifted from the real value as the baseline was re-trained. It now reads from `comparison.json`. This is the same failure mode as the metric-choice text that used to justify PR-AUC with "highly imbalanced" after the imbalance direction was corrected — **numbers baked into prose are numbers nobody re-checks.**
- **The mandated baseline was under-converged and under-reported.** `LogisticRegression` hit the `max_iter=1000` limit (unscaled lbfgs on features spanning click counts in the thousands vs scores 0–100), which understated it. With `StandardScaler` + `max_iter=5000` it goes from 0.8440/0.8929/0.9142 to **0.8727/0.9052/0.9251** — and at week 12 it comes within 1.6 points of LightGBM (0.9251 vs 0.9416). A knowingly hobbled baseline would have inflated the main model's apparent advantage, so the scaler is not optional. **Notable takeaway: on this feature set a well-converged linear model is a much stronger baseline than the headline numbers first suggested.**
- **Label imbalance assumption wrong:** `at_risk` is near-balanced / majority in OULAD, not the minority as the spec anticipated. Class-rebalancing tools still run but target the actual minority (`not_at_risk`).
- **Precision = 0.5 operating point is degenerate:** since at_risk is prevalent, flagging everyone clears precision=0.5 for free (recall 100%, precision ~57%). It requires zero model discrimination and cannot trigger a useful intervention. Replaced with precision=0.75 as the primary actionable operating point.
- **GGG/EEE boundary condition:** EEE and GGG have no assessment closed by week 4 (GGG none by week 8). Per the agreed decision these students are **retained** with an explicit `no_assessment_closed` indicator rather than dropped, so coverage is full but week-4/week-8 prediction is harder for those students (no assessment signal). This is why week-4 PR-AUC (0.8903) is lower than it would be on a fully-assessed cohort.
- **`no_assessment_closed` earns its place only as an interpretability handle, not as signal - and it is *exactly* redundant, not approximately.** It is definitionally `n_assessments_due == 0` (100% agreement over all 83,727 rows, verified by `eval/collinearity_check.py`), so any tree split on `n_assessments_due` already recovers it perfectly. SHAP confirms it: rank 62/74, mean|SHAP| = 0.00007, while `n_assessments_due` itself ranks 23. Kept for explicit, self-documenting missingness - it contributes no predictive information by construction.
  - The tempting shortcut — "the `n_missed`/`frac_missed`/`first_assessment_score` triple already encodes it" — is **wrong and worth stating as a trap**: that triple signature is genuinely ambiguous, covering 167 rows (0.2%) where assessments *were* due, nothing was missed, and a literal first score of 0 was recorded. Those rows run 84% at-risk against 41% for true no-assessment-yet rows, so the triple would conflate two very different populations. The redundancy runs through `n_assessments_due`, which is the clean account.
- **`no_assessment_closed` was *not* actually justified by importance.** It is worth being explicit about this: the feature survived because the brief asked for an explicit "no assessment yet" indicator, and it turned out to be zero-value. The correct lesson is that the indicator was the right thing to *try*, and the SHAP/collinearity audit is what proved it unnecessary.
- **Fairness is flagged, not resolved.** Recall is broadly comparable across groups, but there is a real ~23-point FPR spread across IMD bands driven by a smooth socioeconomic gradient (deprived students are flagged as false positives ~1.6–1.7× more often even after partial base-rate adjustment). Gender shows essentially no gap (≤0.08). Mitigation was out of scope by design, but the size of the IMD gap is large enough that treating it as a footnote is not adequate for a follow-up iteration.
- **The overall label framing is the weakest part of the brief.** Merging Withdrawn and Fail into one positive class hides a real predictive asymmetry: the model catches Withdrawn students markedly better than Fail students (recall 0.88–0.95 vs 0.76–0.88 at the operating point). If the intervention differs for the two failure modes, a two-stage or three-class formulation would likely serve better — that is a modeling decision, not a bug.

## Generalizability note

OULAD is a UK Open University (adult distance-learning) population; results may not generalize to other institution types or to residential/younger student cohorts.

## Reproducing

Requires Python ≥ 3.11. Dependencies are declared in `pyproject.toml` (also resolvable as `uv sync --extra dev` for the pytest suite).

```bash
# 0. Data: download OULAD from the UCI mirror into data/raw/ (7 CSV tables).
#    data/raw/ and data/processed/ are gitignored and regenerable.

# 1. Features + labels + leakage audit (full-set, raises on violation)
python features/build_features.py

# 2. Train models (baseline LR + pooled & per-checkpoint LightGBM)
python models/train.py

# 3. Evaluate (PR-AUC + bootstrap CI, calibration, operating point, subgroup, fairness, metrics.json)
python eval/run_eval.py

# 4. SHAP explainability
python eval/shap_explain.py

# 4b. Fairness deep-dive + no_assessment_closed SHAP standing
python eval/fairness_breakdown.py

# 4c. Exact-collinearity proof for no_assessment_closed
python eval/collinearity_check.py

# 5. Serve
uvicorn serving.api:app --reload            # FastAPI
streamlit run serving/dashboard.py          # dashboard
# or build the Docker image: docker build -t oulad-risk-api -f Dockerfile .

# 5b. Test the API by hand
python serving/make_sample_input.py         # export a real student's features to JSON
python serving/test_client.py               # hit /health and /predict
python serving/test_client.py --full        # use the full 74-feature payload

# Tests
python -m pytest features/test_no_leakage.py -q    # also runs automatically inside step 1

# Frontend verification (headless — no browser needed)
python serving/test_dashboard.py                # renders the app, asserts no exceptions
python serving/test_dashboard_interactive.py    # flips presets, asserts high-risk >> low-risk
python serving/test_dashboard_manual_input.py   # types into fields, asserts the score moves both ways
```

**Not verified:** the `Dockerfile` has not been built in this environment (no Docker available). The FastAPI service and the Streamlit dashboard have both been verified, headlessly:

```bash
python serving/test_dashboard.py               # renders the app, asserts no exceptions
python serving/test_dashboard_interactive.py   # flips the presets, asserts high-risk >> low-risk
python serving/test_dashboard_manual_input.py  # types into fields, asserts the score moves both ways
python -m pytest features/test_no_leakage.py -q
```

Treat the Docker image build as the one step to confirm on a machine with Docker.
