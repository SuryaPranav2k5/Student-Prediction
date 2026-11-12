# Student At-Risk Prediction System (OULAD)

End-to-end ML engineering project predicting whether an Open University Learning Analytics Dataset (OULAD) student is **at-risk** (`final_result ∈ {Withdrawn, Fail}`) at three checkpoints per course presentation — week 4, week 8, and week 12 — using only data available on or before each checkpoint.

## Data

- **Source:** OULAD — Open University Learning Analytics Dataset, 7 relational tables (studentInfo, studentRegistration, studentAssessment, assessments, vle, studentVle, courses).
- **Download:** UCI Machine Learning Repository, dataset id 349 — <https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset> (direct: <https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip>). Unzip the CSV tables into `data/raw/`; that directory is gitignored and fully regenerable from this URL.
- **License:** **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>) — sharing and adaptation permitted with attribution.
- **Citation:** Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University Learning Analytics dataset* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5KK69>.
- **Data protection:** OULAD is a pseudonymised research release (student identifiers are arbitrary integers; no names, emails or contact details). No restricted personal data is redistributed with this repository — only the licence, the citation and derived, regenerable features. See "Privacy, representativeness and failure modes".
- **Train split:** presentations `2013B` + `2013J` (13,529 students).
- **Validation split:** presentation `2014B` (7,804 students), used for model/threshold decisions.
- **Test split:** presentation `2014J` (11,260 students) — held out, untouched until final evaluation.
- **Label:** `at_risk = 1` if `final_result ∈ {Withdrawn, Fail}`, else `0`.

| Split | Students | at_risk = 1 | at_risk rate |
|-------|----------|-------------|--------------|
| Train (2013B+2013J) | 13,529 | 6,959 | 51.44% |
| Validation (2014B) | 7,804 | 4,446 | 56.97% |
| Test (2014J) | 11,260 | 5,803 | 51.54% |

### Presentation coverage exceptions

The split is constrained by OULAD's actual course/presentation calendar; there is no
three-way assignment using these presentation codes that gives every course complete
train/validation/test coverage:

| Course | 2013B | 2013J | 2014B | 2014J |
|---|---:|---:|---:|---:|
| AAA | 0 | 383 | 0 | 365 |
| BBB | 1,767 | 2,237 | 1,613 | 2,292 |
| CCC | 0 | 0 | 1,936 | 2,498 |
| DDD | 1,303 | 1,938 | 1,228 | 1,803 |
| EEE | 0 | 1,052 | 694 | 1,188 |
| FFF | 1,614 | 2,283 | 1,500 | 2,365 |
| GGG | 0 | 952 | 833 | 749 |

CCC has no training rows, so its test results are reported as a zero-shot
cross-course generalization check. AAA has no validation rows, so model and
threshold decisions have no AAA-specific validation feedback. BBB, DDD and FFF
receive additional 2013B training data; the other trained courses use 2013J only.

> **Data-vs-spec deviation flagged:** the spec expected `at_risk` to be the minority class. In OULAD it is **near-balanced / majority** (~51–57%). This is reported honestly; it materially affects the operating-point analysis (see below).

## EDA and data quality (`eval/eda_report.py`)

Run `python eval/eda_report.py` to reproduce every figure below.

- **Volumes:** 32,593 student-presentation rows (28,785 distinct students — a student may enrol in more than one module/presentation), 7 modules, 4 presentations, 206 assessments, 10.66M `studentVle` click rows.
- **Key integrity:** exactly one `studentInfo` row per student-presentation (max = 1) and **zero** students with conflicting `final_result` — the label join cannot duplicate or contradict rows.
- **Class balance:** Pass 12,361 · Withdrawn 10,156 · Fail 7,052 · Distinction 3,024 → `at_risk` = **52.80%**. That is the opposite of the brief's "rare at-risk event" assumption, and it is why accuracy is not the headline metric (see "Metric choice").
- **Missing values:** only `imd_band` is missing (3.41%, stored as `?`) plus 11 `assessments.date` sentinels; `gender`, `age_band`, `disability` and `studentAssessment.score` are complete. Missing demographics become an explicit `"unknown"` category — never mean/mode imputed — and never enter the model.
- **Dates:** day-offsets from presentation start. `studentVle.date` spans −25 → 269 (6.47% of clicks are pre-course access, i.e. before day 0); assessment due dates span day 12 → 261. Negative days are kept: they are real pre-course behaviour and are legitimately "before the cutoff".
- **Outliers / heavy tail:** total VLE clicks per student have median 824, p90 3,857, p99 9,013, max 28,615 (**35× the median**). These are real behavioural extremes, not corruption, so they are **not clipped or winsorised** — LightGBM is scale-invariant and the LR baseline is wrapped in `StandardScaler`. No row is dropped for being extreme.
- **Built matrix:** 97,779 rows × 84 columns, **0 NaN cells** — by design, not by accident: the assessment-dependent fields (`first_assessment_score`, `second_assessment_score`, `score_delta`) are 0-filled, so LightGBM's native NaN branch is not used. Missingness is instead carried by explicit indicators (`no_assessment_closed`, `first_score_missing`, `engagement_lag_missing`, `reg_date_missing`) plus the counts `n_assessments_due` / `n_assessments_taken`, so "not yet assessed" stays recoverable. The cost is narrow and stated: 347 rows (0.35%) hold a genuine score of 0 that is indistinguishable from "no score yet". 8,002 rows (**8.2%**) have no assessment closed by their cutoff — retained and flagged, per the cutoffs decision above.

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

Per the spec's "flag and stop" instruction, this boundary condition was surfaced before implementation rather than silently worked around. The full cohort is retained at every checkpoint (97,779 feature rows = 32,593 student-presentation records × 3 checkpoints, covering 28,785 distinct students); week-4/week-8 PR-AUC is therefore computed over students who may have no assessment signal yet, which is an explicit scope caveat.

## Leakage prevention (Steps 3 + 4)

This is the core design constraint. Every feature is derived exclusively from records with a day-offset ≤ the row's checkpoint cutoff:

- VLE clicks: `studentVle.date <= cutoff`
- Assessments: only `due_date <= cutoff` assessments enter the "due by cutoff" universe; only `date_submitted <= cutoff` submissions count as taken/missed.
- Scores / earliness / first-two-assessment delta computed strictly from on/before-cutoff submissions.
- Registration date is static pre-course information.

`features/test_no_leakage.py` independently re-derives every date-sensitive feature from the raw tables (strictly filtered to ≤ cutoff) and asserts element-wise equality with the stored feature matrix for the **full** feature set (all 97,779 rows — not a sample). It **raises** (not warns) on any violation and runs as part of the feature-build pipeline (`python features/build_features.py`). Current result: **passed**. An independent inference test (`serving/test_inference.py`) covers the other half: the API must load the saved artifact and return real scores.

## Feature engineering (Step 3)

Per student, per checkpoint, using only pre-cutoff data:

- Cumulative VLE clicks by `activity_type`: all-time-to-date AND last-7-days-only.
- Engagement lag: days between material release and the student's first click (where release week is known).
- Submission timeliness: days early/late per assessment due by the checkpoint; count & fraction of missed submissions.
- Early assessment scores + score delta between the first two assessments taken.
- `num_of_prev_attempts`, `studied_credits`, `date_registration` relative to course start.

**Excluded from model features entirely** (routed to `fairness_eval.parquet`, used only in Step 6): `gender`, `disability`, `age_band`, `imd_band`.

Missing demographic fields are encoded as an explicit `"unknown"` category — **never imputed** with mean/mode (e.g., `imd_band` contains `?` values).

Output: `data/processed/features.parquet` (97,779 rows). It holds 84 columns total: **74 model features + 10 metadata columns** (`id_student`, `module_presentation`, `code_module`, `code_presentation`, `checkpoint_label`, `cutoff_day`, `week_elapsed`, `at_risk`, `final_result`, `split`). And `data/processed/fairness_eval.parquet`.

> **Verified correction:** an earlier build carried `gender`, `imd_band`, `age_band` and `disability` in the model feature list (74 → 78 columns), which broke the exclusion rule above. They were silently coerced to constant-zero by `pd.to_numeric(..., errors="coerce")`, so the reported metrics were **not** contaminated — LightGBM validation PR-AUC is bit-identical before and after the fix (0.8905 / 0.9177 / 0.9380). This is exactly the kind of failure that only shows up as an exception in an unrelated script, so the columns are now dropped physically in `features/build_features.py` (with an assert), re-asserted in `models/train.py`, and `serving/make_sample_input.py` will fail loudly if one ever reappears. See "What didn't work".

## Metric choice

**PR-AUC (area under the precision-recall curve)**, not accuracy. `at_risk` is near-balanced (~51–57%), not strongly imbalanced in either direction, so accuracy is not "misleading" in the classic minority-class sense. The real justification is **cost-sensitivity and class-specific performance**: the system's job is to surface students who need intervention, so we care specifically about how well the model ranks / identifies the at-risk (positive) class. PR-AUC isolates that performance and is robust to the operating threshold; accuracy would collapse both classes into a single number and hide whether at-risk students are actually being caught. PR-AUC is reported with a **bootstrap 95% confidence interval** (~1,000 resamples of the test set).

## Modeling (Step 5)

Three algorithm families on the identical feature set, **no SMOTE/synthetic oversampling**:

- **Baseline:** `LogisticRegression(class_weight='balanced')` (with `StandardScaler`, `max_iter=5000`)
- **Main:** `LGBMClassifier(objective='binary', is_unbalance=True)` — pooled and per-checkpoint
- **Second family:** `RandomForestClassifier(n_estimators=300, class_weight='balanced')`

Pooled and separate LightGBM strategies are compared using the validation split;
the final held-out test results are reported separately:

| Model | week 4 PR-AUC | week 8 PR-AUC | week 12 PR-AUC |
|-------|---------------|---------------|----------------|
| Pooled LightGBM — validation | 0.8905 | 0.9177 | 0.9380 |
| Pooled LightGBM — test | 0.8558 | 0.8998 | 0.9193 |
| Separate LightGBM — test | 0.8504 | 0.8996 | 0.9186 |
| RandomForest — validation | 0.8903 | 0.9170 | 0.9393 |
| RandomForest — test | 0.8555 | 0.8959 | 0.9191 |
| LogisticRegression — validation | 0.8716 | 0.9102 | 0.9322 |
| LogisticRegression — test | 0.8115 | 0.8768 | 0.9025 |

**RandomForest is competitive rather than dominated:** it edges LightGBM on week-12 validation PR-AUC (0.9393 vs 0.9380) but loses all three held-out checkpoints, by ≤ 0.004 (week 8: 0.8959 vs 0.8998). Pooled LightGBM keeps production on the strength of the held-out results — the gaps are small enough that this is a reported tie-break rather than a decisive win, and it is the reason the comparison is on a fixed test split rather than on the training fit.

The **pooled model is kept as the production artifact** (single artifact,
`week_elapsed` lets one model serve all checkpoints). Full numbers are in
`artifacts/comparison.json` and `artifacts/metrics.json`.

No LSTM/deep sequence model was built (explicitly out of scope for this version).

**PR-AUC increases across checkpoints** (0.8558 → 0.8998 → 0.9193 on test).
The SHAP drivers show why early signal is already useful: missed assessments,
assessment scores and early VLE engagement are available by week 4.

## Evaluation (Step 6)

Headline PR-AUC with bootstrap 95% CI on the held-out 2014J test set:

| Checkpoint | PR-AUC | 95% CI | n |
|------------|--------|--------|---|
| Week 4 | 0.8558 | 0.8479 – 0.8633 | 11,260 |
| Week 8 | 0.8998 | 0.8936 – 0.9058 | 11,260 |
| Week 12 | 0.9193 | 0.9135 – 0.9242 | 11,260 |

Overall PR-AUC (pooling all checkpoint-level predictions into a single precision-recall curve): **0.894**.

**Operating point.** Because `at_risk` is the majority class (~57%), a precision=0.5 target is cleared trivially by flagging everyone (recall = 100%, precision ≈ prevalence). That is not a useful actionable threshold for triggering interventions, so we do **not** use it. The primary operating point is **precision = 0.75** (above prevalence, so it requires real model signal). The threshold is selected on the **validation** split — the max-recall point on the validation PR curve with precision ≥ 0.75 — and recall at that point is reported:

| Checkpoint | Precision target (validation) | Recall at threshold (validation) | Threshold |
|------------|------------------|---------------------|-----------|
| Week 4 | 0.75 | 0.8286 | 0.3695 |
| Week 8 | 0.75 | 0.8763 | 0.2855 |
| Week 12 | 0.75 | 0.9123 | 0.2340 |

That threshold is then **frozen** and applied unchanged to the held-out test set, which is what the classification-metrics table below reports. Test precision/recall therefore differ slightly from the validation figures above — e.g. week 4 test precision is 0.7133 rather than 0.7500 — because the two splits have different precision-recall curves. This is selection on validation, reporting on a split never used for tuning, not a contradiction.

Plain language at week 8: "the model catches about 88% of at-risk students while 3 in 4 flags are correct." The F1-optimal threshold is also reported as a secondary reference in `metrics.json`.

The required classification metrics are also persisted in `artifacts/metrics.json` for each checkpoint at this operating point:

| Checkpoint | Precision | Recall | F1 |
|------------|-----------|--------|----|
| Week 4 | 0.7133 | 0.7827 | 0.7464 |
| Week 8 | 0.7197 | 0.8544 | 0.7813 |
| Week 12 | 0.7047 | 0.8978 | 0.7896 |

Each checkpoint entry includes `confusion_matrix` in `[ [TN, FP], [FN, TP] ]` order. Thresholds are selected on validation data and these measures are calculated on the held-out 2014J test set, not on training data.

The test-set per-course PR-AUC breakdown is stored under `course_pr_auc` in
`artifacts/metrics.json`; CCC is marked `zero_shot_no_training_rows` and AAA is
marked `no_validation_rows`.

This threshold choice is a deviation from the spec's precision=0.5 suggestion, made because precision=0.5 is degenerate at this prevalence.

> **Scope caveat (full coverage):** all 11,260 held-out test students are scored at
> every checkpoint, including EEE/GGG students who have no assessment closed yet
> at week 4 (and GGG at week 8). The week-4/week-8 results therefore include
> students with no assessment signal.

- **Calibration curve:** one plot, `artifacts/plots/calibration_curve.png` (`sklearn.calibration.calibration_curve`).
- **Subgroup diagnostic:** recall within the predicted at-risk class, split by true label — Withdrawn recall (0.836 / 0.900 / 0.933) consistently exceeds Fail recall (0.680 / 0.767 / 0.829) at weeks 4 / 8 / 12. The model does not train on Withdrawn vs Fail separately, but at the prediction threshold it is better at catching Withdrawn students. This is the asymmetry the error analysis below explains.

### Fairness slice (reporting only — no mitigation implemented)

Actual week-8 breakdown at the validation-selected operating point (threshold 0.2855), recall / FPR / flag-rate by group:

| Attribute | Group | n | Recall | FPR | Flag-rate | Mean P(at-risk) \| not at-risk |
|-----------|-------|---|--------|-----|-----------|----------------|
| gender | F | 4,761 | 0.858 | 0.434 | 0.646 | 0.2909 |
| gender | M | 6,499 | 0.852 | 0.292 | 0.587 | 0.2430 |
| imd_band | 0–10% | 1,104 | 0.888 | 0.450 | 0.723 | 0.2973 |
| imd_band | 10–20 | 1,198 | 0.876 | 0.362 | 0.671 | 0.2743 |
| imd_band | 20–30% | 1,252 | 0.871 | 0.342 | 0.649 | 0.2583 |
| imd_band | 30–40% | 1,200 | 0.844 | 0.387 | 0.626 | 0.2722 |
| imd_band | 40–50% | 1,147 | 0.872 | 0.408 | 0.651 | 0.2818 |
| imd_band | 50–60 | 1,103 | 0.848 | 0.361 | 0.606 | 0.2631 |
| imd_band | 60–70% | 1,043 | 0.822 | 0.336 | 0.565 | 0.2552 |
| imd_band | 70–80% | 1,004 | 0.834 | 0.325 | 0.564 | 0.2549 |
| imd_band | 80–90% | 960 | 0.828 | 0.354 | 0.570 | 0.2681 |
| imd_band | 90–100% | 854 | 0.859 | 0.287 | 0.525 | 0.2471 |
| imd_band | ? (unknown) | 395 | 0.731 | 0.234 | 0.403 | 0.2106 |

(week-4 and week-12 show the same qualitative pattern; full per-checkpoint tables are produced by `eval/fairness_breakdown.py` and stored in `artifacts/metrics.json` under `fairness`.)

**Reading the two gaps — recall is flat, FPR is not:**

- **Recall is near-parity everywhere.** 0.822–0.888 across IMD bands, and 0.858 (F) vs 0.852 (M) — the model catches at-risk students at a similar rate in every slice, which is the property that matters most for an intervention tool.
- **The IMD gap follows a real base-rate gradient.** FPR runs 0.287 (least deprived) → 0.450 (most deprived) while true prevalence runs 0.416 → 0.623 over the same bands, and flag-rate falls 0.723 → 0.525. Advisor workload is therefore concentrated on deprived students — and so are their true positives.
- **Gender is the residual gap, and it is *not* explained by base rate.** Prevalence is near-identical (F 0.499 vs M 0.527) and recall is at parity, yet FPR is 0.434 vs 0.292 — women who are not at risk are flagged ~14 points more often. An earlier reading of this section concluded "gender is a non-issue" from prevalence + recall parity; the FPR column says otherwise.

**Recall alone hides the FPR gap, so the mean-score column is the useful second half.** It shows the effective threshold: students who are *not* at risk mean-score 0.297 in the most deprived band vs 0.247 in the least deprived, and 0.291 (F) vs 0.243 (M) — those groups cross the threshold more readily in both directions.

Deprived-vs-affluent aggregate at week 8 (IMD 0–20% vs 70–100%): recall 0.878 vs 0.839, FPR 0.380 vs 0.322, mean P given not-at-risk 0.275 vs 0.257.

**The mechanism is a lower effective threshold for both groups.** Students in those slices who are *not* at risk still score higher on average, so they cross the threshold more readily in both directions: relative to affluent students, deprived students gain ~4 more true positives per 100 at-risk students and take ~6 more false positives per 100 non-at-risk students. (Those are rates on different denominators — recall per 100 *actual at-risk*, FPR per 100 *actual not-at-risk* — so they do not subtract into a single net figure.) This is a directional equity/efficiency trade the model made on its own, and it is the specific thing to examine first if this system were ever deployed to allocate scarce advisor time.

**The residual, honest caveat:** the gender FPR gap (14 points, with prevalence and recall at parity) is *not* a base-rate artifact — women who are not at risk mean-score 0.291 vs 0.243 for men, i.e. a lower effective risk threshold for that group. **Mitigation was out of scope for this iteration and is not implemented**; group-aware thresholds or reweighting are the first things to revisit in a follow-up.

True at-risk prevalence by group — the base rate the IMD FPR gradient tracks (week-8 test rows):

| Group | n | at-risk prevalence |
|-------|---|--------------------|
| IMD 0–10% (most deprived) | 1,104 | 0.623 |
| IMD 10–20 | 1,198 | 0.601 |
| IMD 20–30 | 1,252 | 0.580 |
| IMD 30–40 | 1,200 | 0.523 |
| IMD 40–50 | 1,147 | 0.524 |
| IMD 50–60 | 1,103 | 0.502 |
| IMD 60–70 | 1,043 | 0.470 |
| IMD 70–80 | 1,004 | 0.469 |
| IMD 80–90 | 960 | 0.455 |
| IMD 90–100% (least deprived) | 854 | 0.416 |
| Gender F | 4,761 | 0.499 |
| Gender M | 6,499 | 0.527 |
| IMD ? (unknown) | 395 | 0.339 |

Deprived-band (IMD 0–20%) at-risk prevalence 0.601 vs affluent-band (70–100%) 0.448 — a **15.2-point real base-rate gap**. Gender shows only a 2.8-point gap, and in the *opposite* direction to the gender FPR gap — which is exactly why the gender FPR gap cannot be dismissed as base rate.

## Error analysis: false positives and false negatives (`eval/error_analysis.py`)

A confusion matrix says how many, not who. Run `python eval/error_analysis.py` to reproduce this. At the week-8 operating point (threshold 0.2855) the held-out test gives **TP 4,958 · FP 1,931 · FN 845 · TN 3,526** — 17.1% of all students are false positives and 7.5% are false negatives. The two error types are asymmetric, and that asymmetry is the most useful thing on this page.

**False positives are students who look fine on assessments but weak on engagement.** At week 12, 87.4% of false positives missed *no* assessment and 98.8% have a first-assessment score, averaging **61.2** against **33.7** for true positives — they are flagged on the engagement side of the feature set (sparse VLE activity, late registration, credit load), not on missed work. The typical case is a quiet student who registers late, clicks little, and still passes. The cost is advisor time and one needless conversation: acceptable for a screening tool, and exactly what the human-review step exists to absorb.

**False negatives are students whose submission record is clean but who still fail.** Missed assessments average **0.01** among false negatives versus **1.07** among true positives, and mean predicted probability is 0.177 — the model is confident they are fine. At week 8, **54.6%** of false negatives are true Fail (vs 45.4% Withdrawn), rising to **57.0%** at week 12. This is the structural blind spot: withdrawal leaves a behavioural trace (the student goes silent), while a failing-but-submitting student does not, and **no feature in this dataset observes late assessment quality**. It is not a tuning problem — it is the honest limit of the available signals, and it is the expensive error because a missed student gets no intervention.

**Two worked examples (week 12):**

| Case | Student | P(at_risk) | True result | What the model saw |
|------|---------|-----------|-------------|--------------------|
| False positive | `1777834` (CCC_2014J) | **0.999** | **Pass** | missed 3 assessments, first score 0, 0 homepage clicks in 7 days — then recovered. CCC is also the zero-shot course with no training rows, so this is a cross-course failure too |
| False negative | `1722493` (FFF_2014J) | **0.011** | **Fail** | missed nothing, first score 98, 10 homepage clicks in 7 days — behaviourally indistinguishable from a passing student at this cutoff |

## Explainability (Step 7)

SHAP `TreeExplainer` on the winning pooled LightGBM model:

- Global feature-importance bar chart: `artifacts/plots/shap_global_importance.png`
- Top global drivers: `frac_missed`, `first_assessment_score`, `homepage_clicks_last7`, `n_missed`, `page_clicks_total` (then `mean_earliness_days`) — all available at week 4 (consistent with PR-AUC already being high there).
- `no_assessment_closed` sits at **SHAP rank 73 of 74 with mean|SHAP| = 0.00000** — effectively zero importance (1 feature ranks below it). This is a verified, mechanistic result, not a top-5 coincidence. It is **exactly, symmetrically collinear** with `n_assessments_due`: `no_assessment_closed == 1` iff `n_assessments_due == 0` holds for 100% of rows (it is defined as that comparison, verified across all 97,779 rows). Since `n_assessments_due` also encodes the *magnitude* (1, 2, 3… assessments due), any tree split on `n_assessments_due` recovers the flag perfectly, so the flag contributes strictly no extra information. `n_assessments_due` itself carries the real signal (mean|SHAP| ≈ 0.02) — the tree splits there and the indicator is simply never needed.
  - Worth noting the tempting-but-wrong explanation, since it is a real trap: the flag does *not* become redundant via `n_missed`/`frac_missed`/`first_assessment_score`. That triple signature is genuinely ambiguous — 189 rows (0.19%) have `n_assessments_due > 0`, missed nothing, and recorded a literal first-assessment score of 0, and those rows are 84.7% at-risk versus 41.5% for the true no-assessment-yet rows. The redundancy is really through `n_assessments_due`, and that is the clean, correct account.
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

| Input | P(at_risk) | Risk band |
|-------|-----------|-----------|
| Low-risk preset (missed nothing, strong scores, active) | **0.075** | Low |
| Low-risk preset, then `n_missed` = 5 typed in | **0.387** | High |
| ...then `first_assessment_score` = 100 typed in | **0.361** | High |
| High-risk preset (missed all 5, no scores, 240 credits) | **0.999** | High |

Verified responses (both run against the pinned production model):

| Payload | P(at_risk) | Risk band | Top SHAP driver |
|---------|-----------|-----------|-----------------|
| `artifacts/sample_input.json` — real held-out student 6516 (AAA_2014J, true result Pass), week 12 | **0.1057** | Low | `frac_missed = −0.65` |
| Synthetic engaged student, missed nothing, strong scores | **0.0750** | Low | `first_assessment_score` |

Note that `gender`, `imd_band`, `age_band` and `disability` are **not accepted as model inputs** — they live only in `fairness_eval.parquet`. Sending them is harmless (they are ignored), but they cannot influence the prediction.

## Serving (Step 8)

- **FastAPI:** `serving/api.py` — `POST /predict` accepts student features and returns `at_risk_probability`, an **advisory `risk_band`** (`low` / `moderate` / `high`, anchored to that checkpoint's validation-selected operating threshold, also returned as `operating_threshold`), the SHAP drivers, an explicit **`human_review_note`** ("advisory only … must never trigger automated grading, punishment or admission decisions"), and the artifact metadata; `GET /health`.
- **Streamlit:** `serving/dashboard.py` - four sections: (1) the per-checkpoint PR-AUC comparison chart from Step 5 as the headline visual — pooled LightGBM, RandomForest and the per-checkpoint LightGBM, with the baseline LR quoted beneath it from `comparison.json`; (2) cohort-level risk distribution, (3) per-student drill-down with SHAP explanation and the same risk band, and (4) a **live what-if form** where you type feature values and watch P(at_risk), the risk band and the SHAP breakdown update. A human-review banner is displayed above every prediction. Run with `streamlit run serving/dashboard.py`.
- No MLflow. Model artifacts are written twice per run: **versioned** (`artifacts/models/lgb_pooled_<timestamp>.joblib`, the history) and **pinned** (`artifacts/models/lgb_pooled_production.joblib`, a fixed filename that serving, evaluation, SHAP and the dashboard all load). That removes the ambiguity of "whatever timestamp sorts last." A `metrics.json` is written per training run (`artifacts/metrics.json`) with model name, checkpoint, PR-AUC + CI, date, plus the subgroup and fairness tables.

## What didn't work / honest dead ends

- **The frontend had never been run until it was tested.** `serving/dashboard.py` was compile-checked and then left alone for most of the project's life. Headlessly rendering it surfaced two real bugs that would each have bitten a user, and one of which my own test initially hid:
  1. **Presets didn't change the prediction.** All three preset buttons returned P(at_risk) = 0.997. Cause: the number inputs were re-seeded from the preset's defaults, but Streamlit keeps widget state keyed by `key`, so on the rerun triggered by the radio the fields silently retained their old values. Fixed by scoping the widget keys to the preset (`in_{preset}_{feature}`), which is also the correct behaviour in a real browser, not just a test artifact.
  2. **A dict/Series type error** in the same block (`base.index` on a `dict`) that only fired for the two example presets, never the default one. Fixed with an explicit `isinstance` branch.
  Both were invisible to a syntax check and would have looked like "the dashboard works" until someone clicked a button. **Lesson: "it compiles" is not "it runs" — the two bugs above lived behind a working default render.**
- **Protected attributes leaked into the model feature list.** `gender`, `imd_band`, `age_band` and `disability` were being carried as model features, violating the Step 3 exclusion rule and making the README's claim that they were excluded false. They were silently coerced to constant-zero by `pd.to_numeric(..., errors="coerce")`, so the effect was benign in the strict sense — LightGBM validation PR-AUC is **bit-identical** before and after the fix (0.8905 / 0.9177 / 0.9380) — but it was a real defect that no test caught. It surfaced only because `serving/make_sample_input.py` crashed on `float('F')`. Now prevented three ways: the columns are dropped physically in `features/build_features.py` (with an assert), re-asserted in `models/train.py::feature_columns`, and the API ignores the four attributes entirely (covered by `serving/test_inference.py`). **Lesson: a spec-level constraint ("exclude these from model input") was enforced by convention, not by code, and convention failed silently.**
- **The mandated baseline was under-converged and under-reported.** `LogisticRegression` hit the `max_iter=1000` limit (unscaled lbfgs on features spanning click counts in the thousands vs scores 0–100), which understated it. With `StandardScaler` + `max_iter=5000` its validation PR-AUC goes from 0.8440/0.8929/0.9142 to **0.8716/0.9102/0.9322** — at week 12 within 0.6 points of LightGBM (0.9322 vs 0.9380). A knowingly hobbled baseline would have inflated the main model's apparent advantage, so the scaler is not optional. **Notable takeaway: on this feature set a well-converged linear model is a much stronger baseline than the headline numbers first suggested.**
- **`first_score_missing` was constant zero — an ordering bug, not a design choice.** The indicator was computed *after* `first_assessment_score` had already been `fillna(0)`'d, so it could never be 1: it was 0 for all 97,779 rows and sat in the 74-feature model as dead weight. Fixed by capturing the mask before the fill. The other indicators were audited in the same pass — `engagement_lag_missing` and `reg_date_missing` are computed before their fills (correct), and no equivalent indicator exists for `second_assessment_score`/`score_delta` (their missingness is recoverable from `n_assessments_taken`). Effect of the fix: **LightGBM predictions are bit-identical** (max |Δp| = 0 across the test set — the column never entered a split), while the linear baseline shifted by up to 0.009 on test, which is why the LR rows in the modeling table were refreshed rather than assumed stable.
- **A hardcoded baseline figure in the dashboard went stale.** The caption quoting the LR PR-AUC was written once and never updated, so it drifted from the real value as the baseline was re-trained. It now reads from `comparison.json`. This is the same failure mode as the metric-choice text that used to justify PR-AUC with "highly imbalanced" after the imbalance direction was corrected — **numbers baked into prose are numbers nobody re-checks. The tables in this README were themselves refreshed from `artifacts/metrics.json` for the same reason.**
- **Label imbalance assumption wrong:** `at_risk` is near-balanced / majority in OULAD, not the minority as the spec anticipated. Class-rebalancing tools still run but target the actual minority (`not_at_risk`).
- **Precision = 0.5 operating point is degenerate:** since at_risk is prevalent, flagging everyone clears precision=0.5 for free (recall 100%, precision ~57%). It requires zero model discrimination and cannot trigger a useful intervention. Replaced with precision=0.75 as the primary actionable operating point.
- **GGG/EEE boundary condition:** EEE and GGG have no assessment closed by week 4 (GGG none by week 8). Per the agreed decision these students are **retained** with an explicit `no_assessment_closed` indicator rather than dropped, so coverage is full but week-4/week-8 prediction is harder for those students (no assessment signal). This is why week-4 validation PR-AUC (0.8905) is lower than week 12's (0.9380).
- **`no_assessment_closed` earns its place only as an interpretability handle, not as signal - and it is *exactly* redundant, not approximately.** It is definitionally `n_assessments_due == 0` (100% agreement over all 97,779 rows, verified by `eval/collinearity_check.py`), so any tree split on `n_assessments_due` already recovers it perfectly. SHAP confirms it: rank 73/74, mean|SHAP| = 0.00000, while `n_assessments_due` carries the real signal. Kept for explicit, self-documenting missingness - it contributes no predictive information by construction.
  - The tempting shortcut — "the `n_missed`/`frac_missed`/`first_assessment_score` triple already encodes it" — is **wrong and worth stating as a trap**: that triple signature is genuinely ambiguous, covering 189 rows (0.19%) where assessments *were* due, nothing was missed, and a literal first score of 0 was recorded. Those rows run 84.7% at-risk against 41.5% for true no-assessment-yet rows, so the triple would conflate two very different populations. The redundancy runs through `n_assessments_due`, which is the clean account.
- **`no_assessment_closed` was *not* actually justified by importance.** It is worth being explicit about this: the feature survived because the brief asked for an explicit "no assessment yet" indicator, and it turned out to be zero-value. The correct lesson is that the indicator was the right thing to *try*, and the SHAP/collinearity audit is what proved it unnecessary.
- **Fairness is flagged, not resolved.** Recall is near-parity across groups (0.82–0.89 by IMD band, 0.858 vs 0.852 by gender), but FPR is not: 0.287 → 0.450 across IMD bands, tracking a real 15.2-point prevalence gap, and **0.434 (F) vs 0.292 (M) on gender where prevalence is near-identical** — a residual gap that base rate does not explain. An earlier version of this section read gender as a non-issue by looking only at prevalence and recall; the FPR and mean-score columns show otherwise. Mitigation was out of scope by design, but the gender gap is large enough that it should not stay a footnote in a follow-up iteration.
- **The overall label framing is the weakest part of the brief.** Merging Withdrawn and Fail into one positive class hides a real predictive asymmetry: the model catches Withdrawn students markedly better than Fail students (recall 0.836–0.933 vs 0.680–0.829 at the operating point). If the intervention differs for the two failure modes, a two-stage or three-class formulation would likely serve better — that is a modeling decision, not a bug.
- **NaN-preserving LightGBM was considered and not tested.** An alternative to 0-fill is to leave assessment-dependent features structurally NaN and let LightGBM branch on missingness natively. It was not run: the 0-fill's information loss is already documented as redundant with `n_assessments_due`/`n_assessments_taken` and confined to 347 genuinely-zero rows, so a retrain-and-compare cycle would buy documentation, not performance. Worth measuring only if the feature set changes materially.

## Privacy, representativeness and failure modes

**Representativeness.** OULAD is a UK Open University (adult distance-learning) population — 28,785 students across 7 modules. Results may not transfer to residential or younger cohorts, to institutions with a different assessment cadence, or to modules whose VLE usage differs from these seven. The split is chronological (2013 → 2014B → 2014J), which tests forward-in-time generalisation but only across one year and one institution.

**Privacy.** OULAD is a pseudonymised research release: student identifiers are arbitrary integers and there are no names, emails, addresses or free text. Nothing external is joined to it and no re-identification is attempted. The four demographic fields (`gender`, `disability`, `age_band`, `imd_band`) are excluded from every model input and used **only** for the aggregate fairness audit and `artifacts/metrics.json`; every group reported here has n ≥ 395, so no small cell identifies an individual. This repository redistributes no personal data — only the licence, the citation and regenerable derived features. The API does not persist request payloads.

**Failure modes, explicitly:**

1. **No assessment closed yet** (EEE/GGG at week 4, GGG at week 8): no assessment signal exists, so the model scores on engagement alone. PR-AUC is lower for this slice; it is a known blind spot, retained rather than hidden by dropping rows.
2. **Zero-shot courses** (CCC has no 2013 training rows): its numbers are cross-course generalisation, not in-distribution performance, and `artifacts/metrics.json` marks it as such.
3. **Quiet failure** (the false-negative class above): a student who submits everything and then fails is behaviourally invisible. Fail recall (0.680–0.829) sits structurally below Withdrawn recall (0.836–0.933).
4. **Threshold drift:** the operating threshold is calibrated to ~52% prevalence. If prevalence moves materially (another institution, a different cohort), the precision = 0.75 target no longer holds and the threshold must be re-derived on that population's validation split.
5. **Input contract:** the API assumes OULAD day-offset semantics (`week_elapsed`, `date_registration` relative to course start) and does not range-check inputs. Out-of-domain payloads still return a confident-looking probability.
6. **Never a decision system:** no automated action path exists — no grading, punishment, admission or automatic advisor allocation. The API returns a `risk_band` plus an explicit `human_review_note` on every response, and the dashboard shows the same note above every prediction.

## Reproducing

Requires Python ≥ 3.11. Dependencies are declared in `pyproject.toml` (also resolvable as `uv sync --extra dev` for the pytest suite).

```bash
# 0. Data: download OULAD (CC BY 4.0, DOI 10.24432/C5KK69) from
#    https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
#    and unzip the 7 CSV tables into data/raw/.
#    data/raw/ and data/processed/ are gitignored and regenerable.

# 0b. EDA and data-quality profile (volumes, class balance, missingness, tail)
python eval/eda_report.py

# 1. Features + labels + leakage audit (full-set, raises on violation)
python features/build_features.py

# 2. Train models (baseline LR + pooled & per-checkpoint LightGBM + random forest)
python models/train.py

# 3. Evaluate (PR-AUC + bootstrap CI, validation-selected operating point, course breakdown, subgroup, fairness, metrics.json)
python eval/run_eval.py

# 4. SHAP explainability
python eval/shap_explain.py

# 4b. Fairness deep-dive + no_assessment_closed SHAP standing
python eval/fairness_breakdown.py

# 4c. Exact-collinearity proof for no_assessment_closed
python eval/collinearity_check.py

# 4d. Error analysis: false positives and false negatives at the operating point
python eval/error_analysis.py

# 5. Serve
uvicorn serving.api:app --reload            # FastAPI
streamlit run serving/dashboard.py          # dashboard

# 5b. Test the API by hand
python serving/make_sample_input.py         # export a real student's features to JSON
python serving/test_client.py               # hit /health and /predict
python serving/test_client.py --full        # use the full 74-feature payload

# Tests
python -m pytest features/test_no_leakage.py -q    # preprocessing / leakage; also runs inside step 1
python -m pytest serving/test_inference.py -q      # inference: real artifact, bands, note, no protected inputs

# Frontend verification (headless — no browser needed)
python serving/test_dashboard.py               # renders the app, asserts no exceptions
python serving/test_dashboard_interactive.py   # flips the presets, asserts high-risk >> low-risk
python serving/test_dashboard_manual_input.py  # types into fields, asserts the score moves both ways
```

Both the FastAPI service and the Streamlit dashboard have been verified locally.

Container packaging was intentionally removed for this iteration — the services run directly with `uvicorn` / `streamlit run`. No Dockerfile is maintained.
