# Used Car Market Analysis — SHAP Report

**What does the model look at** when it builds a price? The model explained here is LightGBM — the one called "the model" in the technical report. CatBoost's two variants appear only in the §4 comparison. It works at the level of the whole dataset: which feature takes what share, and how its effect changes across the range. At the single-listing scale there is only one example, in §6.

## 1. What SHAP is, and how it differs from LOFO

**SHAP** splits a single prediction into parts: on this listing age pushed the price up this much, mileage pulled it down that much; the parts add up to the prediction exactly. The **ablation/LOFO** (leave-one-feature-out) in the technical report asks how much the *error* grows when a feature is removed.

## 2. How it was computed

- **Data:** **all** 29,988 listings — no sampling.
- **Model: OOF.** Every listing is explained by the fold model that never saw it in training — the chain's own 5-fold setup. The rebuilt folds' predictions match the chain's stored OOF predictions (largest difference ₺0.00).
- **Method:** `shap.TreeExplainer` (exact). Additivity error 5.1e-13 (measured on the final model), so the parts reconstruct the prediction exactly.
- **Scale:** target `log1p(price)`, so contributions add in log and **multiply** on price (see the table below).
- **Grouping:** the 170 SVD dimensions of the model/series name mean nothing individually, so they are summed into `MODEL_SERIES`. In the §3 table `ENGINE` = hp + cc and `DAMAGE` = the 12 features derived from 13 panels plus the heavy-damage record; in the beeswarm and cohort charts they appear as separate rows. The §6 breakdown only merges the SVD dimensions (`MODEL_SERIES`); every other feature stays separate — there power has its own row; displacement's share on that listing is small, so it stays inside the "other features" bar. A group's value is the **signed sum** of its members on each row — the group's net effect on that listing; the table reports its mean absolute value (§7).

### Which scale in which chart

Each chart uses the scale that is **most meaningful to the reader**:

| scale | where | why |
|---|---|---|
| **log** | beeswarm, cohorts, interaction | The only scale that is comparable across listings **and additive**: `+0.62` means the same thing on a cheap and an expensive car. |
| **%** | dependence curves, the typical-effect column of the §3 table | Turns a contribution into a multiplier: `e^s − 1`. +0.10 → ×1.105, i.e. 10.5% more expensive; −0.10 → ×0.905, 9.5% cheaper. |

**Worked example.** The mean |SHAP| of Age (years) is 0.2870 → `e^0.2870 = 1.332`: a contribution of that size moves the price by 33.2% (the exponential of a mean overstates the typical effect; the typical value is in the table). If Engine (hp + cc) adds 0.1347 on the same listing the percentages do not add — the multipliers multiply: `1.332 × 1.144 = 1.525` → 52.5%, not 33.2% + 14.4% = 47.7%.

## 3. What sets the price

![What drives price](figures/en-sh-01-importance.png)

| feature | mean \|SHAP\| | share | typical effect on price |
|---|---:|---:|---:|
| Age (years) | 0.2870 | 43.9% | 23.5% |
| Engine (hp + cc) | 0.1347 | 20.6% | 12.3% |
| Mileage | 0.0842 | 12.9% | 7.8% |
| Model/series name | 0.0666 | 10.2% | 5.5% |
| Damage (panels + heavy damage) | 0.0412 | 6.3% | 3.9% |
| Segment | 0.0282 | 4.3% | 2.4% |
| Body type | 0.0070 | 1.1% | 0.4% |
| Transmission | 0.0043 | 0.7% | 0.2% |
| Fuel | 0.0009 | 0.1% | 0.1% |
| Drivetrain | 0.0001 | 0.0% | 0.0% |
| Brand | 0.0001 | 0.0% | 0.0% |

*These shares are specific to LightGBM (§4). **Typical effect on price** is the median over listings of the contribution's price equivalent (`|e^s − 1|`).*

**Age (years)** alone takes 43.9% of the attribution, the top three 77.4%.

The ranking is not fully stable across folds: the item in position(s) 3, 4 changes in at least one fold; do not read much into the order of items with close shares.

### Direction check

![Each dot is a listing](figures/en-sh-02-beeswarm.png)

Direction of effect (Spearman rank correlation): age **-0.99**, mileage **-0.96**, engine power **+0.89**. More age and mileage push the price down, more power pushes it up. This is a sanity check enforced in the generator: if any sign flips, the report is not produced.

![Feature value and its effect on price](figures/en-sh-03-dependence.png)

The three curves:

- **Age:** median contribution +90.4% at 0–2 years and -47.0% at 18+, a **3.6×** gap in the credit the model gives age; the actual median price ratio of the same two groups is 5.7×. The difference comes from old cars also differing in other features; SHAP books that part to them — mostly Mileage, Engine (hp + cc), Damage (panels + heavy damage).
- **What mileage costs** (difference between consecutive window medians; window centre = the window's median mileage; scaled to 100k km because the windows are not evenly spaced): 8.7% between 28k and 127k km, 11.4% between 127k and 223k, 7.2% between 223k and 385k. The rate drops clearly in the last window (0.64× the previous one): at very high mileage extra kilometres cost less, but they still cost. At 350k km and above there are 1,723 listings.
- **Engine power:** median contribution across power bands <150 hp -14.9% → 150–200 hp +7.7% → 200–250 hp +15.5% → 250+ hp +43.2%; from sub-150 hp to 250+ hp the gap in credit is **1.7×**.

### shap's own split: two groups

![Two cohorts](figures/en-sh-04-cohorts.png)

We did not pick the split: shap's own decision tree cut the data at **Age (years) = 9.5**. The mean |SHAP| of *Age (years)* is 0.396 in `Age (years) < 9.5` and 0.230 in `Age (years) >= 9.5`. Reading that as "the model looks at age more on new cars" misleads: |SHAP| measures the deviation from the average, so at a typical age the age contribution is small. Mean |SHAP| by age band: 0–4 0.577 · 5–8 0.333 · 9–12 0.084 · 13–16 0.271 · 17+ 0.611 — V-shaped; age weighs heavily on both new and very old cars.

### Age and mileage act together

![Same mileage, different age](figures/en-sh-05-km-age.png)

Colour is the car's age. In the same **150–250k km** band the median mileage contribution is -0.030 under 10 years (2,232 listings) and -0.006 at 10 or older (9,000 listings): the model charges less for mileage on an older car. The two features are not independent (§7).

## 4. Three variants build the same price for different reasons

| feature | LightGBM | CatBoost (SVD) | CatBoost (native) |
|---|---:|---:|---:|
| Age (years) | 43.9% | 30.9% | 32.1% |
| Engine (hp + cc) | 20.8% | 17.2% | 17.1% |
| Mileage | 12.9% | 22.1% | 22.3% |
| Model/series name | 8.8% | 13.7% | 16.7% |
| Damage (panels + heavy damage) | 6.5% | 7.6% | 5.5% |
| Segment | 5.2% | 7.1% | 0.1% |
| Body type | 1.0% | 0.4% | 2.9% |
| Transmission | 0.7% | 0.9% | 1.4% |
| Fuel | 0.1% | 0.1% | 0.3% |
| Drivetrain | 0.0% | 0.1% | 0.5% |
| Brand | 0.0% | 0.0% | 1.1% |

On accuracy the three variants are close (MAPE 6.49% · 6.44% · 6.58%). They differ on reasoning: LightGBM gives age 43.9%, CatBoost (SVD) 30.9% (13.0 points apart); on mileage it reverses (12.9% · 22.1%). Age and mileage move together, so which one gets the credit is the model's preference. **Takeaway:** the top three items are the same in all three models (Age (years), Engine (hp + cc), Mileage), but their order and shares depend on the model; in CatBoost (native) the 3rd and 4th items are only 0.4 points apart.

*This table compares the three models in their **final** form and under the same rule as §3: a group's value is the signed per-row sum. In the native variant the model and series names are two separate features; here they are summed per row as well. The §3 shares come from the OOF models, the LightGBM column here from the final model; the same item differs by at most 1.4 points between the tables (Model/series name).*

## 5. Why brand is next to nothing

`brand` has mean |SHAP| **0.0001**, 0.0% of the attribution. The model name already determines the brand (technical report §3: U(brand | model) = 1.00); adding brand on top of series+model changes the mean error by ₺1.

Three methods — dependence, ablation, SHAP — land in the same place: **brand carries no separate information.** That is not "brand does not affect price"; its effect sits inside the model name.

## 6. How one prediction is built

![118i Standart](figures/en-sh-06-waterfall.png)

At the bottom `E[f(X)]` is what the model predicts before it sees any feature, at the top `f(x)` is its prediction for this listing; each arrow between them is one feature's contribution, and the "15 other features" arrow is the rest summed (log scale, §2).

- **Listing:** 118i Standart · 2007 · 330,000 km
- **Actual price:** ₺595,000
- **Model's prediction** (it never saw this listing)**:** ₺519,680 (-12.7%)

The three biggest contributions:

1. Age (years) = 19 yr → ×0.54
2. Model/series name = 118i Standart → ×0.87
3. Engine power (hp) = 138 hp → ×0.89

The other 21 items together multiply the price by 0.76 (-23.5%) — not small in total.

No bar accounts for the 12.7% gap: SHAP decomposes the **prediction**, not the actual price. What the model does not know is not written in the breakdown.

## 7. Limitations

- **Attribution, not causation.** SHAP says what the model used, not how the market works.
- **Features that move together share the credit.** How age and mileage split it depends on the model (§4). TreeSHAP runs here in `tree_path_dependent` mode, i.e. observational, which also shapes the split.
- **Contributions live in log space.** When comparing items, read the **multipliers**: they mean the same thing on every listing. Converting to lira is listing-specific and order-dependent, so this report does not do it.
- **The grouping is a decision.** `MODEL_SERIES` is the signed per-row sum of 170 SVD dimensions, i.e. the name's net effect on that listing; taken one by one each dimension looks small. Summing the members' |SHAP| separately was not used: that sum grows with how many pieces the input is split into — the same name summed over its 170 dimensions comes out 2.5× larger.
