# Truth probes in Gemma-2-9B: what transfers across datasets and what doesn't

Coursework project (BSU, 2026) on the mechanistic interpretability of language models. I train linear probes for "is this statement true?" on Gemma-2-9B activations and ask a practical question: **if a probe works on one dataset, can you use it on another?**

Short answer: the *direction* transfers almost perfectly, the *decision threshold* does not.

Built on top of [ARENA](https://www.arena.education/) section 1.3.1 (Geometry of Truth datasets: cities, Spanish–English translation, larger/smaller-than, their negations, companies, common claims), adapted to run on a free Kaggle T4. Two probe types: mass-mean (MM) and logistic regression (LR).

## Key results

All numbers are from the saved outputs of `gemma-2-9b-truth-probes.ipynb`.

**1. Where truth becomes linearly readable.** A layer sweep over all 42 layers: cross-dataset transfer between cities / sp_en_trans / larger_than appears around layers 12–17 and peaks at **layer 23** (mean transfer AUROC 0.999, worst pair 0.994).

**2. Ranking transfers, thresholds don't.** On five logical groups (each dataset merged with its negation, plus companies and common_claim, layer 20), MM probes keep a mean off-diagonal AUROC of **0.890**. Using the threshold inherited from the source dataset costs **0.231** accuracy on average compared with the best threshold for the target. The extreme case: a probe trained on cities± and tested on size± has **AUROC 1.000 but accuracy 0.513**. It separates true from false perfectly, and the inherited threshold still gives a coin flip.

**3. Why: a three-factor decomposition.** Transfer separability decomposes as

  d′<sub>transfer</sub> ≈ d′<sub>native</sub> × cos(source, target direction) × σ<sub>target</sub> / σ<sub>source</sub>

and the formula matches the measured value closely (e.g. cities → sp_en_trans: 4.06 measured vs 4.14 from the formula). Cosines between dataset directions are only **0.25–0.40**, but the spread ratio σ<sub>tgt</sub>/σ<sub>src</sub> of **1.4–3.0** compensates. This is a decomposition of causes, not a predictive formula.

**4. Threshold recalibration without target labels.** Choosing the threshold from unlabeled target activations recovers most of the gap: quantile thresholding recovers 93% (LR) / 84% (MM), a 2-component GMM 49% / 62%. The datasets are class-balanced, which favours the quantile method.

**5. The direction is causal.** Intervening along the truth direction at layers 12–23 (cities):

| Intervention | True statements, mean P(TRUE) − P(FALSE) | False statements |
|---|---|---|
| none | +0.885 | −0.945 |
| add | +0.867 | −0.636 |
| subtract | **−0.901** | −0.915 |
| ablate (project out) | +0.857 | −0.946 |

Subtracting the direction flips true statements to "false". Projecting it out barely matters, consistent with a small projection relative to ‖h‖. On companies, subtracting the direction is **6.3σ** away from random directions of the same norm.

**6. Deception steering (Gemma-2-9B-it).** Steering along a deception direction gives a sharp behavioural switch. With greedy decoding the model still refuses to make health claims about a supplement up to coefficient 0.25; at **0.30** it states the product is "scientifically proven", and at 0.40–0.50 it presents itself as "a leading expert". With sampling (T = 1) the switch already happens at 0.25. Control: random directions of the same norm at 0.30 keep the honest refusal under greedy decoding (5 of 5 seeds) and in 4 of 5 seeds under sampling.

**Also:** on negated statements the probes beat the model's own output readout (neg_cities: probes AUROC 1.000 vs logits 0.729). Probes trained without negations invert on them, which is why negated pairs are merged into one training group.

## Files

| Notebook | Contents |
|---|---|
| `gemma-2-9b-truth-probes.ipynb` | Main notebook: all results above. |
| `llama-2-13b-truth-probes.ipynb` | The same experiments set up for LLaMA-2-13B: calibration cost, layer selection for interventions, sign-asymmetry causal test. Code only; this version has no saved outputs. |

## Limitations

- One train/test split per dataset; no confidence intervals on most numbers.
- The probe and intervention layers were chosen on the same datasets they are evaluated on.
- All datasets are class-balanced.
- The written thesis is in progress.

## Running

The notebooks run on Kaggle with a T4 GPU. Gemma is gated: accept the licence on Hugging Face and add your token as a Kaggle Secret named `HF_TOKEN`.
