# BayesOpt — Bayesian optimisation for MNA / LV simulation

Bayesian optimisation layer for tuning MNA parameters (`contraction`, `contraction_velocity`) against haemodynamic outputs from a numerical LV + MNA simulation.

---

## Two different “scores” (do not confuse them)

### 1. Cardiac objective (you define this)

Maps simulation outputs to one scalar `y` to **maximise**:

```text
(contraction, contraction_velocity)
        -> simulation
        -> LVEDP, flows, volumes, ...
        -> cardiac objective y
```

This tells BO which MNA settings are good.

Implemented in `objective.py` (`LiteratureObjective` only).

### 2. GP hyperparameter tuning (automatic)

The Gaussian Process learns:

```text
(params) -> y
```

from past runs. Once the kernel is chosen, prediction uses **covariance matrices**:

```text
K   = k(X_train, X_train) + noise
k*  = k(X_train, X_new)

mean(x_new) = k*ᵀ K⁻¹ y
var(x_new)  = k** - k*ᵀ K⁻¹ k*
```

Before predicting, sklearn **tunes kernel hyperparameters** (`length_scale`, `signal_variance`, `noise_level`) by maximising the **log marginal likelihood (LML)**:

```text
log p(y | X, θ)
```

That is the GP’s internal “loss”. It helps choose how smooth/noisy the surrogate should be. **You do not write this loss yourself** — `GPSurrogate.fit()` handles it.

**Summary**

| What | Optimised by | Score |
|------|--------------|-------|
| MNA parameters | Bayesian optimiser | Literature suitability score |
| GP kernel HPs | LML during `gp.fit()` | Automatic |

Workflow:

```text
1. Build GP from past (params, y) pairs
2. Tune kernel HPs with LML on that training set
3. Use covariance matrices to infer mean + uncertainty at new params
4. Acquisition function picks the next simulation to run
```

---

## Literature-consistent safety filter

Before the soft objective is scored, `LiteratureSafetyFilter` applies a **hard reject** (`y = -inf`) if any rule fails.

A parameter set is **literature-consistent** if:

| Rule | Criterion | Source |
|------|-----------|--------|
| Aortic flow | `aortic_flow > baseline` | Moreira 1992; Caputo 2000; Frey 1993 |
| Stroke volume | `SV_LV > baseline` | Moreira 1992; Goldenberg 1996; Chiu 1997 |
| Filling pressure vs baseline | `LVEDP <= baseline + margin` | Caputo 2000; Goldenberg 1996; Frey 1993 |
| Absolute LVEDP | `LVEDP < 16 mmHg` | Nagueh ASE 2025 |
| Ejection fraction | `EF_LV >= 0.35` | Conservative engineering limit; recalibrate for advanced HF |
| Flow balance | `\|aortic_flow - pulmonary_flow\| / ref <= 25%` | Engineering circulation-consistency check |

Default margins (configurable in `LiteratureSafetyFilter`):

- `lvedp_margin_mmHg = 2.0`
- `lvedp_absolute_max = 16.0`
- `min_ef = 0.35`
- `max_flow_imbalance_ratio = 0.25`

Implementation: `objective.py` → `LiteratureSafetyFilter`, `LiteratureObjective`.

### Soft score (after passing the filter)

```text
y = w_aortic_flow * (aortic_flow / baseline_aortic_flow)
  + w_stroke * (SV_LV / baseline_SV)
  + w_stroke_work * (stroke_work_proxy / baseline_SW)
  - penalties(LVEDP, RVEDP)
```

Default weights (`2.0`, `1.0`, `0.5`, ...) are initial engineering priorities,
not fitted from data. They encode: aortic flow > stroke volume > stroke work.

Stroke-work proxy: `(LVESP - LVEDP) * SV_LV` (Kass, CV Physiology; Moreira 1992).

---

## Quick start

### Simple supervisor-facing API

For most use cases, define one episode function and let the API run the full
Bayesian optimisation loop:

```python
from api import run_mna_bayesopt


def simulate_episode(contraction, contraction_velocity, n_beats):
    """Run one complete LV + MNA simulation episode at fixed parameters."""
    # Replace this with the supervisor's model call.
    return {
        "LVEDP": ..., "LVEDV": ...,
        "LVESP": ..., "LVESV": ...,
        "RVEDP": ..., "RVEDV": ...,
        "RVESP": ..., "RVESV": ...,
        "aortic_flow": ..., "pulmonary_flow": ...,
    }


results = run_mna_bayesopt(
    simulate_episode=simulate_episode,
    n_beats=10000,
    kernel_name="matern52",
    contraction_bounds=(0.1, 1.0),
    contraction_velocity_bounds=(0.05, 2.0),
    n_init=5,
    n_iter=25,
    save_path="BayesOpt/results/my_run.json",
)

print(results["best_params"])
print(results["best_outputs"])
```

If the model must be launched as a Python script, the script should print one
JSON object with the haemodynamic outputs. Then wrap it like this:

```python
from api import make_script_episode_function, run_mna_bayesopt

simulate_episode = make_script_episode_function("path/to/simulation_script.py")
results = run_mna_bayesopt(
    simulate_episode=simulate_episode,
    n_beats=10000,
    kernel_name="matern52",
)
```

The lower-level optimizer API below is kept for advanced experiments.

### 1. Run a baseline simulation (MNA off / nominal)

```python
baseline_outputs = my_lv_sim(contraction=0.0, contraction_velocity=0.0)
baseline = BaselineProfile.from_outputs(baseline_outputs)
```

### 2. Build objective with literature filter

```python
from objective import BaselineProfile, LiteratureObjective

objective = LiteratureObjective(baseline=baseline)
```

### 3. Run optimisation

```python
from optimizer import BayesianOptimizer
from kernels import get_kernel
from acquisition import CoolingUCB
from api import FunctionAdapter

adapter = FunctionAdapter(my_lv_sim, param_names=["contraction", "contraction_velocity"])

opt = BayesianOptimizer(
    bridge=adapter,
    objective=objective,
    param_bounds={
        "contraction": (0.1, 1.0),
        "contraction_velocity": (0.05, 2.0),
    },
    kernel="matern52",
    acquisition=CoolingUCB(budget=25),
)
opt.run(n_init=5, n_iter=25)
```

### 4. Inspect safety of a result

```python
passed, violations = objective.safety_check(opt.best_outputs)
print(passed, violations)
```

---

## Folder layout

```text
BayesOpt/
├── api.py                 connect simulation + run BO (main entry point)
├── objective.py           cardiac objectives + literature safety filter
├── optimizer.py           main BO loop
├── gp_surrogate.py        GP wrapper
├── kernels.py             kernel zoo
├── acquisition.py         EI, UCB, CoolingUCB, ...
├── run_demo.py            end-to-end demo with MockSimulation
└── tools/plot_convergence.py
```

Smoke test:

```bash
python BayesOpt/run_demo.py
```

---

## References

### Dynamic cardiomyoplasty / skeletal muscle assist

1. **Caputo RJ et al.** Dynamic cardiomyoplasty decreases myocardial workload as assessed by tissue tagged MRI. *ASAIO Journal* 2000. Successful assist: leftward PV-loop shift, increased peak pressure and dP/dt, LVEDP stable or slightly decreased.  
   https://journals.lww.com/asaiojournal/fulltext/2000/09000/dynamic_cardiomyoplasty_decreases_myocardial.9.aspx

2. **Moreira LFP et al.** Full-thickness dynamic cardiomyoplasty of the left ventricle with free revascularized latissimus dorsi myografts. *J Thorac Cardiovasc Surg* 1992. Synchronized assist increased cardiac output (~24%) and LV stroke work (~44%) with LA pressure 8–12 mmHg.  
   https://www.sciencedirect.com/science/article/pii/S0022522319347014

3. **Frey AW et al.** Long-term follow-up after dynamic cardiomyoplasty. *J Thorac Cardiovasc Surg* 1993. Unstimulated wrap can increase LVEDP and reduce output; synchronized stimulation augments CO and LV systolic pressure.  
   https://www.sciencedirect.com/science/article/pii/0735109793901887

4. **Goldenberg S et al.** Left ventricular function changes after cardiomyoplasty in dilated cardiomyopathy. *Eur J Cardiothorac Surg* 1996. Stroke volume and stroke work index increased; pulmonary wedge pressure decreased (~25 to ~17 mmHg).  
   https://www.sciencedirect.com/science/article/pii/S0022522319365924

5. **Chiu RC-J et al.** An engineering model of dynamic cardiomyoplasty. *Ann Biomed Eng* 1997. Model framework linking assist to stroke volume augmentation.  
   https://pubmed.ncbi.nlm.nih.gov/9570227/

### LV filling pressure thresholds

6. **Nagueh SF et al.** Recommendations for the evaluation of left ventricular diastolic function by echocardiography (2025 ASE update). Elevated LVEDP: **> 16 mmHg** at rest. Normal filling pressures discussed in context of HFpEF.  
   https://onlinejase.com/article/S0894-7317(25)00157-9/fulltext

7. **Multicenter invasive validation (2025).** LVEDP ≥ 16 mmHg used as elevated filling pressure on catheterization.  
   https://www.medrxiv.org/content/10.1101/2025.09.08.25335376v1

### Cardiac assist optimal control (multi-objective compromise)

8. **Grosan Y et al.** An intra-cycle optimal control framework for ventricular assist devices. *Ann Biomed Eng* 2021. Multi-objective: ventricular unloading + aortic valve opening / perfusion.  
   https://link.springer.com/article/10.1007/s10439-021-02848-2

9. **Timm F et al.** Numerical optimal control of turbo dynamic ventricular assist devices. *Bioengineering* 2014. Objective combines aortic flow and LV stroke work; target CO ~ 5 L/min in human model.  
   https://www.mdpi.com/2306-5354/1/1/22

10. **Azarnoush H et al.** Extremum-seeking control of LVAD to maximize cardiac output and prevent suction. *Commun Nonlinear Sci Numer Simul* 2021. Maximize flow subject to suction / safety constraints.  
    https://www.sciencedirect.com/science/article/abs/pii/S0960077921003672

### Pressure–volume physiology

11. **Kass DA et al.** Ventricular pressure-volume relationship. *CV Physiology* (Educational). SV = EDV − ESV; stroke work = area of PV loop.  
    https://cvphysiology.com/cardiac-function/cf024

12. **Frontiers in Physiology (2021).** Rodent PV-loop methods: SV, CO, LVEDP definitions for small-animal models.  
    https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.751326/full

---

## Notes for rat vs human models

Thresholds above are derived mainly from **human/clinical** and **large-animal** literature. For rat simulations:

1. Run an **unassisted baseline** in your model first.
2. Use **relative improvements** (`output / baseline`) in the objective.
3. Keep the hard filter but **calibrate** `lvedp_absolute_max` and `min_ef` to your species/model if needed.

The filter is intentionally conservative: unsafe states are rejected before BO can reward them.
