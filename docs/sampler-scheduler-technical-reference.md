# Sampler and Scheduler Technical Reference

This document covers the sampler and scheduler set requested for the native
sdkit3 backend. A **sampler** is the numerical update rule that advances the
latent. A **scheduler** constructs the decreasing noise-level (`sigma`) sequence
on which that rule operates. They are separate choices: a solver can run on many
schedules, even when one pairing is usually preferable.

The `*` shown in the UI is advisory. It marks the best pairing in the supplied
[ComfyUI compatibility matrix][compatibility-matrix]; it does not change the
selection or reject other combinations. Blank cells in that guide mean “not a
clear best pairing,” not “mathematically impossible.”

Native status below is based on the compiled source path:

- public enums in [`stable-diffusion.h`][native-api];
- sigma schedules and solver updates in [`denoiser.hpp`][native-denoiser];
- the Easy Diffusion API translation in [`server.cpp`][native-server]; and
- the image controls in [`image-settings.plugin.html`][image-settings].

## Citation accuracy note

The supplied research index was useful as a checklist, but several identifiers
must not be used as implementation sources:

- arXiv `2310.13261` is a mixed-integer-programming paper, not DPM-Solver-v3 or
  the source of DPM++ 3M SDE;
- arXiv `2403.02321` concerns axion sensing, not RES Multistep;
- arXiv `2401.07122` concerns federated learning, not ER-SDE;
- arXiv `2410.05439` concerns barycentric interpolation, not Gradient
  Estimation; and
- [Align Your Steps][ays-paper] (`2404.14507`) is a real diffusion scheduling
  paper, but it defines AYS rather than the `atan`/`tan` KL Optimal schedule.

The PNDM paper (`2202.09778`) is relevant historical lineage, but the exact
IPNDM behavior audited here follows the current diff-sampler/ComfyUI method.
The sections below therefore cite the paper or reference implementation that
actually matches each option's equations and the code in this repository.

## ModelsLab Diffusers++ source audit

[`ModelsLab/diffusers_plus_plus`][diffusers-plus-plus] is a useful secondary
Python port reference, particularly because it keeps each combined solver and
noise schedule in a self-contained PyTorch class. Its current `main` checkout
is commit [`d1fd977`][diffusers-plus-plus-commit], dated 2024-09-18 and reporting
Diffusers `0.31.0.dev0`. Comparing that merge with its Hugging Face upstream
parent shows that the scheduler tree came from upstream Diffusers; it is not a
separate ModelsLab implementation of these algorithms. It should therefore be
treated as a readable 2024 snapshot, not as the newest authority in 2026.

For the requested native backlog it contributes three useful implementation
views:

- [`scheduling_unipc_multistep.py`][dpp-unipc] contains UniP/UniC history,
  `bh1`/`bh2` coefficient construction, corrector-disable control, solver-order
  warmup, and lower-order final-step handling.
- [`scheduling_deis_multistep.py`][dpp-deis] contains first-, second-, and
  third-order log-rho DEIS updates, including variable-grid integral
  coefficients. Its default recommendation is order two for guided sampling
  and order three for unconditional sampling.
- [`scheduling_ipndm.py`][dpp-ipndm] is a compact cross-check for IPNDM's
  trigonometric alpha/sigma schedule and the 1→4 order Adams–Bashforth history
  ramp already present in this native core.

Its [`DPMSolverMultistepScheduler`][dpp-dpm] is **not** a source for the
requested DPM++ 3M SDE option. In this snapshot the documented SDE configuration
is second order; the third-order update has deterministic `dpmsolver` and
`dpmsolver++` branches only. The random SDE term is used by the first- and
second-order paths, so that class must not be relabeled as DPM++ 3M SDE.

Diffusers++ also identifies possible future, out-of-scope native candidates:
[`SASolverScheduler`][dpp-sa-solver] is a high-order stochastic Adams-style
predictor/corrector with a time-dependent `tau` controlling ODE/SDE behavior,
and [`CosineDPMSolverMultistepScheduler`][dpp-cosine-dpm] is a second-order
cosine-preconditioned DPM solver for Stable Audio Open. FlowMatch Heun and TCD
are also present. These are distinct options, not aliases for any missing item
in the requested matrix.

The snapshot does not supply implementations named RES Multistep, Gradient
Estimation, ER-SDE, DDIM Uniform, Linear Quadratic, or KL Optimal. Their sources
remain the references cited in the corresponding sections below.

## Current Hugging Face Diffusers cross-check

The current official [`huggingface/diffusers`][hf-diffusers] tree was also
audited at commit [`c419dac`][hf-diffusers-commit], dated 2026-09-11 and
reporting version `0.41.0.dev0`. This is the preferred Python cross-check over
the older ModelsLab snapshot.

It materially improves the reference set used for the native ports:

- Current [`DPMSolverMultistepScheduler`][hf-dpm] implements a third-order
  `sde-dpmsolver++` branch. It forms `D0`, `D1`, and `D2` from three denoised
  histories, applies the exponential reverse-SDE coefficients, and injects a
  seeded or caller-supplied variance-noise tensor. This is a valid formula and
  edge-case reference for DPM++ 3M SDE. The native implementation follows the
  ComfyUI specialization instead, using the existing step-count-stable
  Brownian-tree noise sampler and native `eta` control.
- Current [`UniPCMultistepScheduler`][hf-unipc] retains `bh1`/`bh2` UniP/UniC
  but now handles epsilon, sample, velocity, and flow prediction. It can build
  Karras, exponential, Beta, or shifted flow sigma grids and accepts terminal
  shifting/custom flow sigmas. Those schedule choices remain in the separate
  native scheduler layer. The native `unipc` option implements ComfyUI's
  canonical `bh1` predictor/corrector, warms from
  first through third order, lowers the order at the end, and currently targets
  the ordinary latent-image sigma parameterization rather than newer flow
  extensions.
- Current [`DEISMultistepScheduler`][hf-deis] likewise supports epsilon,
  sample, velocity, and flow prediction plus Karras, exponential, Beta, and
  shifted flow grids. Its solver itself remains log-rho DEIS with first- through
  third-order update paths and lower-order final-step stabilization. The native
  implementation ports those analytic log-rho coefficients directly into the
  k-diffusion state instead of using ComfyUI's per-step numerical quadrature.
- Current [`IPNDMScheduler`][hf-ipndm] confirms the same trigonometric
  alpha/sigma construction and Adams–Bashforth 1→4 history ramp already audited
  in the native implementation.
- Current [`DDIMScheduler`][hf-ddim] provides another reference for uniform
  discrete timestep selection: `timestep_spacing="leading"` uses an integer
  training-step ratio and reverses the resulting grid. The native DDIM Uniform
  implementation matches ComfyUI's exact stored-sigma indexing and endpoint
  convention rather than aliasing a whole Diffusers scheduler class.

The official [scheduler mapping guide][hf-scheduler-guide] also reinforces an
important API distinction: Diffusers classes combine an update algorithm and
schedule configuration, while this native UI deliberately exposes **sampler**
and **scheduler** as independent controls. Flags such as `use_karras_sigmas`,
`use_exponential_sigmas`, and `use_beta_sigmas` therefore map to native
scheduler choices, not to new sampler names.

### New upstream techniques outside the requested matrix

The official tree contains newer work worth tracking, but none should be
silently substituted for a requested option:

- `FlowMatchLCMScheduler` applies LCM-style few-step updates to flow-matching
  models and adds static/dynamic resolution-dependent shifts and optional
  scale-wise latent upsampling.
- `SASolverScheduler` is a high-order stochastic Adams predictor/corrector;
  its time-dependent `tau` controls ODE/SDE behavior.
- `SCMScheduler` implements few-step stochastic consistency sampling with its
  model-specific trigonometric-flow prediction.
- FlowMatch Euler and Heun integrate flow velocity rather than the usual
  epsilon/data prediction; they require the matching model parameterization.
- Cosine DPM, MiniMax-H3, Helios, and LTX Euler ancestral RF are model-specific
  audio/video methods rather than general image UI choices.
- Discrete DDIM, Entropy Bound, and Block Refinement operate on token IDs and
  logits, not continuous image latents. Discrete DDIM is therefore unrelated
  to the requested DDIM Uniform image sigma scheduler.

The official tree still has no implementations named RES Multistep, Gradient
Estimation, ER-SDE, Linear Quadratic, or KL Optimal. Their direct sources remain
the references cited in the corresponding sections below.

## Requested samplers

| UI option | Native identifier | Method and cost | Randomness | Matrix recommendation | Native status |
|---|---|---|---|---|---|
| Euler | `euler` | First-order explicit Euler; one model evaluation per step | Deterministic at the current zero-churn defaults | Normal | Available |
| Euler a | `euler_a` | First-order Euler ancestral; one evaluation per step | Adds ancestral noise at every nonterminal step; native default `eta=1` | Exponential | Available |
| DPM++ 2M | `dpm++2m` | Second-order, data-prediction multistep solver; one evaluation per step after history startup | Deterministic | Karras | Available |
| DPM++ 2M SDE | `dpm++2m_sde` | Second-order multistep reverse-SDE solver, midpoint variant | Stochastic; native default `eta=1` | Karras | Available; API routing added in this audit |
| DPM++ 3M SDE | `dpm++3m_sde` | Third-order multistep reverse-SDE solver | Stochastic with seeded Brownian-tree increments; native default `eta=1` | Linear Quadratic | Available; native solver and UI/API routing added |
| UniPC | `unipc` | Unified `bh1` predictor/corrector; order warms from one to three and is lowered at the end | Deterministic in its ordinary ODE form | KL Optimal | Available; native solver and UI/API routing added |
| LCM | `lcm` | Consistency-model step followed by re-noising for the next level; one evaluation per step | Stochastic before the terminal step | SGM Uniform | Available |
| DEIS | `deis` | Analytic log-rho exponential-integrator solver; order warms from one to three | Deterministic in the standard ODE form | Simple | Available; native solver and UI/API routing added |
| RES Multistep | `res_multistep` | Second-order exponential multistep update using the previous denoised estimate and `phi_1`/`phi_2` coefficients | Native UI route is deterministic because its default `eta=0` | Karras | Available; UI/API routing added in this audit |
| Heun | `heun` | Explicit trapezoidal predictor/corrector; normally two evaluations per step, with a first-order terminal step | Deterministic at zero churn | No recommendation for plain Heun; the guide recommends Karras for **Heun++2**, which is a different sampler | Available |
| DDIM | `ddim_trailing` | Native compatibility path uses the Euler-ancestral update with `eta=0` and normally the Simple schedule | Deterministic with the native default | No recommendation for the DDIM sampler itself | Available as **DDIM trailing**, not as every DDIM timestep-spacing variant |
| IPNDM | `ipndm` | Explicit Adams–Bashforth multistep integration; order ramps from one to four as derivative history becomes available | Deterministic | DDIM Uniform | Available |
| Gradient Estimation | `euler_ge` | Euler derivative extrapolation; after startup the default `gamma=2` uses `2*d_current - d_previous` | Deterministic through the current UI route | No recommendation for the plain method; Beta is recommended only for the separate CFG++ variant | Available; UI/API routing added in this audit |
| ER-SDE | `er_sde` | Third-stage extended reverse-time SDE solver with denoised-history corrections | Stochastic; native default `eta=1` | Exponential | Available; UI/API routing added in this audit |

### Euler and Euler ancestral

Euler uses the score-derived ODE derivative
`d = (x - denoised) / sigma` and performs `x += d * delta_sigma`. It is the
lowest-cost baseline and reacts strongly to scheduler placement because it has
only first-order local accuracy. The native implementation has no churn exposed
through the Easy Diffusion UI, so repeated runs with the same inputs and seed are
deterministic.

Euler ancestral first splits the requested transition into `sigma_down` and a
noise scale `sigma_up`, advances to `sigma_down`, and injects fresh Gaussian
noise scaled by `sigma_up`. This makes the path seed-dependent throughout the
trajectory, not only at initial latent creation. For rectified-flow models the
native core uses a separate bounded ancestral transformation and clamps `eta`
to its stable range.

Both implementations follow the Euler family used by the [EDM reference
sampler][edm-paper] and [ComfyUI's k-diffusion sampler][comfy-kdiffusion].

### Heun

Heun first takes an Euler prediction at the next sigma, evaluates the network
again there, and advances with the average of the two derivatives. That
predictor/corrector step gives second-order accuracy but nearly doubles network
work relative to one-evaluation multistep methods. At terminal sigma zero, the
implementation falls back to a single Euler-style denoised result because the
second derivative evaluation would be singular. See the [EDM paper][edm-paper].

### DPM++ 2M, 2M SDE, and 3M SDE

DPM-Solver++ is formulated for guided sampling in data-prediction space. `2M`
means second-order **multistep**: it reuses the previous denoised prediction, so
it obtains second-order behavior with one new network evaluation per step after
startup. The final zero-sigma step returns the denoised prediction directly.

`2M SDE` adds a reverse-SDE term controlled by `eta`. The native option is the
midpoint form and uses fresh seeded Gaussian noise. The core also contains a
separate Brownian-tree variant (`dpm++2m_sde_bt`), but that variant is not the
requested UI option. `3M SDE` keeps two prior denoised estimates and step sizes,
adds first- and second-difference corrections, and uses step-count-stable
Brownian increments. Its default is `eta=1`; `eta=0` removes the stochastic
term while retaining the multistep update. Rectified-flow denoisers use
`log((1-sigma)/sigma)` half-log-SNR and the corresponding `1-sigma` data
coefficient, with the singular first sigma offset before integration. The
mathematical source is the
[DPM-Solver++ paper][dpmpp-paper], and the concrete specialization follows
[ComfyUI's sampling source][comfy-kdiffusion].

### UniPC

UniPC combines a unified predictor (UniP) and corrector (UniC). It is
training-free and is designed to retain accuracy at very low step counts by
increasing solver order. The native path uses the data prediction already
returned by the denoiser, solves the small `bh1` coefficient systems locally,
applies UniC before each subsequent UniP update, and follows ComfyUI's
first-to-third-order warmup and lower-order-final behavior. A terminal zero is
evaluated through the same `0.001` sigma limit used by ComfyUI. Newer flow and
alternate `bh2` modes remain distinct future extensions. See the [official
UniPC implementation][unipc-code] and [paper][unipc-paper].

### LCM

LCM sampling is intended for a Latent Consistency Model or a compatible
LCM-LoRA/distillation. Each step takes the model's denoised estimate and, unless
it is the last step, re-noises it at the next sigma. Its value is few-step
inference; selecting it for an ordinary non-distilled checkpoint does not turn
that checkpoint into an LCM. See the [Latent Consistency Models paper][lcm-paper].

### DEIS and IPNDM

Both are multistep ODE methods associated with exponential-integrator diffusion
sampling. DEIS builds variable coefficients for the selected sigma grid and
combines derivative history; the native path uses the current Diffusers
analytic variable-grid log-rho integrals and caps the order at three, avoiding
ComfyUI's tabulated 10,000-substep coefficient quadrature. IPNDM uses the
familiar fixed-step Adams–Bashforth coefficients and ramps through orders one,
two, three, and four as history is collected. Both are now implemented in the
native core. See the [DEIS paper][deis-paper] and the [official diff-sampler
method table][diff-sampler].

### RES Multistep

The concrete ComfyUI/native algorithm is a history-based exponential update. It
uses `phi_1` and `phi_2` functions to weight the current and previous denoised
predictions, with an Euler startup step before history exists. The non-ancestral
UI option fixes `eta=0`; ancestral and CFG++ variants in ComfyUI are separate
methods and are not exposed here. The implementation reference is
[ComfyUI's sampling source][comfy-kdiffusion].

### DDIM

DDIM defines a non-Markovian implicit sampling process that can be deterministic
when `eta=0`. “DDIM sampler” and “DDIM Uniform scheduler” are independent
concepts: the former is the update, while the latter chooses uniformly strided
training timesteps. This fork currently exposes a `ddim_trailing` compatibility
method whose update is Euler ancestral with `eta=0`. The separately exposed
`ddim_uniform` scheduler now supplies the requested training-table spacing. See
the [DDIM paper][ddim-paper].

### Gradient Estimation

The Gradient Estimation sampler interprets diffusion sampling through an
optimization lens and improves the Euler direction using a previous-gradient
estimate. In this core, the first step is Euler and later steps use
`gamma*d_current + (1-gamma)*d_previous`, with `gamma=2` by default. The native
engine can parse a different `gamma` through extra sampler arguments, but the
Easy Diffusion request bridge does not currently expose that tuning field. The
plain method is implemented; the separately named CFG++ variant from the matrix
is not. See the [official implementation][ge-code] and [ICML paper][ge-paper].

### ER-SDE

ER-SDE-Solver integrates an extended reverse-time SDE and provides a
convergence-order construction for stochastic diffusion sampling. This fork
uses the VP ER-SDE stage-three form, retains up to two denoised-history terms,
and injects scaled seeded noise. The authors report the method as a roughly
20-evaluation high-quality sampler; exact behavior still depends on prediction
type and alpha/noise schedules. See the [official implementation][er-sde-code]
and [paper][er-sde-paper].

## Requested schedulers

Schedulers below are deterministic: “stochastic” behavior comes from the
sampler update, not from construction of the sigma list.

| UI option | Native identifier | Sigma placement | Native status |
|---|---|---|---|
| Normal | `discrete` (`normal` accepted as an alias) | Evenly spaced continuous training timesteps from high to low noise, plus terminal zero | Available; native UI/API routing added in this audit |
| Karras | `karras` | EDM power-law ramp in sigma with `rho=7` | Available |
| Exponential | `exponential` | Uniform spacing in `log(sigma)` | Available |
| SGM Uniform | `sgm_uniform` | Uniform model-time grid using `steps+1` points, dropping the last before appending zero | Available |
| Simple | `simple` | Fixed strides through the stored 1000-step training sigma table | Available |
| DDIM Uniform | `ddim_uniform` | Fixed integer stride through the training sigma table, then reversed into descending-noise order | Available; native scheduler and UI/API routing added |
| Beta | `beta` | Timesteps from the inverse Beta CDF; default `alpha=beta=0.6` | Available; native UI/API routing added in this audit |
| Linear Quadratic | `linear_quadratic` | First half changes linearly, second half quadratically, ending at zero and scaled by the active `sigma_max` | Available as a general scheduler; kept separate from model-specific `mochi` |
| KL Optimal | `kl_optimal` | Linear interpolation in noise angle `atan(sigma)`, mapped back with `tan` | Available; native UI/API routing added in this audit |

### Normal

Normal samples evenly in the model's continuous training-time coordinate and
converts each point through `t_to_sigma`. The native public name is `discrete`,
but `normal` is accepted as an alias. In this wrapper, the older `Uniform` UI
option also maps to `discrete`, so Normal and Uniform currently produce the same
native schedule.

### Karras

Karras uses the EDM ramp
`sigma_i = (sigma_max^(1/rho) + u_i*(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`
with `rho=7` here. It concentrates steps nonlinearly over noise scale and is a
common match for DPM++ multistep methods. See the [EDM paper][edm-paper].

### Exponential

Exponential linearly spaces `log(sigma)` between the model's maximum and minimum
positive sigmas, exponentiates the points, and appends zero. Ratios between
adjacent positive sigmas are therefore constant.

### SGM Uniform

SGM Uniform takes `steps+1` uniformly spaced model-time points from training
timestep 999 to zero, converts the first `steps`, and appends terminal sigma
zero. The endpoint convention differs slightly from Normal and is important at
small step counts.

### Simple

Simple indexes backward through the 1000-entry training schedule with stride
`1000 / steps`, using integer-truncated offsets, then appends zero. It is cheap
and direct but its exact grid depends on the stored training schedule rather
than a continuous sigma formula.

### DDIM Uniform

ComfyUI's DDIM Uniform scheduler walks forward through the stored sigma table
with integer stride `max(table_length // steps, 1)`, then reverses the collected
values so sampling proceeds from high to low noise. It handles a table that
already starts near zero specially. The native implementation preserves that
index-one/terminal-zero convention and is not an alias for the native
`ddim_trailing` sampler. See
[ComfyUI's scheduler source][comfy-schedulers].

### Beta

Beta maps uniformly spaced probabilities through the inverse Beta distribution
CDF, scales them to the training timestep range, rounds to discrete indices,
removes adjacent duplicates, and appends zero. Defaults are `alpha=0.6` and
`beta=0.6`; the core can parse different positive values from extra sampler
arguments, but the current UI does not expose them. See the
[Beta Sampling paper][beta-paper].

### Linear Quadratic

The Genmo/ComfyUI recipe spends the first half of its steps in a linear segment
up to a `0.025` threshold and the remainder on a quadratic segment, reverses the
result to descending noise, and scales it by the model's maximum sigma. This
fork keeps `mochi` as its normalized model-specific name and exposes the
sigma-max-scaled form independently as `linear_quadratic`. See the [Genmo
implementation][genmo-schedule] and
[ComfyUI's scheduler source][comfy-schedulers].

### KL Optimal

KL Optimal spaces an angle linearly between `atan(sigma_max)` and
`atan(sigma_min)`, applies `tan`, and appends zero. In geometric terms it is
uniform in noise angle rather than sigma or log sigma. The local implementation
matches [ComfyUI's scheduler source][comfy-schedulers].

## UI recommendations and remaining variant gap

The UI can mark only recommendations whose scheduler option actually exists.
After this audit it marks these native combinations:

- Euler + Normal
- Euler a + Exponential
- DPM++ 2M + Karras
- DPM++ 2M SDE + Karras
- DPM++ 3M SDE + Linear Quadratic
- UniPC + KL Optimal
- LCM + SGM Uniform
- DEIS + Simple
- IPNDM + DDIM Uniform
- RES Multistep + Karras
- ER-SDE + Exponential

No star is shown for plain Heun, DDIM, or plain Gradient Estimation because the
provided matrix does not give those exact sampler variants a best pairing. The
only remaining named matrix pairing that cannot be selected end-to-end natively
is **Gradient Estimation CFG++ + Beta**. The native `gradient_estimation` option
is the plain method; it must not be relabeled as its separate CFG++ variant.

## Validation and follow-up variants

The five formerly missing native paths now have compiled numerical regression
coverage in [`sampler_scheduler_test.cpp`][native-numerical-test]:

- DDIM Uniform and Linear Quadratic sigma vectors are checked against explicit
  reference values.
- DPM++ 3M SDE is checked for constant-data behavior, fixed-seed Brownian
  reproducibility, and seed sensitivity.
- UniPC and DEIS exercise their high-order paths and constant-data solutions.

Useful follow-up work is image-level golden regression across SD 1.x/SDXL,
alternate UniPC `bh2`, and explicit flow-prediction versions of UniPC/DEIS.
Those are extensions rather than gaps in the ordinary image sigma paths added
here.

[compatibility-matrix]: https://comfyui.dev/docs/guides/Other%20Resources/sampler-and-scheduler-compatibility-matrix/
[native-api]: ../source/sdkit3-port-source/stable-diffusion.cpp/include/stable-diffusion.h
[native-denoiser]: ../source/sdkit3-port-source/stable-diffusion.cpp/src/runtime/denoiser.hpp
[native-server]: ../source/sdkit3-port-source/src/server.cpp
[native-numerical-test]: ../source/sdkit3-port-source/stable-diffusion.cpp/tests/sampler_scheduler_test.cpp
[image-settings]: ../ui/plugins/ui/image_plugin/image-settings.plugin.html
[comfy-kdiffusion]: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/k_diffusion/sampling.py
[comfy-schedulers]: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/samplers.py
[edm-paper]: https://arxiv.org/abs/2206.00364
[dpmpp-paper]: https://arxiv.org/abs/2211.01095
[unipc-code]: https://github.com/wl-zhao/UniPC
[unipc-paper]: https://arxiv.org/abs/2302.04867
[lcm-paper]: https://arxiv.org/abs/2310.04378
[deis-paper]: https://arxiv.org/abs/2204.13902
[diff-sampler]: https://github.com/zju-pi/diff-sampler
[ddim-paper]: https://arxiv.org/abs/2010.02502
[ge-code]: https://github.com/ToyotaResearchInstitute/gradient-estimation-sampler
[ge-paper]: https://openreview.net/forum?id=o2ND9v0CeK
[er-sde-code]: https://github.com/QinpengCui/ER-SDE-Solver
[er-sde-paper]: https://arxiv.org/abs/2309.06169
[beta-paper]: https://arxiv.org/abs/2407.12173
[genmo-schedule]: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py
[ays-paper]: https://arxiv.org/abs/2404.14507
[diffusers-plus-plus]: https://github.com/ModelsLab/diffusers_plus_plus
[diffusers-plus-plus-commit]: https://github.com/ModelsLab/diffusers_plus_plus/commit/d1fd977c5ac0cef08c6f965213c20f4460f6c37a
[dpp-unipc]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_unipc_multistep.py
[dpp-deis]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_deis_multistep.py
[dpp-ipndm]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_ipndm.py
[dpp-dpm]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
[dpp-sa-solver]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_sasolver.py
[dpp-cosine-dpm]: https://github.com/ModelsLab/diffusers_plus_plus/blob/d1fd977c5ac0cef08c6f965213c20f4460f6c37a/src/diffusers/schedulers/scheduling_cosine_dpmsolver_multistep.py
[hf-diffusers]: https://github.com/huggingface/diffusers
[hf-diffusers-commit]: https://github.com/huggingface/diffusers/commit/c419dac0152186060246c93a095bc1bfaea342b3
[hf-dpm]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
[hf-unipc]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/schedulers/scheduling_unipc_multistep.py
[hf-deis]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/schedulers/scheduling_deis_multistep.py
[hf-ipndm]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/schedulers/scheduling_ipndm.py
[hf-ddim]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/schedulers/scheduling_ddim.py
[hf-scheduler-guide]: https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/docs/source/en/api/schedulers/overview.md
