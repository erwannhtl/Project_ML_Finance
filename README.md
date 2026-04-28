# Deep Learning Volatility

*A review of the paper by Bayer, Horvath, Muguruza, Stemper, and Tomas*

**Authors:** Erwann Hotellier · Capucine Rousset · Taddéo Vivet — April 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Technical Summary](#3-technical-summary)
4. [Numerical Illustration](#4-numerical-illustration)
5. [Limitations and Possible Improvements](#5-limitations-and-possible-improvements)
6. [Appendix](#appendix)

---

## 1. Introduction

In financial mathematics, one of the most important tasks is model calibration: finding the parameters of a mathematical model so that the prices it produces match the prices observed in real markets. For simple models, such as the famous Black-Scholes model, this task is fast because an exact pricing formula exists. However, more realistic models, such as the family of *rough volatility models*, require slow numerical simulation methods to compute prices, which makes calibration far too slow for practical use in the financial industry.

This report presents and discusses the paper *"Deep Learning Volatility"* by Horvath, Muguruza, and Tomas (2019). Their key idea is to train a neural network offline to act as a fast pricing calculator, and then use this learned approximation to perform calibration very efficiently. This two-step approach combines the theoretical accuracy of rough volatility models with the computational speed required for real-world applications.

---

## 2. Literature Review

### 2.1 Historical Context and Prior Work

Bayer, Friz, and Gatheral (2015) introduced the **rough Bergomi model**, where the variance process $V_t$ is driven by a fractional Brownian motion:

$$V_t = \xi_0(t)\,\mathcal{E}\!\left(\nu\sqrt{2H}\int_0^t(t-s)^{H-1/2}dZ_s\right)$$

where $\xi_0(t)$ is the forward variance curve and $H < 1/2$ controls the roughness. This model accurately captures the *volatility skew*, but does not admit a closed-form pricing formula — creating a **calibration bottleneck**.

Hutchinson, Lo, and Poggio (1994) pioneered neural networks for option pricing:

$$\frac{C}{K} = f_{\mathrm{NN}}\!\left(\frac{S}{K},\, T;\, w\right)$$

Hernandez (2017) proposed direct calibration inversion via $W : \mathbb{R}^N \rightarrow S \subseteq \mathbb{R}^n$, but this is mathematically unstable.

### 2.2 Contribution of the Paper

Horvath, Muguruza, and Tomas introduce a two-step **Deep Calibration** framework:

1. **Offline**: Train a neural network to learn the *forward* pricing map (parameters → prices).
2. **Online**: Solve calibration as a deterministic optimisation using the trained network as surrogate.

Key innovation: the network outputs a complete $8 \times 11$ implied volatility surface simultaneously (88 points), treating it like an image to ensure structural consistency and uniqueness.

---

## 3. Technical Summary

### 3.1 Mathematical Framework

$$\tilde{F}(\Theta, \zeta) \approx \tilde{P}(\mathcal{M}(\Theta, \zeta))$$

$$F^*(\theta) = \{\sigma_{BS}^{\mathcal{M}(\theta)}(T_i, k_j)\}_{i=1,\, j=1}^{n,\, m}$$

### 3.2 Models

**Rough Bergomi** — 11 parameters (piecewise $\xi_0$, $H$, $\nu$, $\rho$)

**1-Factor Bergomi** — $\theta = (\xi_0, \beta, \eta, \rho)$:
$$V_t = \xi_0(t)\,\mathcal{E}\!\left(\eta\int_0^t \exp(-\beta(t-s))\,dZ_s\right)$$

**Heston** — $\theta = (a, b, v, \rho)$:
$$dV_t = a(b - V_t)\,dt + v\sqrt{V_t}\,dZ_t$$

### 3.3 Network Architecture and Offline Training

- **Architecture**: input → 4 × 30 hidden layers (ELU) → 88 outputs → **5,878 parameters**
- **ELU activation**:

$$\sigma_{\text{ELU}}(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$$

- **Training data**: 68,000 synthetic samples (60,000-path Monte Carlo), 12,000 test
- **Loss**:

$$\hat{w} = \arg\min_{w} \sum_{u=1}^{N_{\text{Train}}} \sum_{i=1}^{n} \sum_{j=1}^{m} \bigl(F(\theta_u, w)_{ij} - F^*(\theta_u)_{ij}\bigr)^2$$

### 3.4 Online Calibration

$$\hat{\theta} = \arg\min_{\theta \in \Theta} \sum_{i=1}^{n} \sum_{j=1}^{m} \bigl(\tilde{F}(\theta)_{ij} - \sigma_{BS}^{MKT}(T_i, k_j)\bigr)^2$$

Primary optimiser: **Levenberg-Marquardt** (exact gradients via ELU). Benchmarked against L-BFGS-B, SLSQP, BFGS, COBYLA, Differential Evolution, Nelder-Mead.

### 3.5 Empirical Performance

| Metric | Neural Network | Monte Carlo |
|---|---|---|
| Avg. relative error | < 0.5% | — |
| Surface evaluation time | ~31 µs | ~500,000 µs |
| **Speedup factor** | **9,000–16,000×** | baseline |

---

## 4. Numerical Illustration

Implementation uses the rough Bergomi model with piecewise forward variance, calibrated on **EUR/USD option data** (2021-06-08 to 2026-04-13), extending the authors' [NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations) repository.

### 4.1 Synthetic Dataset

$N = 60{,}000$ configurations sampled uniformly:

| Parameter | Distribution |
|---|---|
| $H$ | $\mathcal{U}(0.025,\ 0.5)$ |
| $\nu$ | $\mathcal{U}(0.3,\ 2.5)$ |
| $\rho$ | $\mathcal{U}(-0.8,\ 0.8)$ |
| $\xi_i$ (×9) | $\mathcal{U}(0.001,\ 0.010)$ |

**Maturity grid** (9 tenors):
$$T \in \{0.019,\ 0.038,\ 0.057,\ 0.082,\ 0.164,\ 0.246,\ 0.493,\ 0.739,\ 1.000\}$$
(1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y)

**Adaptive strike grid**:
$$K_j = S_0 \cdot \exp\!\left(z_j \cdot \sigma_{\text{ref}} \cdot \sqrt{T}\right), \quad z_j \in \{-2.5,\ldots,2.5\}, \quad \sigma_{\text{ref}} = 0.08$$

> Monte Carlo: 10,000 paths, 500 steps/year.

### 4.2 Network and Training

| Component | Spec |
|---|---|
| Input | 12 nodes |
| Hidden | 4 × 30, ELU |
| Output | 99 nodes ($9 \times 11$) |
| Parameters | **6,249** |
| Optimiser | Adam, batch=32 |
| Epochs | 200 max, early stopping (patience=25) |
| Split | 85/15 (51,000/9,000) |

**Input normalisation**:
$$\tilde{x}_i = \frac{x_i - \tfrac{1}{2}(u_i + l_i)}{\tfrac{1}{2}(u_i - l_i)}$$

**Custom maturity-weighted loss**:
$$w_i = \begin{cases} 2 - T_i & T_i < 1 \\ 1 & T_i \geq 1 \end{cases}$$

$$\hat{w} = \arg\min_{w} \sum_u \sum_i \sum_j w_i \cdot \bigl(F(\theta_u, w)_{ij} - F^*(\theta_u)_{ij}\bigr)^2$$

Short tenors (1W resolved by ~10 steps) receive higher weights to counteract coarse-discretisation noise.

### 4.3 Validation on Synthetic Data

$$\hat{\theta} = \arg\min_{\theta \in \Theta} \sum_{i,j} \bigl(\tilde{F}(\theta)_{ij} - \sigma_{BS}^{\mathrm{target}}(T_i, k_j)\bigr)^2$$

![NN prediction vs Monte Carlo](figures/test_image_predict.png)
*Out-of-sample prediction #113: $H=0.1875$, $\nu=1.9260$, $\rho=0.2347$*

### 4.4 Calibration on CME EUR/USD Data

- 5 delta levels × 9 tenors = 45 points/day
- Delta-to-moneyness via **Garman-Kohlhagen**:

$$k = \ln\!\left(\frac{K}{F}\right) = -\sigma\sqrt{T}\,\Phi^{-1}(\Delta) + \frac{1}{2}\sigma^2 T$$

- Interpolated onto 99-point grid; binary mask $w \in \{0,1\}^{99}$ excludes extrapolated points
- **Warm-start**: previous day's parameters as initial guess

![EUR/USD calibration](figures/Test_fx_good_day.png)
*14 February 2025: $H=0.1563$, $\nu=1.6757$, $\rho=-0.1599$*

![Calibration error over time](figures/evolution_timeline.png)
*RMSE and MAE over the full sample period*

Error stays **below 0.50%** except during two stress episodes:
- **2022–2023**: Fed tightening, Ukraine war, EUR/USD below parity
- **Early 2025**: US tariff shock (2 Apr 2025), VIX > 50, disorderly FX repricing

---

## 5. Limitations and Possible Improvements

**1. Training data quality** — The network inherits Monte Carlo bias. *Fix*: variance reduction techniques (antithetic sampling, control variates).

**2. Fixed grid** — Off-grid options require separate interpolation. *Fix*: arbitrage-free interpolation post-processing.

**3. No uncertainty quantification** — No confidence bounds on calibrated parameters. *Fix*: Bayesian neural networks or conformal prediction.

---

## Appendix

### A. Figures

| File | Description |
|---|---|
| `figures/test_image_predict.png` | NN vs MC ground truth |
| `figures/Test_fx_good_day.png` | EUR/USD calibration (14 Feb 2025) |
| `figures/evolution_timeline.png` | Calibration error 2021–2026 |
| `figures/surface_vol.png` | Volatility surface visualisation |
| `figures/rmse_mat_fx.png` | Custom loss function impact |

### B. Delta Quoting and the Adaptive Strike Grid

From the Black-Scholes delta:
$$\Delta = \Phi\!\left( \frac{-k + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}} \right)$$

Inverting gives:
$$k = -\sigma\sqrt{T}\,\Phi^{-1}(\Delta) + \frac{1}{2}\sigma^2 T$$

Setting $z = -\Phi^{-1}(\Delta)$ recovers our synthetic grid:
$$k \approx z \cdot \sigma \cdot \sqrt{T}$$

FX delta-quoted data therefore naturally enforces the same adaptive structure as our training grid — guaranteeing the network is never asked to interpolate in illiquid regions.

---

*Based on: Horvath, Muguruza, and Tomas (2019), "Deep Learning Volatility". Code adapted from [NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations).*
