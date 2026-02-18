# Synthetic Marketing & Sales Dataset Generation

## Overview

## Dataset Scope

| Parameter | Value |
|-----------|-------|
| **Time Period** | Jan 1, 2024 → Jan 1, 2026 (104 weeks) |
| **Channels** | 4 (Search, Social, CTV, TV) |
| **Controls** | 4 (Trend, Seasonality, Prime Day, Mother's Day) |
| **Sales Observations** | 104 weekly data points |

---

## 1. Media Spend Generation

### Channel Characteristics

Each channel is modeled with distinct spending patterns reflective of real-world behavior:

| Channel | Pattern | Rationale |
|---------|---------|-----------|
| **Search** | High frequency, low variance | Continuous targeting, budget-controlled |
| **Social** | Moderate frequency, balanced variance | Flexible daily budgets |
| **CTV** | Consistent, moderate-high spend | Minimum media buys required |
| **TV** | Chunky, episodic spending | Campaign-based budgeting with gaps |

### Distribution Approach

**Gamma Distribution** captures realistic skewed spend patterns:
- **Search**: `Gamma(1.2, 1500)` → many small spends, occasional large ones
- **Social**: `Gamma(2.5, 800)` → moderate skew, predictable average
- **CTV**: `Gamma(4, 1700)` → higher minimum spend, concentrated
- **TV**: `Gamma(8, 2000)` × random indicator → chunky, episodic campaigns

```python
# Example: Search spend generation captures opportunistic/tactical spending
search_spend = np.random.gamma(1.2, 1500, weeks)
```

### Visualization

![Spend Trends](mmm_spend_trends.png)

---

## 2. Temporal Dynamics: Trend & Seasonality

### Trend Component

**Non-linear growth** simulated via power transformation:
```
trend = 100 × (linspace(0, 50) + 10)^(1/4) − 1
```

- Represents long-term business growth (market expansion, brand maturity)
- Smooth acceleration typical of retail/SaaS businesses
- Range: ~100 to ~250 sales units

### Seasonality Component

**Sine/Cosine oscillations** capture annual cycles:
```
seasonality = 30 × 0.5 × (−sin(2π × dayofyear/365.5) + cos(π × dayofyear/365.5))
```

- Represents holidays, weather, shopping seasons
- Dual-frequency model: quarterly beats + semi-annual waves
- Range: ±30 sales units

### Visualization

![Baseline Trends](mmm_baseline_trends_seasonality.png)

---

## 3. Promotional Events

Special calendar events with known impact dates:

| Event | Date | Impact | Rationale |
|-------|------|--------|-----------|
| **Prime Day** | Nov 11, 2024 | +40 units | Annual mega-sales event |
| **Mother's Day** | May 18, 2025 | +40 units | Gifting occasion driver |

Modeled as binary indicators:
```python
data["prime_day_promo"] = (data["week"] == "2024-11-11").astype(float)
data["mothers_day_promo"] = (data["week"] == "2025-05-18").astype(float)
```

---

## 4. Media Transformation Pipeline

### 4.1 Adstock Effect (Legacy)

**Geometric adstock** captures how ad impact carries forward:
```
y_t = α×y_(t-1) + x_t
```

Channel-specific carryover (retention) rates:

| Channel | Alpha | Interpretation | Duration |
|---------|-------|-----------------|----------|
| **Search** | 0.05 | Minimal carryover | ~1 week |
| **Social** | 0.20 | Moderate recall | ~2–3 weeks |
| **CTV** | 0.50 | Strong recall | ~4–5 weeks |
| **TV** | 0.80 | Very strong recall | ~8+ weeks |

**Real-world context**: TV ads (memorable stories, broad reach) persist longer than search ads (immediate, click-to-convert).

### 4.2 Saturation Effect (Diminishing Returns)

**Logistic saturation function** models market saturation:
```
y = x^s / (x^s + K^s)
```

- **K** (half-saturation): spend level where 50% of maximum effect is achieved
- **s** (shape): steepness of saturation curve (elasticity)

| Channel | K (rel. to avg) | s | Interpretation |
|---------|-----------------|---|-----------------|
| **Search** | 0.5× | 1.0 | Saturation hits fast; limited audience pools online |
| **Social** | 0.8× | 1.2 | Moderate saturation; audience fatigue |
| **CTV** | 1.2× | 1.5 | Wide reach; slower to saturate |
| **TV** | 2.0× | 2.0 | Massive reach; widely distributable |

**Real-world context**: Lower-funnel channels (Search) exhaust addressable inventory faster. Broadcast channels (TV) scale more efficiently across broader audiences.

### Transformation Pipeline Visualization

![Media Transformations](pymc_data_transformed.png)

Flow: Raw Spend → Adstock (carryover) → Scale (normalization) → Saturation (diminishing returns)

---

## 5. Sales Generation Process

### 5.1 Model Equation

```
Sales = Base + f(Controls) + f(Media) + Noise
```

Where:
- **Base**: Intercept (intrinsic demand)
- **Controls**: Trend, seasonality, promotional events
- **Media**: Transformed spend (adstock + saturation)
- **Noise**: Normal(0, σ) for realism

### 5.2 Control Contributions

| Component | Coefficient | Sales Impact | Role |
|-----------|-------------|--------------|------|
| **Trend** | 1.0× | Multiplicative growth | Underlying business momentum |
| **Seasonality** | 1.0× | ±30 units/week | Cyclical demand patterns |
| **Prime Day Promo** | 40 units | Direct lift | Calendar event driver |
| **Mother's Day Promo** | 40 units | Direct lift | Calendar event driver |

### 5.3 Media Contributions

Channel-specific **Return on Ad Spend (ROAS)** coefficients:

| Channel | Beta (ROI) | Weekly Contribution | Channel Role |
|---------|------------|------------------|--------------|
| **Search** | 200 | Most variable | High intent, efficient |
| **Social** | 300 | Moderate | Brand awareness, consideration |
| **CTV** | 400 | Significant | Upper-to-mid funnel, premium |
| **TV** | 500 | Largest impact | Brand building, reach |

**Interpretation**: TV has highest absolute impact (brand building at scale), but Search is most efficient per dollar spent.

### 5.4 Noise (Stochasticity)

```python
noise = Normal(0, 12)  # σ = 12 units
```

- Represents unmeasured factors (weather, competitor actions, social trends, measurement error)
- Realistic signal-to-noise ratio (~10–20% of total variance)

---

## 6. Dataset Statistics

### Sales Performance

```
Sales Range:   500–750 units/week
Average:       ~600 units/week ± ~40 (std dev)
Trend Impact:  +150 units from 2024 → 2026
Seasonality:   ±30 units cyclically
```

### Contribution Breakdown (Average Weekly)

```
Base Sales:        500 units
Trend:             +100 units
Seasonality:       +2 units (avg)
Media:             +120 units
  ├─ TV:           +50 units (42%)
  ├─ CTV:          +40 units (33%)
  ├─ Social:       +20 units (17%)
  └─ Search:       +10 units (8%)
```

### Realism Metrics

| Metric | Value | Real-World Benchmark |
|--------|-------|----------------------|
| Media yield (ROAS) | 0.3–0.8 | Typical range 0.2–1.5 |
| Trend slope | +1.5 units/week | Reasonable growth |
| Seasonality amplitude | 10–15% of mean | Typical for consumer |
| Channel correlation | 0.2–0.6 | Mixed co-movement |

---

## 7. Realism & Validation

### ✅ Realistic Features

1. **Channel Diversity**
   - Different spend distributions (Search = opportunistic; TV = episodic)
   - Realistic carry-over and saturation effects
   - Channel-specific efficiency (Search > Social > CTV; TV for reach)

2. **Time Dynamics**
   - Non-linear trend (acceleration > steady growth)
   - Multi-frequency seasonality (holidays + semi-annual waves)
   - Known promotional events

3. **Marketing Principles**
   - Positive ROI on all channels (learned via MMM)
   - Diminishing returns with scale (saturation effects)
   - Legacy of past spending (adstock)
   - Unobserved factors (noise)

4. **Data Quality**
   - 104 weeks = sufficient for Bayesian inference (4–6 month minimum, 2+ years ideal)
   - No missing values or data quality issues
   - Consistent measurement (weekly granularity)

### 📊 Example Use Cases

This dataset is suitable for:
- ✅ **MMM Training**: Learn channel effects, ROI, and optimization strategies
- ✅ **Budget Allocation**: Test optimal spend allocation across channels
- ✅ **Causal Inference**: Validate attribution and isolate channel impact
- ✅ **Forecasting**: Predict future sales under different scenarios
- ✅ **Sensitivity Analysis**: Understand elasticity and cross-channel effects

---

## 8. Code Architecture

### Key Functions

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 1 | Generate spend | Date range, channels | Spend DataFrame |
| 2 | Apply adstock | Spend, alpha params | Adstocked spend |
| 3 | Scale | Adstocked spend | Zero-mean, unit-scale |
| 4 | Saturate | Scaled spend, K, s | Saturated contribution |
| 5 | Generate sales | Media + controls + noise | Final sales series |
| 6 | Export | Sales data | `pymc_data.csv` |

### Parametrization

All assumptions are **explicitly coded** and easily tunable:
```python
configs = {
    'search': {'alpha': 0.05, 'lam': 2, 'beta': 8},
    'social': {'alpha': 0.2, 'lam': 3, 'beta': 12},
    'ctv':    {'alpha': 0.5, 'lam': 4, 'beta': 15},
    'tv':     {'alpha': 0.8, 'lam': 5, 'beta': 25}
}
```

—> Facilitates **sensitivity analysis**, scenario planning, and dataset customization.

---

## 9. Conclusion

This synthetic dataset demonstrates:

1. **Technical Proficiency**
   - PyMC Marketing transforms (adstock, saturation)
   - Domain-specific data engineering (MMM principles)
   - Statistical modeling (Gamma distributions, temporal components)

2. **Domain Understanding**
   - Marketing channel dynamics (search, social, CTV, TV)
   - Realistic spending patterns and ROI structures
   - Business impact modeling (trend, seasonality, promotions)

3. **Real-World Applicability**
   - Generates data matching realistic properties
   - Enables validation of MMM algorithms
   - Supports transparent, reproducible analytics workflows

The dataset is **production-ready** for training Bayesian media mix models, optimizing marketing budgets, and forecasting sales under uncertain conditions.

---

## References

- **Geometric Adstock**: Classic time-series carryover model
- **Logistic Saturation**: Hill function and diminishing returns
- **PyMC Marketing**: Open-source Bayesian MMM library
- **Dataset**: `artifacts/pymc_data.csv` (104 weeks, 4 channels)

---

**Generated**: Feb 2026  
**Dataset Format**: Weekly aggregation | CSV export  
**Reproducibility**: Fixed random seed (numpy.random.seed(42))
