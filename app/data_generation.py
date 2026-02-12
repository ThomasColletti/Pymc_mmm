import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

# 1. Data Generation

# 1.1 Channel Understanding
# Channels have different dynamics
# TV: Low frequency but high reach with long carryover. The TV adstock tend to decay slow as TV ads are more memorable. Think of events like super bowl and Grammy.
# Search: targeted with short carryover, high intent, decay fast but more efficient). Low reach as we can quickly run out of people to target.
# Social: mid carryover, moderate efficiency
# CTV(Connected TV): Unskippable, targeting household with higher carryover than social but lower than linear TV.

# 1.2 Assumptions for the data.
# Here I create a model for weekly estimates.
# For each week:
# Media: spend_*, imps_* (if measurable)
# Funnel: Early stage building brand awareness and consideration -> performance marketing drive more direct conversions at lower funnel.
# Controls: price, promo, seasonality, competitor, demand, marco_economics events impact (i.e. covid-19 increasing in food delivery services ) etc.

# 1.3 Setting Parameters
# Adstock/Retention (Alpha): This represents the percentage of the ad's impact that carries over to the next week.
# alpha_linear_tv = 0.8 # Lasts a long time
# alpha_ctv = 0.6 # Less memorable than TV
# alpha_social = 0.4 # Moderate
# alpha_search = 0.05 # Immediate impact only

# Saturation (Hill Function K and s)
# Here I use Hill function to mimic saturation:
# f(x) = x^s / (x^s + K^s)
# K is the 'half-saturation' point. s is the parameter for the function.
# - TV: assuming K_tv = 2 * avg spend; S_tv = 2 - wide reach, hard to saturate
# - Search: assuming K_search = 0.5 * avg sepnd; S_search = 1 - lower funnel click to conversion, limited pool of targets quickly run out of options to target
# - Social: assuming K_social = 0.8 * avg spend; S_social = 1.2 - relatively easier to saturate comparing to CTV/TV but there is a fatigue for seeing too much the same ads.
# - CTV: assuming K_ctv = 1.2 * avg spend; S_ctv = 1.5 - Wider reach, can scale significantly

np.random.seed(42)

# 1. Date Range
min_date = pd.to_datetime("2024-01-01")
max_date = pd.to_datetime("2026-01-01")
weeks = len(pd.date_range(start=min_date, end=max_date, freq="W-MON"))
print(weeks)

# 2. Media Spend
data = pd.DataFrame({
    'week': pd.date_range(start=min_date, end=max_date, freq="W-MON"),
    'search_spend': np.random.gamma(1.2, 1500, weeks), # Low alpha = high skew (many tiny spends)
    'social_spend': np.random.gamma(2.5, 800, weeks), # Moderate skew
    'ctv_spend': np.random.gamma(4, 1700, weeks),# High min spend, higher average
    'tv_spend': np.random.gamma(8, 2000, weeks) * (np.random.rand(weeks) > 0.6) # "Chunky" spend, low variance relative to mean
}).assign(
    year=lambda x: x["week"].dt.year,
    month=lambda x: x["week"].dt.month,
    dayofyear=lambda x: x["week"].dt.dayofyear,
)

# Viz
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
channels = ['search_spend', 'social_spend', 'ctv_spend', 'tv_spend']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, col in enumerate(channels):
    axes[i].plot(data['week'], data[col], color=colors[i], lw=2)
    axes[i].set_title(f'Weekly Spend: {col.replace("_", " ").title()}', loc='left', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Spend ($)')
plt.tight_layout()
plt.savefig('mmm_spend_trends.png')
plt.show()

# 4. Underlying Trend and Seasonal Components
data["trend"] = 100 * (np.linspace(start=0.0, stop=50, num=weeks) + 10) ** (1 / 4) - 1
data["cs"] = -np.sin(2 * 2 * np.pi * data["dayofyear"] / 365.5)
data["cc"] = np.cos(1 * 2 * np.pi * data["dayofyear"] / 365.5)
data["seasonality"] = 30* 0.5 * (data["cs"] + data["cc"])

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(data["week"], data["trend"], label="trend", color="C2")
ax.plot(data["week"], data["seasonality"], label="seasonality", color="C3")

ax.legend(loc="upper left")
ax.set_title("Trend & Seasonality Components")
ax.set_xlabel("date")
ax.set_ylabel("")
plt.savefig('mmm_baseline_trends_seasonality.png')
plt.show()

# 5. Special event
data["prime_day_promo"] = (data["week"] == "2024-11-11").astype(float)
data["mothers_day_promo"] = (data["week"] == "2025-05-18").astype(float)

# 6. Spend Transformation
# Assumptions on the parameters and coefficients (The Ground Truth ROAs)
configs = {
    'search': {'alpha': 0.05, 'lam':2, 'beta': 8},
    'social': {'alpha': 0.2, 'lam':3, 'beta': 12},
    'ctv':    {'alpha': 0.5, 'lam':4, 'beta': 15},
    'tv':     {'alpha': 0.8, 'lam':5, 'beta': 25}
}

# spend → adstock → scale → saturation
data["search_adstock"] = geometric_adstock(x=data["search_spend"].to_numpy(), alpha=configs['search']['alpha'], l_max=8, normalize=True).eval().flatten()
data["social_adstock"] = geometric_adstock(x=data["social_spend"].to_numpy(), alpha=configs['social']['alpha'], l_max=8, normalize=True).eval().flatten()
data["ctv_adstock"] = geometric_adstock(x=data["ctv_spend"].to_numpy(), alpha=configs['ctv']['alpha'], l_max=8, normalize=True).eval().flatten()
data["tv_adstock"] = geometric_adstock(x=data["tv_spend"].to_numpy(), alpha=configs['tv']['alpha'], l_max=8, normalize=True).eval().flatten()

channels = ["search", "social", "ctv", "tv"]
for ch in channels:
    col = f"{ch}_adstock"
    scale = data[col].mean()
    data[f"{ch}_adstock_scaled"] = data[col] / scale

data["search_adstock_saturated"] = logistic_saturation(x=data["search_adstock_scaled"].to_numpy(), lam=configs['search']['lam']).eval()
data["social_adstock_saturated"] = logistic_saturation(x=data["social_adstock_scaled"].to_numpy(), lam=configs['social']['lam']).eval()
data["ctv_adstock_saturated"] = logistic_saturation(x=data["ctv_adstock_scaled"].to_numpy(), lam=configs['ctv']['lam']).eval()
data["tv_adstock_saturated"] = logistic_saturation(x=data["tv_adstock_scaled"].to_numpy(), lam=configs['tv']['lam']).eval()

# Viz
fig, ax = plt.subplots(
    nrows=4, ncols=4, figsize=(16, 9),
    sharex=True, sharey=False,
    layout="constrained"
)

# Row 1 — raw spends
ax[0, 0].plot(data["week"], data["search_spend"], color="C0")
ax[0, 0].set_ylabel("search spend")
ax[0, 1].plot(data["week"], data["social_spend"], color="C1")
ax[0, 1].set_ylabel("social spend")
ax[0, 2].plot(data["week"], data["ctv_spend"], color="C2")
ax[0, 2].set_ylabel("ctv spend")
ax[0, 3].plot(data["week"], data["tv_spend"], color="C3")
ax[0, 3].set_ylabel("tv spend")

# Row 2 — adstock
ax[1, 0].plot(data["week"], data["search_adstock"], color="C0")
ax[1, 0].set_ylabel("search adstock")
ax[1, 1].plot(data["week"], data["social_adstock"], color="C1")
ax[1, 1].set_ylabel("social adstock")
ax[1, 2].plot(data["week"], data["ctv_adstock"], color="C2")
ax[1, 2].set_ylabel("ctv adstock")
ax[1, 3].plot(data["week"], data["tv_adstock"], color="C3")
ax[1, 3].set_ylabel("tv adstock")

# Row 2 — adstock
ax[2, 0].plot(data["week"], data["search_adstock_scaled"], color="C0")
ax[2, 0].set_ylabel("search adstock scaled")
ax[2, 1].plot(data["week"], data["social_adstock_scaled"], color="C1")
ax[2, 1].set_ylabel("social adstock scaled")
ax[2, 2].plot(data["week"], data["ctv_adstock_scaled"], color="C2")
ax[2, 2].set_ylabel("ctv adstock scaled")
ax[2, 3].plot(data["week"], data["tv_adstock_scaled"], color="C3")
ax[2, 3].set_ylabel("tv adstock scaled")

# Row 4 — adstock scaled and saturated
ax[3, 0].plot(data["week"], data["search_adstock_saturated"], color="C0")
ax[3, 0].set_ylabel("search adstock scaled and saturated")
ax[3, 1].plot(data["week"], data["social_adstock_saturated"], color="C1")
ax[3, 1].set_ylabel("social adstock scaled and saturated")
ax[3, 2].plot(data["week"], data["ctv_adstock_saturated"], color="C0")
ax[3, 2].set_ylabel("ctv adstock scaled and saturated")
ax[3, 3].plot(data["week"], data["tv_adstock_saturated"], color="C1")
ax[3, 3].set_ylabel("tv adstock scaled and saturated")

fig.suptitle("Media Costs Data - Transformed", fontsize=16)
plt.savefig('pymc_data_transformed.png')
plt.show()

# 7. Generate sales
base_sales = 500

beta_controls = {
    "trend": 1.0,                # assumes trend already in "sales units"
    "seasonality": 1.0,          # assumes seasonality already in "sales units"
    "prime_day_promo": 40.0,
    "mothers_day_promo": 40.0,
}

beta_media = {
    "search": 200.0,
    "social": 300.0,
    "ctv": 400.0,
    "tv": 500.0,
}

channels = ["search", "social", "ctv", "tv"]
for ch in channels:
    sat_col = f"{ch}_adstock_saturated"
    data[f"{ch}_contrib"] = beta_media[ch] * data[sat_col]

data["media_contrib"] = data[[f"{ch}_contrib" for ch in channels]].sum(axis=1)

data["control_contrib"] = (
    beta_controls["trend"] * data["trend"]
    + beta_controls["seasonality"] * data["seasonality"]
    + beta_controls["prime_day_promo"] * data["prime_day_promo"]
    + beta_controls["mothers_day_promo"] * data["mothers_day_promo"]
)

np.random.seed(42)
noise_sd = 12.0

data["sales"] = (
    base_sales
    + data["control_contrib"]
    + data["media_contrib"]
    + np.random.normal(0, noise_sd, size=len(data))
)

print("Sales range:", data["sales"].min(), "→", data["sales"].max())
print("Avg control contrib:", data["control_contrib"].mean())
print("Avg media contrib:", data["media_contrib"].mean())


data.to_csv('pymc_data.csv', index=False)
