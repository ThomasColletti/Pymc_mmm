import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pymc_marketing.mmm.budget_optimizer import optimizer_xarray_builder

from pymc_marketing.mmm import MMM

warnings.filterwarnings("ignore")

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100

#load model
mmm = MMM.load("trained_pymc_mmm_model_01.nc")
#
# response_curve_fig = mmm.plot_direct_contribution_curves()
# response_curve_fig.show()
#
# mmm.plot_direct_contribution_curves(show_fit=True, xlim_max=1.5).show()

print(f"Model was train using the {mmm.saturation.__class__.__name__} function")
print(f"and the {mmm.adstock.__class__.__name__} function")

az.summary(
    data=mmm.fit_result,
    var_names=[
        "saturation_beta",
        "saturation_lam",
        "adstock_alpha",
    ],
)


# ---- Settings ----
time_unit_budget = 5   # interpret as 5K, 5M, etc. per week
campaign_period = 8    # weeks

total_budget = time_unit_budget * campaign_period
print(f"Total budget for the {campaign_period} weeks: {total_budget}")

# ---- Channels ----
channels = ["search_spend", "social_spend", "ctv_spend", "tv_spend"]

# ---- Initial budget split (equal split) ----
budget_per_channel = time_unit_budget / len(channels)

initial_budget = optimizer_xarray_builder(
    np.array([2.0, 1.0, 1.0, 1.0]),
    channel=channels
)

# ---- Bounds per channel (per week) ----
min_budget, max_budget = 1, 5

budget_bounds = optimizer_xarray_builder(
    np.array([[min_budget, max_budget]] * len(channels)),
    channel=channels,
    bound=["lower", "upper"],
)

model_granularity = "weekly"
allocation_strategy, optimization_result = mmm.optimize_budget(
    budget=time_unit_budget,
    num_periods=campaign_period,
    budget_bounds=budget_bounds)

response = mmm.sample_response_distribution(
    allocation_strategy=allocation_strategy,  # dict or xarray.DataArray
    time_granularity=model_granularity,
    num_periods=campaign_period,
    noise_level=0.01,
)

print("\n=== Budget Allocation Summary ===\n")
for channel, budget in response.allocation.to_dataframe().iterrows():
    print(f"Channel {channel:>2}: {budget['allocation']:>8.2f}M")
print("\n" + "-" * 30)
print(f"Total Budget: {sum(response.allocation.to_numpy()):>8.2f}M")
print("-" * 30)
