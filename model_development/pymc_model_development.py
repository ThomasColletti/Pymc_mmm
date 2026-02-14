import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
import arviz as az
import pandas as pd
import numpy as np
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior
import matplotlib.pyplot as plt

data = pd.read_csv('../artifacts/pymc_data.csv')
data['week'] = pd.to_datetime(data['week'])
# trend feature
data['t'] = range(len(data['week'])) # for underlying upward trend
X = data[['week','search_spend', 'social_spend','ctv_spend','tv_spend','prime_day_promo',
                   'mothers_day_promo','dayofyear','t']]
y = data['sales']

total_spend_per_channel = data[['search_spend', 'social_spend','ctv_spend','tv_spend']].sum(axis=0)
spend_share = total_spend_per_channel / total_spend_per_channel.sum()
print(spend_share)

n_channels = 4
prior_sigma = n_channels * spend_share.to_numpy()
prior_sigma.tolist()
print(prior_sigma)

my_model_config = {
    "intercept": Prior("Normal", mu=500, sigma=150),
    "channel_coef": Prior("HalfNormal", sigma=prior_sigma),
    "control_coef": Prior("Normal", mu=0, sigma=0.05),
    "seasonality": Prior("Laplace", mu=0, b=0.2),
    "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)) # check on this
}

mmm = MMM(
    model_config=my_model_config,
    sampler_config={"progressbar": True},
    date_column="week",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    channel_columns=['search_spend', 'social_spend','ctv_spend','tv_spend'],
    control_columns=['prime_day_promo',
          'mothers_day_promo' ,'t'],
    yearly_seasonality=2,
)

# Generate prior predictive samples
mmm.sample_prior_predictive(X, y, samples=2_000)

fig, ax = plt.subplots()
mmm.plot_prior_predictive(ax=ax, original_scale=True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
plt.show()

rng: np.random.Generator = np.random.default_rng(seed=42)

mmm.fit(X=X, y=y, chains=4, target_accept=0.85, nuts_sampler="numpyro", random_seed=rng)


### Save the model
mmm.save("trained_pymc_mmm_model_01.nc")

#### Model Diagnost + Analysis

g = mmm.graphviz()
g.view()

print("Number of diverging samples:",mmm.idata["sample_stats"]["diverging"].sum().item())

#mmm.fit_result

az.summary(
    data=mmm.fit_result,
    var_names=[
        "intercept",
        "y_sigma",
        "saturation_beta",
        "saturation_lam",
        "adstock_alpha",
        "gamma_control",
        "gamma_fourier",
    ],
).to_csv("../analysis/pymc_model_result.csv")

_ = az.plot_trace(
    data=mmm.fit_result,
    var_names=[
        "intercept",
        "y_sigma",
        "saturation_beta",
        "saturation_lam",
        "adstock_alpha",
        "gamma_control",
        "gamma_fourier",
    ],
    compact=True,
    backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
)
plt.gcf().suptitle("Model Trace", fontsize=16)
plt.show()

mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)
mmm.plot_posterior_predictive(original_scale=True).show()
mmm.plot_posterior_predictive(add_mean=False, add_gradient=True).show()


groups = {
    "Base": [
        "intercept",
        "prime_day_promo",
        "mothers_day_promo",
        "t",
        "yearly_seasonality",
    ],
    "Search": ["search_spend"],
    "Social": ["social_spend"],
    "CTV": ["ctv_spend"],
    "TV": ["tv_spend"]
}

fig = mmm.plot_grouped_contribution_breakdown_over_time(
    stack_groups=groups,
    original_scale=True,
    area_kwargs={
        "color": {
            "Search": "C0",
            "Social": "C1",
            "CTV": "C2",
            "TV": "C3",
            "Base": "gray",
            "Seasonality": "black",
        },
        "alpha": 0.7,
    },
)

fig.suptitle("Contribution Breakdown over Time", fontsize=10)
fig.set_size_inches(18, 10)
fig.subplots_adjust(right=0.80)
fig.show()

mmm.plot_waterfall_components_decomposition().show()

fig = mmm.plot_direct_contribution_curves()
fig.show()

fig = mmm.plot_channel_contribution_grid(start=0, stop=1.5, num=12)
fig.set_size_inches(18, 10)
fig.subplots_adjust(right=0.80)
plt.show()


### ROAs

channel_contribution_original_scale = mmm.compute_channel_contribution_original_scale()
spend_sum = X[["social_spend", "search_spend","ctv_spend","tv_spend"]].sum().to_numpy()

roas_samples = (
    channel_contribution_original_scale.sum(dim="date")
    / spend_sum[np.newaxis, np.newaxis, :]
)

fig, axes = plt.subplots(
    nrows=4, ncols=1, figsize=(12, 10), sharex=True, sharey=False, layout="constrained"
)
az.plot_posterior(roas_samples, ax=axes)
axes[0].set(title="Search")
axes[1].set(title="Social")
axes[2].set(title="CTV")
axes[3].set(title="TV", xlabel="ROAS")
fig.suptitle("ROAS Posterior Distributions", fontsize=18, fontweight="bold", y=1.02)
plt.show()


## ROAs + Contribution

# Get the contribution share samples (posterior)
share_samples = mmm.get_channel_contribution_share_samples()

channels = ["search_spend", "social_spend", "ctv_spend", "tv_spend"]

fig, ax = plt.subplots(figsize=(9, 7))

for i, channel in enumerate(channels):
    # Contribution share mean and HDI
    share_mean = share_samples.sel(channel=channel).mean().to_numpy()
    share_hdi = az.hdi(share_samples.sel(channel=channel)).to_array().to_numpy().ravel()
    # share_hdi -> [lower, upper]

    # ROAS mean and HDI
    roas_mean = roas_samples.sel(channel=channel).mean().to_numpy()
    roas_hdi = az.hdi(roas_samples.sel(channel=channel)).to_array().to_numpy().ravel()
    # roas_hdi -> [lower, upper]

    # Vertical line: ROAS uncertainty at the mean contribution share
    ax.vlines(share_mean, roas_hdi[0], roas_hdi[1], color=f"C{i}", alpha=0.8)

    # Horizontal line: contribution share uncertainty at the mean ROAS
    ax.hlines(roas_mean, share_hdi[0], share_hdi[1], color=f"C{i}", alpha=0.8)

    # Point: mean vs mean, sized by spend share
    ax.scatter(
        share_mean,
        roas_mean,
        s=float(spend_share[channel]) * 1000,  # scale factor for visibility
        color=f"C{i}",
        label=channel.replace("_spend", ""),
    )

# Format x-axis as percent
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

ax.legend(loc="upper left", title="Channel", title_fontsize=12)
ax.set(
    title="Channel Contribution Share vs ROAS",
    xlabel="Contribution Share",
    ylabel="ROAS",
)

plt.show()
