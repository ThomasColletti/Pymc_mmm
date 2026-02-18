import warnings
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm.budget_optimizer import CustomModelWrapper, BudgetOptimizer
from pymc_marketing.mmm import MMM

warnings.filterwarnings("ignore")

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100

# === Load Model ===
print("=== Loading Model ===")
mmm = MMM.load("artifacts/trained_pymc_mmm_model_01.nc")

print(f"Model saturation: {mmm.saturation.__class__.__name__}")
print(f"Model adstock: {mmm.adstock.__class__.__name__}")

print("\n=== Model Parameters Summary ===")
az.summary(
    data=mmm.fit_result,
    var_names=[
        "saturation_beta",
        "saturation_lam",
        "adstock_alpha",
    ],
)

# === Settings ===
print("\n=== Budget Settings ===")
time_unit_budget = 5   # interpret as 5K, 5M, etc. per week
campaign_period = 8    # weeks
total_budget = time_unit_budget * campaign_period

channels = ["search_spend", "social_spend", "ctv_spend", "tv_spend"]

print(f"Weekly budget: {time_unit_budget}M")
print(f"Campaign period: {campaign_period} weeks")
print(f"Total budget: {total_budget}M")
print(f"Channels: {channels}")

# === Load Training Data for Date Range ===
print("\n=== Loading Training Data ===")
training_data = pd.read_csv("artifacts/pymc_data.csv")
training_data["week"] = pd.to_datetime(training_data["week"])

start_date = training_data["week"].min().strftime("%Y-%m-%d")
end_date = training_data["week"].max().strftime("%Y-%m-%d")

print(f"Training period: {start_date} to {end_date}")
print(f"Training weeks: {len(training_data)}")

# === Posterior Summary ===
print("\n=== Posterior Channel Contributions ===")
channel_contrib = mmm.fit_result['channel_contribution']
print(f"Shape: {channel_contrib.shape}")

# Mean contribution per channel
for i, ch in enumerate(channels):
    mean_contrib = float(channel_contrib.isel(channel=i).mean().values)
    std_contrib = float(channel_contrib.isel(channel=i).std().values)
    print(f"  {ch:>15}: {mean_contrib:>7.2f} ± {std_contrib:>6.2f}")

# === Initialize Optimizer ===
print("\n=== Initializing Budget Optimizer ===")
try:
    # Convert Dataset to InferenceData if needed
    if hasattr(mmm.fit_result, 'attrs'):  # It's a Dataset
        idata = az.convert_to_inference_data(mmm.fit_result)
    else:
        idata = mmm.fit_result
    
    # CustomModelWrapper wraps the model for optimization
    model_wrapper = CustomModelWrapper(
        base_model=mmm.model,
        idata=idata,
        channels=channels,
    )
    print("✓ Model wrapper created successfully")
    
    # BudgetOptimizer uses the wrapper
    optimizer = BudgetOptimizer(
        num_periods=len(training_data),
        model=model_wrapper
    )
    print("✓ Budget optimizer created successfully")
except Exception as e:
    print(f"✗ Error initializing optimizer: {e}")
    import traceback
    traceback.print_exc()
    raise

# === Run Optimization ===
print("\n=== Running Budget Optimization ===")
print(f"Optimizing for total budget: {total_budget}M")

try:
    allocation_xarray, scipy_opt_result = optimizer.allocate_budget(
        total_budget=total_budget
    )
    print("✓ Optimization succeeded")
    print(f"  Success: {scipy_opt_result.success}")
    print(f"  Message: {scipy_opt_result.message}")
    print(f"  Objective value: {scipy_opt_result.fun:.4f}")
    
except Exception as e:
    print(f"✗ Optimization failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# === Display Optimal Allocation ===
print("\n=== Optimal Budget Allocation ===")
print(f"\nAllocation xarray dims: {allocation_xarray.dims if hasattr(allocation_xarray, 'dims') else 'N/A'}")
print(f"Allocation xarray shape: {allocation_xarray.shape if hasattr(allocation_xarray, 'shape') else 'N/A'}")

if hasattr(allocation_xarray, 'values'):
    alloc_values = allocation_xarray.values.flatten()
    total_alloc = alloc_values.sum()
    
    print(f"\n{'Channel':<20} {'Budget (M)':<15} {'% of Total':<15}")
    print("-" * 50)
    for i, ch in enumerate(channels):
        if i < len(alloc_values):
            budget = alloc_values[i]
            pct = (budget / total_alloc * 100) if total_alloc > 0 else 0
            print(f"{ch:<20} {budget:>10.2f}  {pct:>12.1f}%")
    
    print("-" * 50)
    print(f"{'Total':<20} {total_alloc:>10.2f}  {100.0:>12.1f}%")

# === Comparison: Optimized vs Equal Split ===
print("\n=== Allocation Comparison ===")
equal_split = np.ones(len(channels)) * (total_budget / len(channels))

if hasattr(allocation_xarray, 'values'):
    alloc_values = allocation_xarray.values.flatten()
    
    print(f"\n{'Channel':<20} {'Optimized':<15} {'Equal Split':<15} {'Difference':<15}")
    print("-" * 65)
    for i, ch in enumerate(channels):
        if i < len(alloc_values):
            opt = alloc_values[i]
            eq = equal_split[i]
            diff = opt - eq
            pct_change = (diff / eq * 100) if eq > 0 else 0
            print(f"{ch:<20} {opt:>10.2f}M  {eq:>10.2f}M  {diff:>+8.2f}M ({pct_change:+6.1f}%)")
print("\n" + "=" * 65)
print("Optimization complete!")
#
# response_curve_fig = mmm.plot_direct_contribution_curves()
#
# Additional Analysis Complete
print("\n" + "=" * 65)
print("Optimization complete!")