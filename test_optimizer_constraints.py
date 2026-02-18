#!/usr/bin/env python
import arviz as az
from pymc_marketing.mmm.budget_optimizer import CustomModelWrapper, BudgetOptimizer
from pymc_marketing.mmm import MMM

mmm = MMM.load('artifacts/trained_pymc_mmm_model_01.nc')
idata = az.convert_to_inference_data(mmm.fit_result)

channels = ['search_spend', 'social_spend', 'ctv_spend', 'tv_spend']
training_data_len = 105
total_budget = 40

wrapper = CustomModelWrapper(
    base_model=mmm.model,
    idata=idata,
    channels=channels,
)

print("=" * 70)
print("Test 1: With default_constraints=True (default)")
print("=" * 70)
optimizer1 = BudgetOptimizer(
    num_periods=training_data_len,
    model=wrapper,
    default_constraints=True
)
alloc1, opt_result1 = optimizer1.allocate_budget(total_budget=total_budget)
print(f'Allocation: {alloc1.values}')
print(f'Objective: {opt_result1.fun:.4f}')

print("\n" + "=" * 70)
print("Test 2: With default_constraints=False")
print("=" * 70)
optimizer2 = BudgetOptimizer(
    num_periods=training_data_len,
    model=wrapper,
    default_constraints=False
)
alloc2, opt_result2 = optimizer2.allocate_budget(total_budget=total_budget)
print(f'Allocation: {alloc2.values}')
print(f'Objective: {opt_result2.fun:.4f}')
