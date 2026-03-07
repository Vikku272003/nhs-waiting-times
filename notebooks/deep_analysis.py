import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/nhs_waiting_times.csv', parse_dates=['month'])
print(f"✅ Loaded {len(df):,} records")

# ── 2. CORRELATION ANALYSIS ───────────────────────────────────────────────────
print("\n🔗 CORRELATION ANALYSIS:")
numeric_cols = ['total_waiting', 'over_52_weeks', 'median_wait_weeks', 'pct_within_18_weeks']
corr = df[numeric_cols].corr()
print(corr.round(2))

# ── 3. COVID IMPACT STATISTICAL TEST ─────────────────────────────────────────
pre_covid = df[df['month'] < '2020-07-01']['pct_within_18_weeks']
during_covid = df[(df['month'] >= '2020-07-01') & (df['month'] < '2021-06-01')]['pct_within_18_weeks']
recovery = df[(df['month'] >= '2021-06-01') & (df['month'] < '2023-01-01')]['pct_within_18_weeks']
current = df[df['month'] >= '2023-01-01']['pct_within_18_weeks']

print("\n📊 COVID IMPACT ANALYSIS:")
print(f"  Pre-COVID avg:      {pre_covid.mean():.1f}%")
print(f"  During COVID avg:   {during_covid.mean():.1f}%")
print(f"  Recovery avg:       {recovery.mean():.1f}%")
print(f"  Current avg:        {current.mean():.1f}%")

t_stat, p_value = stats.ttest_ind(pre_covid, during_covid)
print(f"\n  T-test (pre vs during): t={t_stat:.2f}, p={p_value:.4f}")
print(f"  {'✅ Statistically significant difference' if p_value < 0.05 else '❌ No significant difference'}")

# ── 4. FORECASTING ────────────────────────────────────────────────────────────
monthly = df.groupby('month').agg(
    total_waiting=('total_waiting', 'sum'),
    pct_within_18=('pct_within_18_weeks', 'mean'),
    over_52_weeks=('over_52_weeks', 'sum')
).reset_index()

# Use only recovery + current data for forecast
forecast_base = monthly[monthly['month'] >= '2021-06-01'].copy()
forecast_base['month_num'] = np.arange(len(forecast_base))

X = forecast_base[['month_num']]
y = forecast_base['pct_within_18']

model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)

# Forecast 12 months ahead
future_months = pd.date_range(start='2025-01-01', periods=12, freq='MS')
future_X = np.arange(len(forecast_base), len(forecast_base) + 12).reshape(-1, 1)
forecast_values = model.predict(future_X)

print(f"\n📈 FORECASTING (Linear Regression):")
print(f"  R² score: {r2:.3f}")
print(f"  Monthly improvement rate: {model.coef_[0]:.2f}% per month")
print(f"  Forecast Jan 2026: {forecast_values[0]:.1f}%")
print(f"  Forecast Dec 2026: {forecast_values[-1]:.1f}%")
target_month = next((i for i, v in enumerate(forecast_values) if v >= 92), None)
if target_month:
    print(f"  🎯 Projected to hit 92% target: {future_months[target_month].strftime('%b %Y')}")
else:
    print(f"  ⚠️  92% target not reached within 2025 forecast window")

# ── 5. BEST & WORST PERFORMERS ────────────────────────────────────────────────
combo = df.groupby(['specialty', 'region']).agg(
    avg_wait=('median_wait_weeks', 'mean'),
    avg_pct_18=('pct_within_18_weeks', 'mean')
).reset_index()

print("\n🏆 TOP 5 BEST PERFORMING (specialty + region):")
print(combo.nlargest(5, 'avg_pct_18')[['specialty', 'region', 'avg_pct_18', 'avg_wait']].to_string(index=False))

print("\n⚠️  TOP 5 WORST PERFORMING (specialty + region):")
print(combo.nsmallest(5, 'avg_pct_18')[['specialty', 'region', 'avg_pct_18', 'avg_wait']].to_string(index=False))

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NHS Waiting Times — Deep Analysis', fontsize=16, fontweight='bold')

# Plot 1: Correlation heatmap
ax1 = axes[0, 0]
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax1, square=True, linewidths=0.5)
ax1.set_title('Correlation Matrix', fontweight='bold')

# Plot 2: Forecast
ax2 = axes[0, 1]
ax2.plot(monthly['month'], monthly['pct_within_18'], color='#005EB8',
         linewidth=2, label='Actual')
ax2.plot(future_months, forecast_values, color='orange',
         linewidth=2, linestyle='--', label='Forecast 2025')
ax2.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax2.fill_between(future_months, forecast_values, 92,
                 where=forecast_values < 92, alpha=0.15, color='red')
ax2.set_title('Forecast: % Within 18 Weeks (2025)', fontweight='bold')
ax2.set_ylabel('% Within 18 Weeks')
ax2.legend()

# Plot 3: Period comparison boxplot
ax3 = axes[1, 0]
df['period'] = pd.cut(df['month'],
    bins=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-07-01'),
          pd.Timestamp('2021-06-01'), pd.Timestamp('2023-01-01'),
          pd.Timestamp('2025-01-01')],
    labels=['Pre-COVID', 'COVID Impact', 'Recovery', 'Current'])
sns.boxplot(data=df, x='period', y='pct_within_18_weeks',
            palette=['#007F3B','#d32f2f','#FF8C00','#005EB8'], ax=ax3)
ax3.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax3.set_title('Performance Distribution by Period', fontweight='bold')
ax3.set_ylabel('% Within 18 Weeks')
ax3.legend()

# Plot 4: Heatmap specialty vs region
ax4 = axes[1, 1]
pivot = df.groupby(['specialty', 'region'])['pct_within_18_weeks'].mean().unstack()
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
            ax=ax4, linewidths=0.3, vmin=60, vmax=90)
ax4.set_title('% Within 18 Weeks: Specialty × Region', fontweight='bold')
ax4.set_xlabel('')

plt.tight_layout()
plt.savefig('data/nhs_deep_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Deep analysis chart saved to data/nhs_deep_analysis.png")
plt.show()