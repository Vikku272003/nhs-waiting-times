import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD REAL NHS DATA ─────────────────────────────────────────────────────
df_raw = pd.read_excel(
    'data/WLMDS-Summary-to-25-Jan-2026.xlsx',
    sheet_name='National-Time Series',
    skiprows=13
)

df = df_raw[['Unnamed: 1', 'Total Waiting List', 'Up to 18 weeks',
             'Over 52 and up to 65 weeks', '% within 18 weeks*', '% > 52 weeks*']].copy()
df.columns = ['week', 'total_waiting', 'within_18_weeks', 'over_52_weeks', 'pct_within_18', 'pct_over_52']
df = df[pd.to_datetime(df['week'], errors='coerce').notna()]
df['week'] = pd.to_datetime(df['week'])
for col in ['total_waiting', 'within_18_weeks', 'over_52_weeks', 'pct_within_18', 'pct_over_52']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
if df['pct_within_18'].mean() < 2:
    df['pct_within_18'] = df['pct_within_18'] * 100
    df['pct_over_52'] = df['pct_over_52'] * 100
df = df.dropna().sort_values('week').reset_index(drop=True)
print(f"✅ Loaded {len(df)} weeks of real NHS data")

# ── 2. PERIOD ANALYSIS ────────────────────────────────────────────────────────
df['period'] = pd.cut(df['week'],
    bins=[pd.Timestamp('2021-01-01'), pd.Timestamp('2022-06-01'),
          pd.Timestamp('2023-06-01'), pd.Timestamp('2024-06-01'),
          pd.Timestamp('2026-12-31')],
    labels=['2021-22 (Backlog)', '2022-23 (Crisis)', '2023-24 (Stabilising)', '2024-25 (Recovery)'])

print("\n📊 PERIOD ANALYSIS:")
period_stats = df.groupby('period').agg(
    avg_waiting=('total_waiting', 'mean'),
    avg_pct_18=('pct_within_18', 'mean'),
    avg_over_52=('over_52_weeks', 'mean')
).round(1)
print(period_stats.to_string())

# ── 3. STATISTICAL TEST ───────────────────────────────────────────────────────
early = df[df['week'] < '2023-01-01']['pct_within_18']
recent = df[df['week'] >= '2023-01-01']['pct_within_18']
t_stat, p_value = stats.ttest_ind(early, recent)
print(f"\n🔬 STATISTICAL TEST (Early vs Recent):")
print(f"   Early avg:  {early.mean():.1f}%")
print(f"   Recent avg: {recent.mean():.1f}%")
print(f"   p-value: {p_value:.4f} — {'✅ Significant improvement' if p_value < 0.05 else '❌ No significant difference'}")

# ── 4. FORECASTING ────────────────────────────────────────────────────────────
recent_df = df[df['week'] >= '2023-01-01'].copy()
recent_df['week_num'] = np.arange(len(recent_df))
X = recent_df[['week_num']]
y_pct = recent_df['pct_within_18']
y_wait = recent_df['total_waiting']

model_pct = LinearRegression().fit(X, y_pct)
model_wait = LinearRegression().fit(X, y_wait)

future_weeks = pd.date_range(start='2026-02-01', periods=52, freq='W')
future_X = np.arange(len(recent_df), len(recent_df) + 52).reshape(-1, 1)
forecast_pct = model_pct.predict(future_X)
forecast_wait = model_wait.predict(future_X)

print(f"\n📈 FORECAST (next 12 months):")
print(f"   Current % within 18 weeks: {df['pct_within_18'].iloc[-1]:.1f}%")
print(f"   Forecast Feb 2027:          {forecast_pct[-1]:.1f}%")
print(f"   Current waiting list:       {df['total_waiting'].iloc[-1]:,.0f}")
print(f"   Forecast Feb 2027:          {forecast_wait[-1]:,.0f}")
target_hit = next((i for i, v in enumerate(forecast_pct) if v >= 92), None)
print(f"   92% target: {'Never reached in forecast window ⚠️' if not target_hit else future_weeks[target_hit].strftime('%b %Y')}")

# ── 5. CORRELATION ────────────────────────────────────────────────────────────
print(f"\n🔗 CORRELATIONS:")
corr = df[['total_waiting', 'over_52_weeks', 'pct_within_18']].corr().round(2)
print(corr)

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NHS RTT — Deep Analysis on REAL DATA', fontsize=15, fontweight='bold')

# Plot 1: Forecast
ax1 = axes[0, 0]
ax1.plot(df['week'], df['pct_within_18'], color='#005EB8', linewidth=2, label='Actual')
ax1.plot(future_weeks, forecast_pct, color='orange', linewidth=2, linestyle='--', label='Forecast 2026')
ax1.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax1.fill_between(future_weeks, forecast_pct, 92, alpha=0.15, color='red')
ax1.set_title('Forecast: % Within 18 Weeks', fontweight='bold')
ax1.set_ylabel('%')
ax1.legend()

# Plot 2: Waiting list forecast
ax2 = axes[0, 1]
ax2.plot(df['week'], df['total_waiting']/1e6, color='#005EB8', linewidth=2, label='Actual')
ax2.plot(future_weeks, forecast_wait/1e6, color='orange', linewidth=2, linestyle='--', label='Forecast 2026')
ax2.set_title('Forecast: Total Waiting List', fontweight='bold')
ax2.set_ylabel('Patients (Millions)')
ax2.legend()

# Plot 3: Period boxplot
ax3 = axes[1, 0]
period_data = df.dropna(subset=['period'])
sns.boxplot(data=period_data, x='period', y='pct_within_18',
            palette=['#d32f2f', '#FF8C00', '#FDD835', '#007F3B'], ax=ax3)
ax3.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax3.set_title('Performance by Period', fontweight='bold')
ax3.set_ylabel('% Within 18 Weeks')
ax3.tick_params(axis='x', rotation=15)
ax3.legend()

# Plot 4: Correlation heatmap
ax4 = axes[1, 1]
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax4, square=True, linewidths=0.5)
ax4.set_title('Correlation Matrix (Real Data)', fontweight='bold')

plt.tight_layout()
plt.savefig('data/nhs_real_deep_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Deep analysis chart saved!")
plt.show()