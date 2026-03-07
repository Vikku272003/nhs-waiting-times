import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/nhs_waiting_times.csv', parse_dates=['month'])
print(f"✅ Loaded {len(df):,} records")
print("\n📊 KEY METRICS:")
print(f"  Total patient-months: {df['total_waiting'].sum():,.0f}")
print(f"  Avg % within 18 weeks: {df['pct_within_18_weeks'].mean():.1f}%")
print(f"  Avg median wait: {df['median_wait_weeks'].mean():.1f} weeks")
print(f"  Peak 52+ week waiters: {df.groupby('month')['over_52_weeks'].sum().max():,.0f}")
monthly = df.groupby('month').agg(
    total_waiting=('total_waiting', 'sum'),
    over_52_weeks=('over_52_weeks', 'sum'),
    pct_within_18=('pct_within_18_weeks', 'mean')
).reset_index()
by_specialty = df.groupby('specialty').agg(
    avg_wait=('median_wait_weeks', 'mean'),
    avg_pct_18=('pct_within_18_weeks', 'mean'),
    total_over_52=('over_52_weeks', 'sum')
).reset_index().sort_values('avg_wait', ascending=False)
by_region = df.groupby('region').agg(
    avg_wait=('median_wait_weeks', 'mean'),
    avg_pct_18=('pct_within_18_weeks', 'mean'),
    total_waiting=('total_waiting', 'sum')
).reset_index().sort_values('avg_pct_18', ascending=True)
sns.set_theme(style='whitegrid', palette='Blues_d')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NHS Referral to Treatment (RTT) Waiting Times Analysis\n2020–2024',
             fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Total waiting list over time
ax1 = axes[0, 0]
ax1.fill_between(monthly['month'], monthly['total_waiting'], alpha=0.3, color='#005EB8')
ax1.plot(monthly['month'], monthly['total_waiting'], color='#005EB8', linewidth=2)
ax1.axvline(pd.Timestamp('2020-07-01'), color='red', linestyle='--', alpha=0.7, label='COVID Impact')
ax1.axvline(pd.Timestamp('2021-06-01'), color='green', linestyle='--', alpha=0.7, label='Recovery Phase')
ax1.set_title('Total Patients Waiting Over Time', fontweight='bold')
ax1.set_ylabel('Total Patients Waiting')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax1.legend()

# Plot 2: % seen within 18 weeks
ax2 = axes[0, 1]
ax2.plot(monthly['month'], monthly['pct_within_18'], color='#007F3B', linewidth=2)
ax2.axhline(92, color='red', linestyle='--', linewidth=1.5, label='NHS 92% Target')
ax2.fill_between(monthly['month'], monthly['pct_within_18'], 92,
                 where=monthly['pct_within_18'] < 92, alpha=0.2, color='red', label='Below Target')
ax2.set_title('% Patients Seen Within 18 Weeks vs NHS Target', fontweight='bold')
ax2.set_ylabel('% Within 18 Weeks')
ax2.set_ylim(50, 100)
ax2.legend()

# Plot 3: Average wait by specialty
ax3 = axes[1, 0]
bars = ax3.barh(by_specialty['specialty'], by_specialty['avg_wait'], color='#005EB8')
ax3.bar_label(bars, fmt='%.1f wks', padding=3, fontsize=9)
ax3.set_title('Average Median Wait by Specialty', fontweight='bold')
ax3.set_xlabel('Median Wait (Weeks)')

# Plot 4: % within 18 weeks by region
ax4 = axes[1, 1]
colors = ['#d32f2f' if x < 75 else '#005EB8' for x in by_region['avg_pct_18']]
bars2 = ax4.barh(by_region['region'], by_region['avg_pct_18'], color=colors)
ax4.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
ax4.axvline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax4.set_title('% Within 18 Weeks by Region', fontweight='bold')
ax4.set_xlabel('% Within 18 Weeks')
ax4.legend()

plt.tight_layout()
plt.savefig('data/nhs_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved to data/nhs_analysis.png")
plt.show()

print("\n📋 SPECIALTY BREAKDOWN:")
print(by_specialty.to_string(index=False))
print("\n🗺️  REGIONAL BREAKDOWN:")
print(by_region.to_string(index=False))
