import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD REAL NHS DATA ─────────────────────────────────────────────────────
df_raw = pd.read_excel(
    'data/WLMDS-Summary-to-25-Jan-2026.xlsx',
    sheet_name='National-Time Series',
    skiprows=13
)

# ── 2. CLEAN & RENAME COLUMNS ─────────────────────────────────────────────────
df = df_raw[['Unnamed: 1', 'Total Waiting List', 'Up to 18 weeks', 'Over 52 and up to 65 weeks', '% within 18 weeks*', '% > 52 weeks*']].copy()
df.columns = ['week', 'total_waiting', 'within_18_weeks', 'over_52_weeks', 'pct_within_18', 'pct_over_52']

# Drop rows with no date
df = df.dropna(subset=['week'])
df = df[pd.to_datetime(df['week'], errors='coerce').notna()]
df['week'] = pd.to_datetime(df['week'])
df = df.sort_values('week').reset_index(drop=True)

# Convert to numeric
for col in ['total_waiting', 'within_18_weeks', 'over_52_weeks', 'pct_within_18', 'pct_over_52']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert pct to percentage if decimal
if df['pct_within_18'].mean() < 2:
    df['pct_within_18'] = df['pct_within_18'] * 100
    df['pct_over_52'] = df['pct_over_52'] * 100

print("✅ Real NHS Data Loaded!")
print(f"   Date range: {df['week'].min().strftime('%d %b %Y')} to {df['week'].max().strftime('%d %b %Y')}")
print(f"   Total weeks: {len(df)}")
print(f"\n📊 KEY METRICS:")
print(f"   Avg total waiting:    {df['total_waiting'].mean():,.0f}")
print(f"   Avg % within 18 wks: {df['pct_within_18'].mean():.1f}%")
print(f"   Peak 52+ week:        {df['over_52_weeks'].max():,.0f}")
print(f"   Latest waiting list:  {df['total_waiting'].iloc[-1]:,.0f}")
print(f"   Latest % within 18:   {df['pct_within_18'].iloc[-1]:.1f}%")

# ── 3. PLOTS ──────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NHS RTT Waiting Times — REAL DATA (Sep 2021 – Jan 2026)',
             fontsize=15, fontweight='bold')

# Plot 1: Total waiting list
ax1 = axes[0, 0]
ax1.fill_between(df['week'], df['total_waiting'], alpha=0.2, color='#005EB8')
ax1.plot(df['week'], df['total_waiting'], color='#005EB8', linewidth=2)
ax1.set_title('Total Patients Waiting', fontweight='bold')
ax1.set_ylabel('Patients')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

# Plot 2: % within 18 weeks
ax2 = axes[0, 1]
ax2.plot(df['week'], df['pct_within_18'], color='#007F3B', linewidth=2)
ax2.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax2.fill_between(df['week'], df['pct_within_18'], 92,
                 where=df['pct_within_18'] < 92, alpha=0.15, color='red', label='Below Target')
ax2.set_title('% Seen Within 18 Weeks vs Target', fontweight='bold')
ax2.set_ylabel('% Within 18 Weeks')
ax2.legend()

# Plot 3: 52+ week waiters
ax3 = axes[1, 0]
ax3.bar(df['week'], df['over_52_weeks'],
        color=['#d32f2f' if v > 200000 else '#FF8C00' for v in df['over_52_weeks']],
        width=5)
ax3.set_title('Patients Waiting 52+ Weeks', fontweight='bold')
ax3.set_ylabel('Patients')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

# Plot 4: Rolling average trend
ax4 = axes[1, 1]
df['rolling_pct'] = df['pct_within_18'].rolling(4).mean()
ax4.plot(df['week'], df['pct_within_18'], color='#005EB8', alpha=0.3, linewidth=1, label='Weekly')
ax4.plot(df['week'], df['rolling_pct'], color='#005EB8', linewidth=2.5, label='4-week avg')
ax4.axhline(92, color='red', linestyle='--', linewidth=1.5, label='92% Target')
ax4.set_title('% Within 18 Weeks — Rolling Average', fontweight='bold')
ax4.set_ylabel('% Within 18 Weeks')
ax4.legend()

plt.tight_layout()
plt.savefig('data/nhs_real_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved to data/nhs_real_analysis.png")
plt.show()
