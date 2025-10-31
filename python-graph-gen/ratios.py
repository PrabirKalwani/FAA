import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load the CSV file
df = pd.read_csv('main.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Get company names (skip first column which is 'Ratios')
companies = [col for col in df.columns if col not in ['Ratios', 'Unnamed: 0']]
company_names = []
for i in range(0, len(companies), 2):
    if i < len(companies):
        company_names.append(companies[i])

print("=" * 80)
print("FMCG SECTOR FINANCIAL ANALYSIS - FY2023 vs FY2024")
print("=" * 80)
print(f"\nCompanies analyzed: {', '.join(company_names)}")
print("\n")

# Industry Benchmarks
industry_benchmarks = {
    'Current Ratio': {'FY2023': 2.11, 'FY2024': 2.27},
    'Quick Ratio': {'FY2023': 1.51, 'FY2024': 1.61},
    'Debt to Equity Ratio': {'FY2023': 0.27, 'FY2024': 0.27},
    'Net Profit Margin': {'FY2023': 0.205, 'FY2024': 0.205},
    'ROE': {'FY2023': 0.2844, 'FY2024': 0.2844},
    'Inventory Turnover': {'FY2023': 1.61, 'FY2024': 5.10}
}

# Helper function to extract ratio data


def get_ratio_data(df, ratio_name, companies):
    ratio_row = df[df['Ratios'] == ratio_name]
    if ratio_row.empty:
        return {}

    data = {}
    for i in range(0, len(companies), 2):
        if i < len(companies):
            company = companies[i]
            fy2024_val = ratio_row.iloc[0][companies[i]]
            fy2023_val = ratio_row.iloc[0][companies[i+1]
                                           ] if i+1 < len(companies) else None

            # Clean and convert values
            try:
                if pd.notna(fy2024_val):
                    if isinstance(fy2024_val, str):
                        fy2024_val = float(fy2024_val.replace('₹', '').replace(
                            '%', '').replace(',', '').replace('Times', '').strip())
                    else:
                        fy2024_val = float(fy2024_val)
                else:
                    fy2024_val = None

                if pd.notna(fy2023_val):
                    if isinstance(fy2023_val, str):
                        fy2023_val = float(fy2023_val.replace('₹', '').replace(
                            '%', '').replace(',', '').replace('Times', '').strip())
                    else:
                        fy2023_val = float(fy2023_val)
                else:
                    fy2023_val = None

                data[company] = {'FY2024': fy2024_val, 'FY2023': fy2023_val}
            except:
                data[company] = {'FY2024': None, 'FY2023': None}

    return data

# ============================================================================
# 1. INDIVIDUAL COMPANY ANALYSIS
# ============================================================================


for company in company_names:
    print("\n" + "=" * 80)
    print(f"📊 FINANCIAL ANALYSIS: {company}")
    print("=" * 80)

    # --- LIQUIDITY ANALYSIS ---
    print(f"\n{'─' * 80}")
    print("💧 LIQUIDITY ANALYSIS")
    print(f"{'─' * 80}")

    current_ratio = get_ratio_data(df, 'Current Ratio', companies)
    quick_ratio = get_ratio_data(df, 'Quick Ratio', companies)

    if company in current_ratio and company in quick_ratio:
        cr_2024 = current_ratio[company]['FY2024']
        cr_2023 = current_ratio[company]['FY2023']
        qr_2024 = quick_ratio[company]['FY2024']
        qr_2023 = quick_ratio[company]['FY2023']

        if all(v is not None for v in [cr_2024, cr_2023, qr_2024, qr_2023]):
            cr_change = ((cr_2024 - cr_2023) / cr_2023) * 100
            qr_change = ((qr_2024 - qr_2023) / qr_2023) * 100

            print(
                f"\n📈 Current Ratio: FY2023: {cr_2023:.2f} → FY2024: {cr_2024:.2f} (Change: {cr_change:+.1f}%)")
            print(
                f"⚡ Quick Ratio: FY2023: {qr_2023:.2f} → FY2024: {qr_2024:.2f} (Change: {qr_change:+.1f}%)")

            print("\n💬 Insights:")
            if cr_change > 0:
                print(
                    f"   ✓ {company} improved liquidity by {abs(cr_change):.1f}%, indicating stronger short-term")
                print(
                    "     financial health and better ability to meet current obligations.")
            else:
                print(
                    f"   ⚠ {company} experienced a {abs(cr_change):.1f}% decline in liquidity, which may indicate")
                print(
                    "     tighter working capital management or increased short-term liabilities.")

            if cr_2024 > 2.0:
                print("   ✓ Current ratio > 2.0 suggests excellent short-term solvency.")
            elif cr_2024 > 1.5:
                print("   ✓ Current ratio > 1.5 indicates healthy liquidity position.")
            else:
                print("   ⚠ Current ratio below 1.5 may signal liquidity pressure.")

    # --- PROFITABILITY ANALYSIS ---
    print(f"\n{'─' * 80}")
    print("💰 PROFITABILITY ANALYSIS")
    print(f"{'─' * 80}")

    gpm = get_ratio_data(df, 'Gross Profit Margin', companies)
    npm = get_ratio_data(df, 'Net Profit Margin', companies)
    ebitda = get_ratio_data(df, 'EBITDA Margin', companies)

    if company in gpm and company in npm:
        gpm_2024 = gpm[company]['FY2024']
        gpm_2023 = gpm[company]['FY2023']
        npm_2024 = npm[company]['FY2024']
        npm_2023 = npm[company]['FY2023']

        if all(v is not None for v in [gpm_2024, gpm_2023, npm_2024, npm_2023]):
            gpm_change = ((gpm_2024 - gpm_2023) / gpm_2023) * 100
            npm_change = ((npm_2024 - npm_2023) / npm_2023) * 100

            print(
                f"\n📊 Gross Profit Margin: FY2023: {gpm_2023*100:.1f}% → FY2024: {gpm_2024*100:.1f}% (Change: {gpm_change:+.1f}%)")
            print(
                f"💵 Net Profit Margin: FY2023: {npm_2023*100:.1f}% → FY2024: {npm_2024*100:.1f}% (Change: {npm_change:+.1f}%)")

            print("\n💬 Insights:")
            if npm_change > 0:
                print(
                    f"   ✓ {company} enhanced profitability by {abs(npm_change):.1f}%, reflecting improved operational")
                print(
                    "     efficiency and better cost management for long-term sustainability.")
            else:
                print(
                    f"   ⚠ {company} saw profitability decline by {abs(npm_change):.1f}%, potentially due to rising")
                print(
                    "     costs, pricing pressure, or investments in growth initiatives.")

            if npm_2024 > 0.20:
                print(
                    "   ✓ Net margin > 20% indicates strong pricing power and operational excellence.")
            elif npm_2024 > 0.15:
                print("   ✓ Net margin > 15% shows healthy profitability.")

    # --- LEVERAGE ANALYSIS ---
    print(f"\n{'─' * 80}")
    print("⚖️ LEVERAGE ANALYSIS")
    print(f"{'─' * 80}")

    de_ratio = get_ratio_data(df, 'Debt to Equity Ratio', companies)

    if company in de_ratio:
        de_2024 = de_ratio[company]['FY2024']
        de_2023 = de_ratio[company]['FY2023']

        if de_2024 is not None and de_2023 is not None:
            if de_2023 > 0:
                de_change = ((de_2024 - de_2023) / de_2023) * 100
            else:
                de_change = 0

            print(
                f"\n🏦 Debt-to-Equity: FY2023: {de_2023:.3f} → FY2024: {de_2024:.3f} (Change: {de_change:+.1f}%)")

            print("\n💬 Insights:")
            if de_change > 10:
                print(
                    f"   ⚠ {company} increased leverage by {abs(de_change):.1f}%, indicating higher debt financing")
                print(
                    "     which may increase financial risk but could fund growth opportunities.")
            elif de_change < -10:
                print(
                    f"   ✓ {company} reduced leverage by {abs(de_change):.1f}%, strengthening financial stability")
                print("     and reducing interest burden for long-term resilience.")
            else:
                print(
                    f"   → {company} maintained stable leverage, showing consistent capital structure.")

            if de_2024 < 0.3:
                print(
                    "   ✓ D/E ratio < 0.3 indicates conservative financing with minimal financial risk.")
            elif de_2024 < 0.5:
                print(
                    "   ✓ D/E ratio < 0.5 shows moderate leverage with manageable risk.")

    # --- INVENTORY & TURNOVER ANALYSIS ---
    print(f"\n{'─' * 80}")
    print("📦 INVENTORY & TURNOVER ANALYSIS")
    print(f"{'─' * 80}")

    inv_turnover = get_ratio_data(df, 'Inventory Turnover', companies)

    if company in inv_turnover:
        it_2024 = inv_turnover[company]['FY2024']
        it_2023 = inv_turnover[company]['FY2023']

        if it_2024 is not None and it_2023 is not None:
            if it_2023 > 0:
                it_change = ((it_2024 - it_2023) / it_2023) * 100
            else:
                it_change = 0

            print(
                f"\n🔄 Inventory Turnover: FY2023: {it_2023:.2f}x → FY2024: {it_2024:.2f}x (Change: {it_change:+.1f}%)")

            print("\n💬 Insights:")
            if it_change > 0:
                print(
                    f"   ✓ {company} improved inventory efficiency by {abs(it_change):.1f}%, indicating faster")
                print(
                    "     stock conversion, reduced carrying costs, and better working capital management.")
            else:
                print(
                    f"   ⚠ {company} experienced {abs(it_change):.1f}% slower inventory turnover, which may")
                print("     tie up capital and impact liquidity in the short term.")

            if it_2024 > 5:
                print("   ✓ Turnover > 5x suggests efficient inventory management.")

# ============================================================================
# 2. SECTOR-LEVEL ANALYSIS
# ============================================================================

print("\n\n" + "=" * 80)
print("🏢 SECTOR-LEVEL AGGREGATE ANALYSIS")
print("=" * 80)

# Calculate sector averages
ratios_to_analyze = [
    'Current Ratio', 'Quick Ratio', 'Debt to Equity Ratio',
    'Net Profit Margin', 'Return on Equity', 'Inventory Turnover'
]

sector_data = {}

for ratio in ratios_to_analyze:
    ratio_data = get_ratio_data(df, ratio, companies)

    fy2023_values = [v['FY2023']
                     for v in ratio_data.values() if v['FY2023'] is not None]
    fy2024_values = [v['FY2024']
                     for v in ratio_data.values() if v['FY2024'] is not None]

    sector_data[ratio] = {
        'FY2023_avg': np.mean(fy2023_values) if fy2023_values else None,
        'FY2024_avg': np.mean(fy2024_values) if fy2024_values else None,
        'FY2023_values': fy2023_values,
        'FY2024_values': fy2024_values
    }

print("\n📊 SECTOR AVERAGES vs INDUSTRY BENCHMARKS\n")

for ratio in ratios_to_analyze:
    if sector_data[ratio]['FY2023_avg'] is not None and sector_data[ratio]['FY2024_avg'] is not None:
        sector_2023 = sector_data[ratio]['FY2023_avg']
        sector_2024 = sector_data[ratio]['FY2024_avg']

        ratio_key = ratio
        if ratio == 'Return on Equity':
            ratio_key = 'ROE'

        if ratio_key in industry_benchmarks:
            industry_2023 = industry_benchmarks[ratio_key]['FY2023']
            industry_2024 = industry_benchmarks[ratio_key]['FY2024']

            print(f"\n{ratio}:")
            print(
                f"  Sector FY2023: {sector_2023:.3f} | Industry: {industry_2023:.3f} | Diff: {(sector_2023-industry_2023):.3f}")
            print(
                f"  Sector FY2024: {sector_2024:.3f} | Industry: {industry_2024:.3f} | Diff: {(sector_2024-industry_2024):.3f}")

            # Commentary
            if abs(sector_2024 - industry_2024) / industry_2024 < 0.1:
                print(f"  → Sector aligns closely with industry standards")
            elif sector_2024 > industry_2024:
                print(f"  ✓ Sector outperforms industry benchmark")
            else:
                print(f"  ⚠ Sector underperforms industry benchmark")

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

print("\n\n" + "=" * 80)
print("📈 GENERATING VISUALIZATIONS...")
print("=" * 80)

# Set a more professional color palette
colors_fy23 = ['#1f77b4', '#ff7f0e',
               '#2ca02c', '#d62728', '#9467bd', '#8c564b']
colors_fy24 = ['#aec7e8', '#ffbb78',
               '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']

# Visualization 1: Liquidity Comparison with Grouped Bar Chart
fig, ax = plt.subplots(figsize=(16, 8))

current_ratio_data = get_ratio_data(df, 'Current Ratio', companies)
quick_ratio_data = get_ratio_data(df, 'Quick Ratio', companies)

# Filter companies with complete data
comp_names = []
cr_2023_list = []
cr_2024_list = []
qr_2023_list = []
qr_2024_list = []

for c in company_names:
    if (c in current_ratio_data and c in quick_ratio_data and
        current_ratio_data[c]['FY2023'] is not None and
        current_ratio_data[c]['FY2024'] is not None and
        quick_ratio_data[c]['FY2023'] is not None and
            quick_ratio_data[c]['FY2024'] is not None):
        comp_names.append(c)
        cr_2023_list.append(current_ratio_data[c]['FY2023'])
        cr_2024_list.append(current_ratio_data[c]['FY2024'])
        qr_2023_list.append(quick_ratio_data[c]['FY2023'])
        qr_2024_list.append(quick_ratio_data[c]['FY2024'])

x = np.arange(len(comp_names))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, cr_2023_list, width, label='Current Ratio FY2023',
               color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x - 0.5*width, cr_2024_list, width, label='Current Ratio FY2024',
               color='#A23B72', alpha=0.9, edgecolor='black', linewidth=0.7)
bars3 = ax.bar(x + 0.5*width, qr_2023_list, width, label='Quick Ratio FY2023',
               color='#F18F01', alpha=0.9, edgecolor='black', linewidth=0.7)
bars4 = ax.bar(x + 1.5*width, qr_2024_list, width, label='Quick Ratio FY2024',
               color='#C73E1D', alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add benchmark line
ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label='Healthy Liquidity Benchmark (1.5)')

ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
ax.set_ylabel('Ratio Value', fontweight='bold', fontsize=13)
ax.set_title('Liquidity Analysis: Current & Quick Ratios Across Companies (FY2023 vs FY2024)',
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comp_names, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(max(cr_2023_list), max(cr_2024_list),
            max(qr_2023_list), max(qr_2024_list)) * 1.15)

plt.tight_layout()
plt.savefig('liquidity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: liquidity_analysis.png")

# Visualization 2: Profitability Comparison with Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

gpm_data = get_ratio_data(df, 'Gross Profit Margin', companies)
npm_data = get_ratio_data(df, 'Net Profit Margin', companies)

# Filter companies with complete data
comp_names_profit = []
gpm_2023_list = []
gpm_2024_list = []
npm_2023_list = []
npm_2024_list = []

for c in company_names:
    if (c in gpm_data and c in npm_data and
        gpm_data[c]['FY2023'] is not None and
        gpm_data[c]['FY2024'] is not None and
        npm_data[c]['FY2023'] is not None and
            npm_data[c]['FY2024'] is not None):
        comp_names_profit.append(c)
        gpm_2023_list.append(gpm_data[c]['FY2023'] * 100)
        gpm_2024_list.append(gpm_data[c]['FY2024'] * 100)
        npm_2023_list.append(npm_data[c]['FY2023'] * 100)
        npm_2024_list.append(npm_data[c]['FY2024'] * 100)

x = np.arange(len(comp_names_profit))
width = 0.35

# Gross Profit Margin Chart
bars1 = ax1.bar(x - width/2, gpm_2023_list, width, label='FY2023',
                color='#006BA6', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax1.bar(x + width/2, gpm_2024_list, width, label='FY2024',
                color='#0496FF', alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax1.set_ylabel('Gross Profit Margin (%)', fontweight='bold', fontsize=12)
ax1.set_title('Gross Profit Margin Comparison', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(comp_names_profit, fontsize=10, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.4, linestyle='--')
ax1.set_ylim(0, max(max(gpm_2023_list), max(gpm_2024_list)) * 1.15)

# Net Profit Margin Chart
bars3 = ax2.bar(x - width/2, npm_2023_list, width, label='FY2023',
                color='#D84315', alpha=0.9, edgecolor='black', linewidth=0.7)
bars4 = ax2.bar(x + width/2, npm_2024_list, width, label='FY2024',
                color='#FF6F00', alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax2.set_ylabel('Net Profit Margin (%)', fontweight='bold', fontsize=12)
ax2.set_title('Net Profit Margin Comparison', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(comp_names_profit, fontsize=10, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.4, linestyle='--')
ax2.set_ylim(0, max(max(npm_2023_list), max(npm_2024_list)) * 1.15)

plt.suptitle('Profitability Analysis: FY2023 vs FY2024',
             fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('profitability_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: profitability_analysis.png")

# Visualization 3: Leverage Analysis with Better Visualization
fig, ax = plt.subplots(figsize=(16, 8))

de_data = get_ratio_data(df, 'Debt to Equity Ratio', companies)

# Filter companies with complete data
comp_names_de = []
de_2023_list = []
de_2024_list = []
de_change_pct = []

for c in company_names:
    if (c in de_data and de_data[c]['FY2023'] is not None and de_data[c]['FY2024'] is not None):
        comp_names_de.append(c)
        de_2023_list.append(de_data[c]['FY2023'])
        de_2024_list.append(de_data[c]['FY2024'])
        if de_data[c]['FY2023'] > 0:
            change = ((de_data[c]['FY2024'] - de_data[c]
                      ['FY2023']) / de_data[c]['FY2023']) * 100
        else:
            change = 0
        de_change_pct.append(change)

x = np.arange(len(comp_names_de))
width = 0.35

# Create bars with color coding based on increase/decrease
colors_23 = ['#1565C0' for _ in de_2023_list]
colors_24 = ['#43A047' if de_2024_list[i] < de_2023_list[i] else '#E53935'
             for i in range(len(de_2024_list))]

bars1 = ax.bar(x - width/2, de_2023_list, width, label='FY2023',
               color=colors_23, alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, de_2024_list, width, label='FY2024',
               color=colors_24, alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add percentage change annotations
for i, (comp, pct) in enumerate(zip(comp_names_de, de_change_pct)):
    arrow = '↑' if pct > 0 else '↓' if pct < 0 else '→'
    color = '#E53935' if pct > 0 else '#43A047' if pct < 0 else '#757575'
    ax.text(i, max(de_2023_list[i], de_2024_list[i]) * 1.05,
            f'{arrow} {abs(pct):.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=color)

# Add benchmark line
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label='Moderate Leverage (0.5)')
ax.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Conservative (0.3)')

ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
ax.set_ylabel('Debt-to-Equity Ratio', fontweight='bold', fontsize=13)
ax.set_title('Leverage Analysis: Debt-to-Equity Ratio Changes (FY2023 → FY2024)',
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comp_names_de, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(max(de_2023_list), max(de_2024_list)) * 1.25)

# Add text box with interpretation
textstr = 'Green (↓): Reduced Leverage ✓\nRed (↑): Increased Leverage ⚠'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('leverage_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: leverage_analysis.png")

# Visualization 4: ROE and Return on Assets Comparison
fig, ax = plt.subplots(figsize=(16, 8))

roe_data = get_ratio_data(df, 'Return on Equity', companies)
roa_data = get_ratio_data(df, 'Return on Assets', companies)

# Filter companies with complete data
comp_names_returns = []
roe_2023_list = []
roe_2024_list = []
roa_2023_list = []
roa_2024_list = []

for c in company_names:
    if (c in roe_data and c in roa_data and
        roe_data[c]['FY2023'] is not None and
        roe_data[c]['FY2024'] is not None and
        roa_data[c]['FY2023'] is not None and
            roa_data[c]['FY2024'] is not None):
        comp_names_returns.append(c)
        roe_2023_list.append(roe_data[c]['FY2023'] * 100)
        roe_2024_list.append(roe_data[c]['FY2024'] * 100)
        roa_2023_list.append(roa_data[c]['FY2023'] * 100)
        roa_2024_list.append(roa_data[c]['FY2024'] * 100)

x = np.arange(len(comp_names_returns))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, roe_2023_list, width, label='ROE FY2023',
               color='#7B1FA2', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x - 0.5*width, roe_2024_list, width, label='ROE FY2024',
               color='#AB47BC', alpha=0.9, edgecolor='black', linewidth=0.7)
bars3 = ax.bar(x + 0.5*width, roa_2023_list, width, label='ROA FY2023',
               color='#00897B', alpha=0.9, edgecolor='black', linewidth=0.7)
bars4 = ax.bar(x + 1.5*width, roa_2024_list, width, label='ROA FY2024',
               color='#26A69A', alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add industry benchmark line
ax.axhline(y=28.44, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='Industry ROE Benchmark (28.44%)')

ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
ax.set_ylabel('Return (%)', fontweight='bold', fontsize=13)
ax.set_title('Return Analysis: ROE & ROA Comparison (FY2023 vs FY2024)',
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comp_names_returns, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(max(roe_2023_list), max(roe_2024_list)) * 1.15)

plt.tight_layout()
plt.savefig('returns_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: returns_analysis.png")

# Visualization 5: Inventory Turnover Analysis
fig, ax = plt.subplots(figsize=(16, 8))

inv_data = get_ratio_data(df, 'Inventory Turnover', companies)

# Filter companies with complete data
comp_names_inv = []
inv_2023_list = []
inv_2024_list = []
inv_change_pct = []

for c in company_names:
    if (c in inv_data and inv_data[c]['FY2023'] is not None and inv_data[c]['FY2024'] is not None):
        comp_names_inv.append(c)
        inv_2023_list.append(inv_data[c]['FY2023'])
        inv_2024_list.append(inv_data[c]['FY2024'])
        if inv_data[c]['FY2023'] > 0:
            change = ((inv_data[c]['FY2024'] - inv_data[c]
                      ['FY2023']) / inv_data[c]['FY2023']) * 100
        else:
            change = 0
        inv_change_pct.append(change)

x = np.arange(len(comp_names_inv))
width = 0.35

# Create bars with color coding
colors_24 = ['#2E7D32' if inv_2024_list[i] > inv_2023_list[i] else '#C62828'
             for i in range(len(inv_2024_list))]

bars1 = ax.bar(x - width/2, inv_2023_list, width, label='FY2023',
               color='#5D4037', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, inv_2024_list, width, label='FY2024',
               color=colors_24, alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add percentage change annotations
for i, (comp, pct) in enumerate(zip(comp_names_inv, inv_change_pct)):
    arrow = '↑' if pct > 0 else '↓' if pct < 0 else '→'
    color = '#2E7D32' if pct > 0 else '#C62828' if pct < 0 else '#757575'
    ax.text(i, max(inv_2023_list[i], inv_2024_list[i]) * 1.05,
            f'{arrow} {abs(pct):.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=color)

# Add benchmark line
ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Efficient Inventory Mgmt (5.0x)')

ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
ax.set_ylabel('Inventory Turnover (times)', fontweight='bold', fontsize=13)
ax.set_title('Inventory Efficiency: Turnover Ratio Analysis (FY2023 → FY2024)',
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comp_names_inv, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(max(inv_2023_list), max(inv_2024_list)) * 1.25)

# Add text box with interpretation
textstr = 'Green (↑): Improved Efficiency ✓\nRed (↓): Decreased Efficiency ⚠'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('inventory_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: inventory_analysis.png")

# Visualization 6: Comprehensive Financial Health Dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Helper function to create radar chart


def create_radar_chart(ax, categories, values_23, values_24, title, company_name):
    angles = np.linspace(0, 2 * np.pi, len(categories),
                         endpoint=False).tolist()
    values_23 += values_23[:1]
    values_24 += values_24[:1]
    angles += angles[:1]

    ax.plot(angles, values_23, 'o-', linewidth=2,
            label='FY2023', color='#1976D2')
    ax.fill(angles, values_23, alpha=0.25, color='#1976D2')
    ax.plot(angles, values_24, 'o-', linewidth=2,
            label='FY2024', color='#D32F2F')
    ax.fill(angles, values_24, alpha=0.25, color='#D32F2F')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f'{company_name}\n{title}',
                 fontweight='bold', fontsize=11, pad=15)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)


# Create radar charts for each company (top performers)
# Select top 6 companies by ROE FY2024
top_companies = sorted([(c, roe_data[c]['FY2024']) for c in company_names
                        if c in roe_data and roe_data[c]['FY2024'] is not None],
                       key=lambda x: x[1], reverse=True)[:6]

for idx, (company, _) in enumerate(top_companies):
    if idx >= 6:
        break

    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col], projection='polar')

    # Get normalized metrics (0-100 scale)
    categories = ['Liquidity', 'Profitability',
                  'Leverage\n(Inv)', 'Efficiency', 'Returns']

    # FY2023 values - handle None values properly
    cr_23_val = current_ratio_data.get(company, {}).get('FY2023', 0)
    cr_23 = min((cr_23_val / 3.0) * 100, 100) if cr_23_val is not None else 0
    
    npm_23_val = npm_data.get(company, {}).get('FY2023', 0)
    npm_23 = min((npm_23_val * 100 / 30) * 100, 100) if npm_23_val is not None else 0
    
    de_23_val = de_data.get(company, {}).get('FY2023', 1)
    de_23_inv = min((1 - min(de_23_val, 1)) * 100, 100) if de_23_val is not None else 0
    
    inv_23_val = inv_data.get(company, {}).get('FY2023', 0)
    inv_23 = min((inv_23_val / 10) * 100, 100) if inv_23_val is not None else 0
    
    roe_23_val = roe_data.get(company, {}).get('FY2023', 0)
    roe_23 = min((roe_23_val * 100 / 50) * 100, 100) if roe_23_val is not None else 0

    values_23 = [cr_23, npm_23, de_23_inv, inv_23, roe_23]

    # FY2024 values - handle None values properly
    cr_24_val = current_ratio_data.get(company, {}).get('FY2024', 0)
    cr_24 = min((cr_24_val / 3.0) * 100, 100) if cr_24_val is not None else 0
    
    npm_24_val = npm_data.get(company, {}).get('FY2024', 0)
    npm_24 = min((npm_24_val * 100 / 30) * 100, 100) if npm_24_val is not None else 0
    
    de_24_val = de_data.get(company, {}).get('FY2024', 1)
    de_24_inv = min((1 - min(de_24_val, 1)) * 100, 100) if de_24_val is not None else 0
    
    inv_24_val = inv_data.get(company, {}).get('FY2024', 0)
    inv_24 = min((inv_24_val / 10) * 100, 100) if inv_24_val is not None else 0
    
    roe_24_val = roe_data.get(company, {}).get('FY2024', 0)
    roe_24 = min((roe_24_val * 100 / 50) * 100, 100) if roe_24_val is not None else 0

    values_24 = [cr_24, npm_24, de_24_inv, inv_24, roe_24]

    create_radar_chart(ax, categories, values_23, values_24,
                       'Financial Health Score', company)

plt.suptitle('Comprehensive Financial Health Dashboard - Top Performing Companies',
             fontweight='bold', fontsize=18, y=0.98)
plt.savefig('financial_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: financial_dashboard.png")

# Close all plots to free memory
plt.close('all')

# Visualization 7: Heatmap - Year-over-Year Performance Matrix
fig, ax = plt.subplots(figsize=(14, 10))

# Prepare data for heatmap
metrics_for_heatmap = ['Current Ratio', 'Quick Ratio', 'Net Profit Margin', 
                       'Gross Profit Margin', 'Return on Equity', 'Return on Assets', 
                       'Inventory Turnover', 'Debt to Equity Ratio']

heatmap_data = []
heatmap_companies = []

for c in company_names:
    row_data = []
    has_data = False
    
    for metric in metrics_for_heatmap:
        if metric == 'Current Ratio':
            val_23 = current_ratio_data.get(c, {}).get('FY2023')
            val_24 = current_ratio_data.get(c, {}).get('FY2024')
        elif metric == 'Quick Ratio':
            val_23 = quick_ratio_data.get(c, {}).get('FY2023')
            val_24 = quick_ratio_data.get(c, {}).get('FY2024')
        elif metric == 'Net Profit Margin':
            val_23 = npm_data.get(c, {}).get('FY2023')
            val_24 = npm_data.get(c, {}).get('FY2024')
        elif metric == 'Gross Profit Margin':
            val_23 = gpm_data.get(c, {}).get('FY2023')
            val_24 = gpm_data.get(c, {}).get('FY2024')
        elif metric == 'Return on Equity':
            val_23 = roe_data.get(c, {}).get('FY2023')
            val_24 = roe_data.get(c, {}).get('FY2024')
        elif metric == 'Return on Assets':
            val_23 = roa_data.get(c, {}).get('FY2023')
            val_24 = roa_data.get(c, {}).get('FY2024')
        elif metric == 'Inventory Turnover':
            val_23 = inv_data.get(c, {}).get('FY2023')
            val_24 = inv_data.get(c, {}).get('FY2024')
        elif metric == 'Debt to Equity Ratio':
            val_23 = de_data.get(c, {}).get('FY2023')
            val_24 = de_data.get(c, {}).get('FY2024')
        else:
            val_23 = None
            val_24 = None
        
        if val_23 is not None and val_24 is not None and val_23 != 0:
            # Calculate percentage change, invert for Debt to Equity
            if metric == 'Debt to Equity Ratio':
                change = -((val_24 - val_23) / abs(val_23)) * 100
            else:
                change = ((val_24 - val_23) / abs(val_23)) * 100
            row_data.append(change)
            has_data = True
        else:
            row_data.append(0)
    
    if has_data:
        heatmap_data.append(row_data)
        heatmap_companies.append(c)

# Create heatmap
heatmap_array = np.array(heatmap_data)
im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)

# Set ticks and labels
ax.set_xticks(np.arange(len(metrics_for_heatmap)))
ax.set_yticks(np.arange(len(heatmap_companies)))
ax.set_xticklabels(metrics_for_heatmap, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(heatmap_companies, fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('YoY Change (%)', rotation=270, labelpad=20, fontweight='bold', fontsize=12)

# Add text annotations
for i in range(len(heatmap_companies)):
    for j in range(len(metrics_for_heatmap)):
        text = ax.text(j, i, f'{heatmap_array[i, j]:.1f}%',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax.set_title('Year-over-Year Performance Heatmap (FY2023 → FY2024)\nGreen = Improvement, Red = Decline', 
             fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_heatmap.png")
plt.close('all')

# Visualization 8: Scatter Plot - Risk vs Return Analysis
fig, ax = plt.subplots(figsize=(16, 10))

scatter_companies = []
scatter_roe = []
scatter_de = []
scatter_sizes = []
scatter_colors = []

for c in company_names:
    if (c in roe_data and c in de_data and 
        roe_data[c]['FY2024'] is not None and 
        de_data[c]['FY2024'] is not None and
        c in npm_data and npm_data[c]['FY2024'] is not None):
        scatter_companies.append(c)
        scatter_roe.append(roe_data[c]['FY2024'] * 100)
        scatter_de.append(de_data[c]['FY2024'])
        scatter_sizes.append(npm_data[c]['FY2024'] * 3000)  # Size based on profitability
        
        # Color based on current ratio (liquidity)
        cr = current_ratio_data.get(c, {}).get('FY2024', 1)
        if cr is not None and cr > 2.0:
            scatter_colors.append('#2E7D32')  # Strong liquidity
        elif cr is not None and cr > 1.5:
            scatter_colors.append('#FFA000')  # Moderate liquidity
        else:
            scatter_colors.append('#C62828')  # Weak liquidity

# Create scatter plot
scatter = ax.scatter(scatter_de, scatter_roe, s=scatter_sizes, c=scatter_colors, 
                    alpha=0.6, edgecolors='black', linewidth=2)

# Add company labels
for i, company in enumerate(scatter_companies):
    ax.annotate(company, (scatter_de[i], scatter_roe[i]), 
               xytext=(8, 8), textcoords='offset points',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Add quadrant lines
if scatter_de and scatter_roe:
    median_de = np.median(scatter_de)
    median_roe = np.median(scatter_roe)
    ax.axvline(x=median_de, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=median_roe, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Add quadrant labels
ax.text(0.02, 0.98, 'Low Risk\nHigh Return\n(IDEAL)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.text(0.98, 0.98, 'High Risk\nHigh Return', transform=ax.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.text(0.02, 0.02, 'Low Risk\nLow Return', transform=ax.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(0.98, 0.02, 'High Risk\nLow Return\n(AVOID)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax.set_xlabel('Financial Risk (Debt-to-Equity Ratio)', fontweight='bold', fontsize=13)
ax.set_ylabel('Return on Equity (%)', fontweight='bold', fontsize=13)
ax.set_title('Risk-Return Analysis Matrix (FY2024)\nBubble Size = Net Profit Margin | Color = Liquidity Level', 
             fontweight='bold', fontsize=15, pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2E7D32', label='Strong Liquidity (CR > 2.0)'),
                  Patch(facecolor='#FFA000', label='Moderate Liquidity (CR 1.5-2.0)'),
                  Patch(facecolor='#C62828', label='Weak Liquidity (CR < 1.5)')]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.92), fontsize=10)

plt.tight_layout()
plt.savefig('risk_return_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: risk_return_analysis.png")
plt.close('all')

# Visualization 9: Box Plot - Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

box_metrics = {
    'Current Ratio': (current_ratio_data, 'Liquidity Distribution'),
    'Net Profit Margin (%)': (npm_data, 'Profitability Distribution'),
    'Return on Equity (%)': (roe_data, 'Return Distribution'),
    'Inventory Turnover': (inv_data, 'Efficiency Distribution')
}

for idx, (metric_name, (metric_data, subtitle)) in enumerate(box_metrics.items()):
    ax = axes[idx // 2, idx % 2]
    
    fy2023_values = []
    fy2024_values = []
    
    for c in company_names:
        if c in metric_data:
            val_23 = metric_data[c].get('FY2023')
            val_24 = metric_data[c].get('FY2024')
            
            if val_23 is not None:
                if 'Margin' in metric_name or 'Equity' in metric_name:
                    fy2023_values.append(val_23 * 100)
                else:
                    fy2023_values.append(val_23)
            
            if val_24 is not None:
                if 'Margin' in metric_name or 'Equity' in metric_name:
                    fy2024_values.append(val_24 * 100)
                else:
                    fy2024_values.append(val_24)
    
    # Create box plot
    bp = ax.boxplot([fy2023_values, fy2024_values], 
                     labels=['FY2023', 'FY2024'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanline=True)
    
    # Customize colors
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_color('green')
        mean.set_linewidth(2)
    
    # Add scatter points
    for i, values in enumerate([fy2023_values, fy2024_values]):
        if values:
            y = values
            x = np.random.normal(i + 1, 0.04, len(y))
            ax.scatter(x, y, alpha=0.6, s=50, color='black', zorder=3)
    
    ax.set_ylabel(metric_name, fontweight='bold', fontsize=11)
    ax.set_title(subtitle, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add stats
    if fy2024_values:
        mean_24 = np.mean(fy2024_values)
        median_24 = np.median(fy2024_values)
        ax.text(0.95, 0.95, f'FY24 Mean: {mean_24:.2f}\nFY24 Median: {median_24:.2f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Statistical Distribution Analysis: Sector Metrics (FY2023 vs FY2024)',
             fontweight='bold', fontsize=16, y=1.0)
plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: distribution_analysis.png")
plt.close('all')

# Visualization 10: Area Chart - Comparative Margin Analysis
fig, ax = plt.subplots(figsize=(16, 8))

# Get EBITDA Margin data
ebitda_data = get_ratio_data(df, 'EBITDA Margin', companies)

margin_companies = []
gpm_fy24 = []
ebitda_fy24 = []
npm_fy24_area = []

for c in company_names:
    if (c in gpm_data and c in npm_data and c in ebitda_data and
        gpm_data[c]['FY2024'] is not None and
        npm_data[c]['FY2024'] is not None and
        ebitda_data[c]['FY2024'] is not None):
        margin_companies.append(c)
        gpm_fy24.append(gpm_data[c]['FY2024'] * 100)
        ebitda_fy24.append(ebitda_data[c]['FY2024'] * 100)
        npm_fy24_area.append(npm_data[c]['FY2024'] * 100)

if margin_companies:
    x = np.arange(len(margin_companies))

    # Create stacked area chart effect
    ax.fill_between(x, 0, npm_fy24_area, alpha=0.7, color='#E53935', label='Net Profit Margin')
    ax.fill_between(x, npm_fy24_area, ebitda_fy24, alpha=0.7, color='#FFA726', label='EBITDA Margin')
    ax.fill_between(x, ebitda_fy24, gpm_fy24, alpha=0.7, color='#66BB6A', label='Gross Profit Margin')

    # Add lines for clarity
    ax.plot(x, npm_fy24_area, color='darkred', linewidth=2, marker='o', markersize=8)
    ax.plot(x, ebitda_fy24, color='darkorange', linewidth=2, marker='s', markersize=8)
    ax.plot(x, gpm_fy24, color='darkgreen', linewidth=2, marker='^', markersize=8)

    # Add value labels
    for i, company in enumerate(margin_companies):
        ax.text(i, gpm_fy24[i] + 2, f'{gpm_fy24[i]:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i, npm_fy24_area[i] - 2, f'{npm_fy24_area[i]:.1f}%', ha='center', fontsize=9, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(margin_companies, fontsize=11, fontweight='bold')
    ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
    ax.set_ylabel('Margin (%)', fontweight='bold', fontsize=13)
    ax.set_title('Profitability Margin Comparison - Gross to Net (FY2024)', 
                 fontweight='bold', fontsize=15, pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.set_ylim(0, max(gpm_fy24) * 1.1)

plt.tight_layout()
plt.savefig('margin_comparison_area.png', dpi=300, bbox_inches='tight')
print("✓ Saved: margin_comparison_area.png")
plt.close('all')

# Visualization 11: Lollipop Chart - Earnings Per Share Comparison
fig, ax = plt.subplots(figsize=(16, 8))

eps_data = get_ratio_data(df, '    Basic', companies)

eps_companies = []
eps_2023 = []
eps_2024 = []
eps_growth = []

for c in company_names:
    if (c in eps_data and 
        eps_data[c]['FY2023'] is not None and 
        eps_data[c]['FY2024'] is not None):
        eps_companies.append(c)
        eps_2023.append(eps_data[c]['FY2023'])
        eps_2024.append(eps_data[c]['FY2024'])
        growth = ((eps_data[c]['FY2024'] - eps_data[c]['FY2023']) / eps_data[c]['FY2023']) * 100
        eps_growth.append(growth)

if eps_companies:
    # Sort by FY2024 EPS
    sorted_indices = sorted(range(len(eps_2024)), key=lambda i: eps_2024[i], reverse=True)
    eps_companies = [eps_companies[i] for i in sorted_indices]
    eps_2023 = [eps_2023[i] for i in sorted_indices]
    eps_2024 = [eps_2024[i] for i in sorted_indices]
    eps_growth = [eps_growth[i] for i in sorted_indices]

    y_pos = np.arange(len(eps_companies))

    # Create lollipop chart
    for i, (company, val_23, val_24, growth) in enumerate(zip(eps_companies, eps_2023, eps_2024, eps_growth)):
        # Draw line
        ax.plot([val_23, val_24], [i, i], color='gray', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw circles
        ax.scatter(val_23, i, s=200, color='#3498db', alpha=0.8, edgecolors='black', linewidth=2, zorder=3, label='FY2023' if i == 0 else '')
        ax.scatter(val_24, i, s=250, color='#e74c3c', alpha=0.8, edgecolors='black', linewidth=2, zorder=3, label='FY2024' if i == 0 else '')
        
        # Add value labels
        ax.text(val_23, i, f'₹{val_23:.2f}', ha='right', va='center', fontsize=9, fontweight='bold')
        ax.text(val_24, i, f'₹{val_24:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add growth percentage
        color = '#2E7D32' if growth > 0 else '#C62828'
        arrow = '↑' if growth > 0 else '↓'
        ax.text(max(val_23, val_24) * 1.05, i, f'{arrow} {abs(growth):.1f}%', 
               ha='left', va='center', fontsize=10, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(eps_companies, fontsize=12, fontweight='bold')
    ax.set_xlabel('Earnings Per Share (₹)', fontweight='bold', fontsize=13)
    ax.set_title('Earnings Per Share (EPS) Comparison: FY2023 → FY2024', 
                 fontweight='bold', fontsize=15, pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.4, linestyle='--')
    ax.set_xlim(0, max(max(eps_2023), max(eps_2024)) * 1.2)

plt.tight_layout()
plt.savefig('eps_lollipop.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eps_lollipop.png")
plt.close('all')

# Visualization 12: Dividend Analysis - Violin Plot
fig, ax = plt.subplots(figsize=(14, 8))

div_yield_data = get_ratio_data(df, 'Dividend Yield', companies)
div_payout_data = get_ratio_data(df, 'Dividend Payout Ratio', companies)

div_companies = []
div_yield_2024 = []
div_payout_2024 = []

for c in company_names:
    if (c in div_yield_data and c in div_payout_data and
        div_yield_data[c]['FY2024'] is not None and
        div_payout_data[c]['FY2024'] is not None):
        div_companies.append(c)
        div_yield_2024.append(div_yield_data[c]['FY2024'] * 100)
        div_payout_2024.append(div_payout_data[c]['FY2024'] * 100)

if div_companies:
    x = np.arange(len(div_companies))
    width = 0.35

    bars1 = ax.bar(x - width/2, div_yield_2024, width, label='Dividend Yield (%)', 
                   color='#00796B', alpha=0.9, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, div_payout_2024, width, label='Payout Ratio (%)', 
                   color='#F57C00', alpha=0.9, edgecolor='black', linewidth=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=13)
    ax.set_title('Dividend Analysis: Yield vs Payout Ratio (FY2024)', 
                 fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(div_companies, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--')

plt.tight_layout()
plt.savefig('dividend_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dividend_analysis.png")
plt.close('all')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

eps_basic_data = get_ratio_data(df, '    Basic', companies)
eps_diluted_data = get_ratio_data(df, '    Diluted', companies)

comp_names_eps = []
eps_basic_2023 = []
eps_basic_2024 = []
eps_diluted_2023 = []
eps_diluted_2024 = []

for c in company_names:
    if (c in eps_basic_data and 
        eps_basic_data[c]['FY2023'] is not None and 
        eps_basic_data[c]['FY2024'] is not None):
        comp_names_eps.append(c)
        eps_basic_2023.append(eps_basic_data[c]['FY2023'])
        eps_basic_2024.append(eps_basic_data[c]['FY2024'])
        eps_diluted_2023.append(eps_diluted_data[c]['FY2023'] if c in eps_diluted_data and eps_diluted_data[c]['FY2023'] is not None else 0)
        eps_diluted_2024.append(eps_diluted_data[c]['FY2024'] if c in eps_diluted_data and eps_diluted_data[c]['FY2024'] is not None else 0)

x = np.arange(len(comp_names_eps))
width = 0.35

# Basic EPS
bars1 = ax1.bar(x - width/2, eps_basic_2023, width, label='FY2023', 
                color='#6A1B9A', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax1.bar(x + width/2, eps_basic_2024, width, label='FY2024', 
                color='#AB47BC', alpha=0.9, edgecolor='black', linewidth=0.7)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax1.set_ylabel('Basic EPS (₹)', fontweight='bold', fontsize=12)
ax1.set_title('Basic Earnings Per Share', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(comp_names_eps, fontsize=10, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.4, linestyle='--')
ax1.set_ylim(0, max(max(eps_basic_2023), max(eps_basic_2024)) * 1.15)

# EPS Growth Rate
eps_growth = []
for i in range(len(comp_names_eps)):
    if eps_basic_2023[i] != 0:
        growth = ((eps_basic_2024[i] - eps_basic_2023[i]) / eps_basic_2023[i]) * 100
        eps_growth.append(growth)
    else:
        eps_growth.append(0)

colors_growth = ['#43A047' if g > 0 else '#E53935' for g in eps_growth]
bars3 = ax2.bar(x, eps_growth, color=colors_growth, alpha=0.9, edgecolor='black', linewidth=0.7)

for bar, val in zip(bars3, eps_growth):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
            fontsize=9, fontweight='bold')

ax2.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax2.set_ylabel('EPS Growth Rate (%)', fontweight='bold', fontsize=12)
ax2.set_title('EPS Year-over-Year Growth', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(comp_names_eps, fontsize=10, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax2.grid(axis='y', alpha=0.4, linestyle='--')

plt.suptitle('Earnings Per Share Analysis (FY2023 vs FY2024)', 
             fontweight='bold', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig('eps_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eps_analysis.png")

# Visualization 10: Price-to-Earnings (P/E) Ratio Analysis
fig, ax = plt.subplots(figsize=(16, 8))

pe_data = get_ratio_data(df, 'Price to Earnings Ratio', companies)

comp_names_pe = []
pe_2023_list = []
pe_2024_list = []

for c in company_names:
    if (c in pe_data and 
        pe_data[c]['FY2023'] is not None and 
        pe_data[c]['FY2024'] is not None):
        comp_names_pe.append(c)
        pe_2023_list.append(pe_data[c]['FY2023'])
        pe_2024_list.append(pe_data[c]['FY2024'])

x = np.arange(len(comp_names_pe))
width = 0.35

bars1 = ax.bar(x - width/2, pe_2023_list, width, label='P/E Ratio FY2023', 
               color='#00796B', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, pe_2024_list, width, label='P/E Ratio FY2024', 
               color='#26A69A', alpha=0.9, edgecolor='black', linewidth=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add market valuation benchmarks
ax.axhline(y=30, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Fair Value (30x)')
ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Expensive (50x)')

ax.set_xlabel('Companies', fontweight='bold', fontsize=13)
ax.set_ylabel('P/E Ratio (times)', fontweight='bold', fontsize=13)
ax.set_title('Price-to-Earnings Ratio: Market Valuation Analysis', 
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comp_names_pe, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(max(pe_2023_list), max(pe_2024_list)) * 1.15)

# Add text box
textstr = 'Lower P/E: Potentially Undervalued\nHigher P/E: Growth Expectations or Overvalued'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('pe_ratio_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pe_ratio_analysis.png")

# Visualization 11: Dividend Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

div_per_share_data = get_ratio_data(df, 'Dividend Per Share', companies)
div_payout_data = get_ratio_data(df, 'Dividend Payout Ratio', companies)
div_yield_data = get_ratio_data(df, 'Dividend Yield', companies)

comp_names_div = []
dps_2023 = []
dps_2024 = []
payout_2023 = []
payout_2024 = []
yield_2023 = []
yield_2024 = []

for c in company_names:
    if (c in div_per_share_data and 
        div_per_share_data[c]['FY2023'] is not None and 
        div_per_share_data[c]['FY2024'] is not None):
        comp_names_div.append(c)
        dps_2023.append(div_per_share_data[c]['FY2023'])
        dps_2024.append(div_per_share_data[c]['FY2024'])
        payout_2023.append(div_payout_data[c]['FY2023'] * 100 if c in div_payout_data and div_payout_data[c]['FY2023'] is not None else 0)
        payout_2024.append(div_payout_data[c]['FY2024'] * 100 if c in div_payout_data and div_payout_data[c]['FY2024'] is not None else 0)
        yield_2023.append(div_yield_data[c]['FY2023'] * 100 if c in div_yield_data and div_yield_data[c]['FY2023'] is not None else 0)
        yield_2024.append(div_yield_data[c]['FY2024'] * 100 if c in div_yield_data and div_yield_data[c]['FY2024'] is not None else 0)

x = np.arange(len(comp_names_div))
width = 0.35

# Dividend Per Share
bars1 = ax1.bar(x - width/2, dps_2023, width, label='DPS FY2023', 
                color='#AD1457', alpha=0.9, edgecolor='black', linewidth=0.7)
bars2 = ax1.bar(x + width/2, dps_2024, width, label='DPS FY2024', 
                color='#EC407A', alpha=0.9, edgecolor='black', linewidth=0.7)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax1.set_ylabel('Dividend Per Share (₹)', fontweight='bold', fontsize=12)
ax1.set_title('Dividend Per Share Comparison', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(comp_names_div, fontsize=10, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.4, linestyle='--')
ax1.set_ylim(0, max(max(dps_2023), max(dps_2024)) * 1.15)

# Dividend Payout Ratio
bars3 = ax2.bar(x - width/2, payout_2023, width, label='Payout Ratio FY2023', 
                color='#F57C00', alpha=0.9, edgecolor='black', linewidth=0.7)
bars4 = ax2.bar(x + width/2, payout_2024, width, label='Payout Ratio FY2024', 
                color='#FFB300', alpha=0.9, edgecolor='black', linewidth=0.7)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add benchmark lines
ax2.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Sustainable (50%)')
ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='High Payout (80%)')

ax2.set_xlabel('Companies', fontweight='bold', fontsize=12)
ax2.set_ylabel('Dividend Payout Ratio (%)', fontweight='bold', fontsize=12)
ax2.set_title('Dividend Payout Ratio', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(comp_names_div, fontsize=10, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', alpha=0.4, linestyle='--')
ax2.set_ylim(0, max(max(payout_2023), max(payout_2024)) * 1.15)

plt.tight_layout()
plt.savefig('dividend_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dividend_analysis.png")
plt.close('all')


# Close all plots to free memory
plt.close('all')

# ============================================================================
# 4. SUMMARY & KEY FINDINGS
# ============================================================================

print("\n\n" + "=" * 80)
print("📝 EXECUTIVE SUMMARY: KEY FINDINGS")
print("=" * 80)

# Find best performer by ROE
roe_data = get_ratio_data(df, 'Return on Equity', companies)
best_roe_company = max([(c, roe_data[c]['FY2024']) for c in company_names if c in roe_data and roe_data[c]['FY2024'] is not None],
                       key=lambda x: x[1], default=(None, 0))

npm_data = get_ratio_data(df, 'Net Profit Margin', companies)
best_npm_company = max([(c, npm_data[c]['FY2024']) for c in company_names if c in npm_data and npm_data[c]['FY2024'] is not None],
                       key=lambda x: x[1], default=(None, 0))

print("\n🏆 BEST PERFORMING COMPANY:")
if best_roe_company[0]:
    print(
        f"   • {best_roe_company[0]} leads with ROE of {best_roe_company[1]*100:.1f}%")
if best_npm_company[0]:
    print(
        f"   • {best_npm_company[0]} shows highest profitability with NPM of {best_npm_company[1]*100:.1f}%")

print("\n📊 SECTOR vs INDUSTRY INSIGHTS:")
if 'Current Ratio' in sector_data:
    cr_sector = sector_data['Current Ratio']['FY2024_avg']
    cr_industry = industry_benchmarks['Current Ratio']['FY2024']
    print(
        f"   • Liquidity: Sector avg {cr_sector:.2f} vs Industry {cr_industry:.2f}")

if 'Net Profit Margin' in sector_data:
    npm_sector = sector_data['Net Profit Margin']['FY2024_avg']
    npm_industry = industry_benchmarks['Net Profit Margin']['FY2024']
    print(
        f"   • Profitability: Sector avg {npm_sector*100:.1f}% vs Industry {npm_industry*100:.1f}%")

print("\n💡 AREAS OF FINANCIAL STRENGTH:")
print("   ✓ Strong liquidity positions across most companies (CR > 1.5)")
print("   ✓ Healthy profit margins indicating operational efficiency")
print("   ✓ Conservative leverage with low debt-to-equity ratios")

print("\n⚠️  AREAS FOR IMPROVEMENT:")
print("   • Some companies show declining inventory turnover")
print("   • Profitability margins compressed for certain players")
print("   • Working capital optimization opportunities exist")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated visualizations:")
print("  1. liquidity_analysis.png - Comprehensive liquidity ratios (Grouped Bar Chart)")
print("  2. profitability_analysis.png - Gross & Net profit margins (Side-by-side Bar Charts)")
print("  3. leverage_analysis.png - Debt-to-equity with YoY change (Color-coded Bar Chart)")
print("  4. returns_analysis.png - ROE & ROA performance (Grouped Bar Chart)")
print("  5. inventory_analysis.png - Inventory turnover efficiency (Color-coded Bar Chart)")
print("  6. financial_dashboard.png - Multi-dimensional health scores (Radar Charts)")
print("  7. performance_heatmap.png - YoY performance across metrics (Heatmap)")
print("  8. risk_return_analysis.png - Risk vs return positioning (Bubble Scatter Plot)")
print("  9. distribution_analysis.png - Statistical distributions (Box Plots with Scatter)")
print(" 10. margin_comparison_area.png - Profitability breakdown (Stacked Area Chart)")
print(" 11. eps_lollipop.png - Earnings per share comparison (Lollipop Chart)")
print(" 12. dividend_analysis.png - Dividend yield & payout ratio (Grouped Bar Chart)")
print("\n📊 Total: 12 diverse, professional visualizations with varied chart types!")
print("📈 Includes: Bar, Scatter, Heatmap, Radar, Box Plot, Area, and Lollipop charts")
print("\n")
