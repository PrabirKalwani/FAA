import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# A. LINE GRAPH: 2-Year Stock Performance (Oct 2023 - Oct 2025)
# ============================================================================

# Stock performance data (indexed to 100 as of Oct 2023)
dates = ['Oct 2023', 'Apr 2024', 'Oct 2024', 'Apr 2025', 'Oct 2025']
stock_data = {
    'ITC': [100, 102, 95, 105, 110],
    'HUL': [100, 98, 100, 105, 108],
    'Dabur': [100, 105, 108, 110, 112],
    'Britannia': [100, 108, 112, 118, 125],
    'GCPL': [100, 115, 122, 130, 138],
    'Tata Consumer': [100, 120, 135, 145, 155]
}

# Create figure for stock performance
fig1, ax1 = plt.subplots(figsize=(16, 9))

# Define colors and styles for each company
colors = {
    'ITC': '#1f77b4',
    'HUL': '#ff7f0e', 
    'Dabur': '#2ca02c',
    'Britannia': '#d62728',
    'GCPL': '#9467bd',
    'Tata Consumer': '#8c564b'
}

markers = {
    'ITC': 'o',
    'HUL': 's',
    'Dabur': '^',
    'Britannia': 'D',
    'GCPL': 'v',
    'Tata Consumer': 'p'
}

# Plot each company's performance
for company, performance in stock_data.items():
    ax1.plot(dates, performance, marker=markers[company], 
             color=colors[company], linewidth=2.5, markersize=10,
             label=company, alpha=0.9)
    
    # Add value labels at the end point
    ax1.annotate(f'{performance[-1]}', 
                xy=(len(dates)-1, performance[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', 
                color=colors[company])

# Add benchmark line at 100
ax1.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5,
           label='Baseline (Oct 2023)')

# Formatting
ax1.set_xlabel('Time Period', fontweight='bold', fontsize=14)
ax1.set_ylabel('Indexed Stock Price (Base = 100)', fontweight='bold', fontsize=14)
ax1.set_title('2-Year Stock Performance Comparison\n(Oct 2023 - Oct 2025)', 
             fontweight='bold', fontsize=16, pad=20)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(85, 165)

# Add annotations for key insights
best_performer = max(stock_data.items(), key=lambda x: x[1][-1])
worst_performer = min(stock_data.items(), key=lambda x: x[1][-1])

textstr = f'Best Performer: {best_performer[0]} (+{best_performer[1][-1]-100}%)\n'
textstr += f'Weakest: {worst_performer[0]} (+{worst_performer[1][-1]-100}%)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

# Save stock performance chart
plt.tight_layout()
plt.savefig('stock_performance_chart.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("✅ Chart 1: Stock Performance Line Graph")
print("=" * 80)
print("📊 Saved: stock_performance_chart.png")
print(f"  • Best Stock Performer: {best_performer[0]} (+{best_performer[1][-1]-100}% growth)")
print(f"  • Growth Range: {worst_performer[1][-1]}% to {best_performer[1][-1]}%")
plt.close()

# ============================================================================
# B. PIE CHART: Market Capitalization
# ============================================================================

# Create figure for market cap
fig2, ax2 = plt.subplots(figsize=(14, 10))

# Market cap data (INR Crores)
market_cap_data = {
    'HUL': 591000,
    'ITC': 525000,
    'Britannia': 143000,
    'GCPL': 116000,
    'Tata Consumer': 116000,
    'Dabur': 85000
}

# Calculate percentages
total_cap = sum(market_cap_data.values())
percentages = {k: (v/total_cap)*100 for k, v in market_cap_data.items()}

# Sort by market cap for better visualization
sorted_companies = sorted(market_cap_data.items(), key=lambda x: x[1], reverse=True)
companies = [item[0] for item in sorted_companies]
caps = [item[1] for item in sorted_companies]
pcts = [percentages[company] for company in companies]

# Define colors matching the line graph
pie_colors = [colors[company] for company in companies]

# Create explode effect for top 2 companies
explode = [0.05, 0.03, 0, 0, 0, 0]

# Create pie chart
wedges, texts, autotexts = ax2.pie(caps, labels=companies, autopct='%1.1f%%',
                                     startangle=90, colors=pie_colors,
                                     explode=explode, shadow=True,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Equal aspect ratio ensures that pie is drawn as a circle
ax2.axis('equal')

ax2.set_title('Market Capitalization Distribution\n(October 2025)', 
             fontweight='bold', fontsize=16, pad=20)

# Add legend with market cap values
legend_labels = [f'{company}: ₹{cap:,} Cr ({pcts[i]:.1f}%)' 
                for i, (company, cap) in enumerate(zip(companies, caps))]
ax2.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), 
          fontsize=10, framealpha=0.9)

# Add total market cap annotation
textstr = f'Total Market Cap:\n₹{total_cap:,} Cr'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.text(0.5, -0.15, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='center', 
        bbox=props, fontweight='bold')

# Save market cap chart
plt.tight_layout()
plt.savefig('market_cap_chart.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("✅ Chart 2: Market Capitalization Pie Chart")
print("=" * 80)
print("📊 Saved: market_cap_chart.png")
print(f"  • Market Leader: HUL (₹5,91,000 Cr - 36.5%)")
print(f"  • Total Industry Market Cap: ₹{total_cap:,} Cr")
print(f"  • Top 2 companies control: {percentages['HUL'] + percentages['ITC']:.1f}% of market cap")
print("\n")
plt.close()

print("=" * 80)
print("✅ ALL CHARTS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated files:")
print("  1. stock_performance_chart.png - 2-Year stock performance line graph")
print("  2. market_cap_chart.png - Market capitalization distribution pie chart")
print("\n")
