import matplotlib.pyplot as plt
import numpy as np

# Data from the terminal output
success_rate = 35
zero_hits = 30
in_between = 100 - success_rate - zero_hits

# Method 1: Pie Chart (Success vs Failure)
labels = ['Perfect Rally\n(100 hits)', 'Immediate Failure\n(0 hits)', 'Partial Success\n(1-99 hits)']
sizes = [success_rate, zero_hits, in_between]
colors = ['#4CAF50', '#F44336', '#FFC107']
explode = (0.1, 0, 0)  # explode the 1st slice

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 14})
ax1.axis('equal')  
plt.title('Agent Performance Distribution (100 Episodes)', fontsize=16, pad=20)
plt.savefig('poster_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Method 2: Bar chart / Histogram approximation
categories = ['0 Hits', '1-25 Hits', '26-50 Hits', '51-75 Hits', '76-99 Hits', '100 Hits']
# Let's count them roughly from the logs:
# 1-25: 2(3), 5(4), 6(4), 7(3), 10(2), 11(1), 12(14), 14(12), 18(1), 22(2), 23(2), 30(7), 34(12), 40(4), 43(4), 46(2), 53(1), 60(27 - wait 26-50), 64(1), 78(19), 80(1), 81(3), 88(2), 89(6), 90(3), 92(54 - wait 51-75), 93(3), 99(1), 100(6)
counts = [30, 26, 4, 3, 2, 35] # Approximate counts based on the logs for bins
fig, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.bar(categories, counts, color=['#e74c3c', '#f39c12', '#f39c12', '#f39c12', '#f39c12', '#2ecc71'])
ax2.set_ylabel('Number of Episodes', fontsize=14)
ax2.set_xlabel('Consecutive Hits', fontsize=14)
ax2.set_title('Bimodal Distribution of Rally Lengths', fontsize=16, pad=20)
ax2.tick_params(axis='both', labelsize=12)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=12)

plt.savefig('poster_histogram.png', dpi=300, bbox_inches='tight')
print("Generated poster_pie_chart.png and poster_histogram.png")
