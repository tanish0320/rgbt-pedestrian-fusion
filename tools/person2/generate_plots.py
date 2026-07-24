import matplotlib.pyplot as plt
import numpy as np

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_width, fig_height = 8, 5

# Data
configs = ['Masked QFDet', 'RGBOnly (Base)', 'Scale Up (opt1)', 'NMS 0.45 (opt2)', 'NMS 0.60 (opt3)']
val_map = [0.053, 0.072, 0.056, 0.070, 0.073]
val_maps = [0.018, 0.017, 0.028, 0.016, 0.017]
test_map = [0.046, 0.055, 0.051, 0.054, 0.057]
test_maps = [0.020, 0.019, 0.030, 0.019, 0.019]

# --- 1. Bar Chart: Performance Comparison ---
x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, val_map, width, label='Val mAP', color='#3F51B5')
rects2 = ax.bar(x + width/2, val_maps, width, label='Val mAP_S (Tiny)', color='#FF5722')

ax.set_ylabel('mAP / mAP_S', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison (Validation Split)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('plots/val_comparison.png', dpi=300)
plt.close()

# Test comparison
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, test_map, width, label='Test mAP', color='#009688')
rects2 = ax.bar(x + width/2, test_maps, width, label='Test mAP_S (Tiny)', color='#FF9800')

ax.set_ylabel('mAP / mAP_S', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison (Test Split)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('plots/test_comparison.png', dpi=300)
plt.close()

# --- 2. Scatter Plot: Complexity vs. Performance ---
# FLOPs (G) and mAP
flops = [162.86, 130.75, 294.18, 130.75, 130.75]
maps_val = [0.053, 0.072, 0.056, 0.070, 0.073]
labels = ['Masked QFDet', 'RGBOnly (Base)', 'Scale Up (opt1)', 'NMS 0.45 (opt2)', 'NMS 0.60 (opt3)']
colors = ['red', 'green', 'blue', 'orange', 'purple']
sizes = [60.18*8, 36.90*8, 36.90*8, 36.90*8, 36.90*8] # bubble size proportional to parameters

plt.figure(figsize=(9, 6))
for i in range(len(flops)):
    plt.scatter(flops[i], maps_val[i], s=sizes[i], color=colors[i], label=labels[i], alpha=0.7, edgecolors='black', linewidths=1.5)

plt.xlabel('GFLOPs (Input: 640x512 / 960x768)', fontsize=12, fontweight='bold')
plt.ylabel('Validation mAP', fontsize=12, fontweight='bold')
plt.title('Complexity (FLOPs) vs. Accuracy (mAP)\nBubble size represents parameter count (M)', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=True, facecolor='white')

# Annotate points
for i, txt in enumerate(labels):
    plt.annotate(txt, (flops[i], maps_val[i]), xytext=(10, -5), textcoords='offset points', fontsize=10, fontweight='bold')

plt.xlim(100, 320)
plt.ylim(0.045, 0.080)
plt.tight_layout()
plt.savefig('plots/complexity_vs_performance.png', dpi=300)
plt.close()

print("Visualization charts generated successfully.")
