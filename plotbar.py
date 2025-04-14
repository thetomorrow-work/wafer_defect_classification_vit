import matplotlib.pyplot as plt
import numpy as np

f1_scores = [0.998, 0.997, 0.987, 0.993, 0.995, 0.911, 0.989, 0.991]

# Class names in the order of indices
class_names = ['Center', 'Donut', 'EdgeLoc', 'EdgeRing', 'Loc', 'NearFull', 'Scratch', 'Random']

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_scores, color='teal')
plt.ylim(0.8, 1.05)
plt.title('Per-Class F1 Scores')
plt.xlabel('Defect Type')
plt.ylabel('F1 Score')

# Add F1 score values above bars (optional)
for bar, score in zip(bars, f1_scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{score:.3f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("figures/per_class_f1_named.png", dpi=300)
plt.show()
