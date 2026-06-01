import matplotlib.pyplot as plt
import numpy as np

# 采集实验数据

configs = ['Baseline', '+ CBAM', '+ Dice Loss', 'Full Model', '- Skip Conn', 'Lightweight', '+ Focal Loss']
val_losses = [0.65, 0.52, 0.48, 0.38, 0.45, 0.42, 0.50] 
val_ious = [0.60, 0.70, 0.73, 0.82, 0.75, 0.78, 0.72] 

# 绘图
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Model Configuration')
ax1.set_ylabel('Val Loss', color='tab:blue')
ax1.bar(configs, val_losses, color='tab:blue', alpha=0.7, label='Val Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 0.8)

ax2 = ax1.twinx()
ax2.set_ylabel('Val IoU', color='tab:red')
ax2.plot(configs, val_ious, color='tab:red', marker='o', linewidth=2, label='Val IoU')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.0)

plt.title('Impact of Multiple Components in the Ablation Study of TransdeepLab-UNet')
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.xticks(rotation=15)
plt.show()