import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成伪数据函数（改进版）
def generate_pseudo_data(base_loss, base_dice, base_iou, decay_rate=0.1, noise_scale=0.02, epochs=50):
    loss = [base_loss * np.exp(-decay_rate * i) + np.random.normal(0, noise_scale) for i in range(epochs)]
    dice = [base_dice * (1 - np.exp(-decay_rate * i * 2)) + np.random.normal(0, noise_scale) for i in range(epochs)]
    iou = [base_iou * (1 - np.exp(-decay_rate * i * 2)) + np.random.normal(0, noise_scale) for i in range(epochs)]
    # 确保值在合理范围内
    loss = np.clip(loss, 0.1, 2.0).tolist()  # 降低最小 Loss 阈值以适应 TransdeepLab-UNet
    dice = np.clip(dice, 0.0, 0.98).tolist()  # 提高最大 Dice 阈值
    iou = np.clip(iou, 0.0, 0.95).tolist()    # 提高最大 IoU 阈值
    return loss, dice, iou

# 生成 TransdeepLab-UNet 的伪数据（优于其他变体）
val_loss_trans, val_dice_trans, val_iou_trans = generate_pseudo_data(
    1.5, 0.90, 0.85, decay_rate=0.18, noise_scale=0.015)  # 快速收敛，最高精度

# 生成五种 U-Net 变体的伪数据（保持原逻辑）
val_loss_residual, val_dice_residual, val_iou_residual = generate_pseudo_data(
    1.7, 0.87, 0.78, decay_rate=0.13, noise_scale=0.02)  # Residual UNet
val_loss_dense, val_dice_dense, val_iou_dense = generate_pseudo_data(
    1.8, 0.91, 0.85, decay_rate=0.09, noise_scale=0.03)  # Dense UNet
val_loss_multires, val_dice_multires, val_iou_multires = generate_pseudo_data(
    1.6, 0.89, 0.82, decay_rate=0.11, noise_scale=0.025)  # MultiRes UNet
val_loss_r2u, val_dice_r2u, val_iou_r2u = generate_pseudo_data(
    1.9, 0.90, 0.84, decay_rate=0.10, noise_scale=0.03)  # R2U-Net
val_loss_efficient, val_dice_efficient, val_iou_efficient = generate_pseudo_data(
    2.0, 0.83, 0.73, decay_rate=0.15, noise_scale=0.015)  # Efficient UNet

# 创建 epoch 列表
epochs = list(range(1, 51))

# 绘制 Val Loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_loss_trans, 'k-', label='TransdeepLab-UNet', linewidth=2.5)  # 突出显示
plt.plot(epochs, val_loss_residual, 'r-', label='Residual UNet')
plt.plot(epochs, val_loss_dense, 'b-', label='Dense UNet')
plt.plot(epochs, val_loss_multires, 'g-', label='MultiRes UNet')
plt.plot(epochs, val_loss_r2u, 'm-', label='R2U-Net')
plt.plot(epochs, val_loss_efficient, 'c-', label='Efficient UNet')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('val_loss_six_unet_variants.png')
plt.show()

# 绘制 Val Dice 曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_dice_trans, 'k-', label='TransdeepLab-UNet', linewidth=2.5)
plt.plot(epochs, val_dice_residual, 'r-', label='Residual UNet')
plt.plot(epochs, val_dice_dense, 'b-', label='Dense UNet')
plt.plot(epochs, val_dice_multires, 'g-', label='MultiRes UNet')
plt.plot(epochs, val_dice_r2u, 'm-', label='R2U-Net')
plt.plot(epochs, val_dice_efficient, 'c-', label='Efficient UNet')
plt.title('Validation Dice Coefficient over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)
plt.savefig('val_dice_six_unet_variants.png')
plt.show()

# 绘制 Val IoU 曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_iou_trans, 'k-', label='TransdeepLab-UNet', linewidth=2.5)
plt.plot(epochs, val_iou_residual, 'r-', label='Residual UNet')
plt.plot(epochs, val_iou_dense, 'b-', label='Dense UNet')
plt.plot(epochs, val_iou_multires, 'g-', label='MultiRes UNet')
plt.plot(epochs, val_iou_r2u, 'm-', label='R2U-Net')
plt.plot(epochs, val_iou_efficient, 'c-', label='Efficient UNet')
plt.title('Validation IoU over Epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU Score')
plt.legend()
plt.grid(True)
plt.savefig('val_iou_six_unet_variants.png')
plt.show()

print("六种 U-Net 变体的指标曲线已分别保存为：")
print("- Validation Loss: val_loss_six_unet_variants.png")
print("- Validation Dice: val_dice_six_unet_variants.png")
print("- Validation IoU: val_iou_six_unet_variants.png")