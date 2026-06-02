import os
import matplotlib.pyplot as plt
from IPython.display import display, Image
# 定义 output_images 文件夹路径
output_dir = "output_images_2"
val_dir = os.path.join(output_dir, "val")

# 查找 epoch49 的图片（batch_0, sample_0）
epoch = 49
batch_idx = 0
sample_idx = 0

# 原始掩码和预测掩码的路径
original_label_mask_path = os.path.join(val_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}_original_label_mask.png")
prediction_mask_path = os.path.join(val_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}_prediction_mask.png")

# 检查文件是否存在
if os.path.exists(original_label_mask_path) and os.path.exists(prediction_mask_path):
    print(f"找到 epoch {epoch} 的图片，正在展示...")
    # 单独显示原始掩码
    plt.figure(figsize=(5, 5))
    original_img = plt.imread(original_label_mask_path)
    plt.imshow(original_img, cmap='gray')
    plt.title(f'Epoch {epoch} Batch {batch_idx} Sample {sample_idx}\nOriginal Label Mask')
    plt.axis('off')
    plt.show()
    print(f"原始掩码路径: {original_label_mask_path}")
    # 单独显示预测掩码
    plt.figure(figsize=(5, 5))
    pred_img = plt.imread(prediction_mask_path)
    plt.imshow(pred_img, cmap='gray')
    plt.title(f'Epoch {epoch} Batch {batch_idx} Sample {sample_idx}\nPrediction Mask')
    plt.axis('off')
    plt.show()
    print(f"预测掩码路径: {prediction_mask_path}")

else:
    print(f"未找到 epoch {epoch} 的图片：")
    print(f"原始掩码路径: {original_label_mask_path}")
    print(f"预测掩码路径: {prediction_mask_path}")
    print("请确认文件是否存在，或检查 batch_idx 和 sample_idx 是否正确。")