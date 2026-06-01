import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import warnings
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import zipfile
from google.colab import files

# 手动上传压缩包
uploaded = files.upload()

# 解压 .zip 文件
for filename in uploaded.keys():
    with zipfile.ZipFile(f'/content/{filename}', 'r') as zip_ref:
        zip_ref.extractall('/content/unzipped_folder')

print(f"已解压到 /content/unzipped_folder，请检查文件结构！")

# 检查解压后的目录内容
if os.path.exists('/content/unzipped_folder'):
    print("解压后的目录内容如下：")
    for root, dirs, files in os.walk('/content/unzipped_folder'):
        level = root.replace('/content/unzipped_folder', '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
else:
    print("/content/unzipped_folder 目录不存在，请确认是否已解压。")
warnings.filterwarnings("ignore")

# Data preprocessing transforms
single_channel_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

label_mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def ensure_same_shape(outputs, labels):
    if len(labels.shape) == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if outputs.shape[2:] != labels.shape[1:]:
        outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
    return outputs, labels


def calculate_iou(preds, labels):
    preds = preds.long()
    labels = labels.long()
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        preds = preds[:, 1, :, :]
        labels = labels.float()
        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(n_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.Dropout(p=0.2)
            )
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


# 
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# 
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


# DenseUNet 
class DenseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, growth_rate=12, block_config=(4, 4, 4, 4, 4)):
        super(DenseUNet, self).__init__()
        self.block_config = block_config

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_relu = nn.ReLU(inplace=True)

        # Encoder path
        self.dense_blocks_down = nn.ModuleList()
        self.trans_down = nn.ModuleList()
        num_features = 32
        self.skip_channels = []

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks_down.append(block)
            num_features_out = num_features + num_layers * growth_rate
            self.skip_channels.append(num_features_out)
            if i != len(block_config) - 1:
                trans = TransitionDown(num_features_out, num_features_out)
                self.trans_down.append(trans)
            num_features = num_features_out

        # Decoder path
        self.dense_blocks_up = nn.ModuleList()
        self.trans_up = nn.ModuleList()
        self.skip_channels = self.skip_channels[:-1][::-1]  # Reverse skip channels excluding bottom

        for i, num_layers in enumerate(block_config[:-1][::-1]):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks_up.append(block)
            num_features_out = num_features + num_layers * growth_rate
            if i != len(block_config) - 2:
                trans = TransitionUp(num_features_out, num_features_out // 2)
                self.trans_up.append(trans)
                num_features = num_features_out // 2 + self.skip_channels[i]
            else:
                num_features = num_features_out + self.skip_channels[i]

        # Final convolution
        self.final_conv = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Initial conv
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)

        # Encoder path with skip connections
        skips = []
        for i, (block, trans) in enumerate(zip(self.dense_blocks_down[:-1], self.trans_down)):
            x = block(x)
            skips.append(x)
            x = trans(x)

        # Bottom of U
        x = self.dense_blocks_down[-1](x)
        skips = skips[::-1]

        # Decoder path
        for i, (block, trans) in enumerate(zip(self.dense_blocks_up[:-1], self.trans_up)):
            x = block(x)
            x = trans(x)
            if x.shape[2:] != skips[i].shape[2:]:
                x = F.interpolate(x, size=skips[i].shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skips[i]], dim=1)

        # Final block
        x = self.dense_blocks_up[-1](x)
        if x.shape[2:] != skips[-1].shape[2:]:
            x = F.interpolate(x, size=skips[-1].shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skips[-1]], dim=1)

        return self.final_conv(x)


# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_mask_dir, transform=None, label_mask_transform=None):
        self.image_dir = image_dir
        self.label_mask_dir = label_mask_dir
        self.transform = transform
        self.label_mask_transform = label_mask_transform

        self.image_files = sorted(os.listdir(image_dir))
        self.label_mask_files = [f"mask_overlay_slice_{file_name.split('_')[-1]}" for file_name in self.image_files]

        self.valid_indices = []
        for idx, file_name in enumerate(self.label_mask_files):
            mask_file_path = os.path.join(self.label_mask_dir, file_name)
            if os.path.exists(mask_file_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Mask file {file_name} does not exist. Skipping this file.")

        self.image_files = [self.image_files[i] for i in self.valid_indices]
        self.label_mask_files = [self.label_mask_files[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_mask_path = os.path.join(self.label_mask_dir, self.label_mask_files[idx])
        image = Image.open(image_path).convert('L')
        label_mask = Image.open(label_mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.label_mask_transform:
            label_mask = self.label_mask_transform(label_mask)

        label_mask = label_mask.squeeze(0)
        return image, label_mask, label_mask_path


# Save predictions
def save_predictions(labels, outputs, label_mask_paths, epoch, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(predictions, dim=1, keepdim=True).float()

    for i in range(labels.size(0)):
        label_mask_path = label_mask_paths[i]
        prediction = predictions[i].unsqueeze(0)
        prediction_np = prediction.cpu().numpy().squeeze()
        prediction_np = (prediction_np * 255).astype(np.uint8)
        prediction_mask_pil = Image.fromarray(prediction_np, mode='L')

        original_label_mask_save_path = os.path.join(output_dir,
                                                     f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_original_label_mask.png")
        original_label_mask = Image.open(label_mask_path).convert('L')
        original_label_mask = original_label_mask.resize((256, 256))
        original_label_mask.save(original_label_mask_save_path)

        prediction_mask_save_path = os.path.join(output_dir,
                                                 f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_prediction_mask.png")
        prediction_mask_pil.save(prediction_mask_save_path)


# Training function
def train_model(model, train_loader, val_loader, criterion, dice_loss, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    val_ious = []

    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    output_dir = 'output_images_2'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels, _) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            outputs, labels = ensure_same_shape(outputs, labels)
            ce_loss = criterion(outputs, labels.long())
            dice = dice_loss(outputs, labels)
            loss = ce_loss + dice
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for batch_idx, (images, labels, label_paths) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)")):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                outputs, labels = ensure_same_shape(outputs, labels)
                ce_loss = criterion(outputs, labels.long())
                dice = dice_loss(outputs, labels)
                loss = ce_loss + dice
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                iou = calculate_iou(preds, labels)
                val_iou += iou

                if batch_idx % 10 == 0:
                    save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx,
                                     os.path.join(output_dir, 'val'))

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, val_ious


# Plot metrics
def plot_metrics(train_losses, val_losses, val_ious):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, 'g-', label='Validation IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集实例（Colab 路径）
    train_dataset = SegmentationDataset(
        image_dir="/content/unzipped_folder/dataset/train_label_1",
        label_mask_dir="/content/unzipped_folder/dataset/train_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    val_dataset = SegmentationDataset(
        image_dir="/content/unzipped_folder/dataset/val_oringe_1",
        label_mask_dir="/content/unzipped_folder/dataset/val_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)

    # Initialize DenseUNet model
    model = DenseUNet(in_channels=1, out_channels=2, growth_rate=12, block_config=(4, 4, 4, 4, 4)).to(device)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Train model
    num_epochs = 50
    train_losses, val_losses, val_ious = train_model(
        model, train_loader, val_loader, criterion, dice_loss, optimizer, scheduler, num_epochs, device
    )

    # Plot results
    plot_metrics(train_losses, val_losses, val_ious)