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