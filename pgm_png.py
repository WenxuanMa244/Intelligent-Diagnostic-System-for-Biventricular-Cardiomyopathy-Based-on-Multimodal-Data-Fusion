import os
from PIL import Image
from tqdm import tqdm

def convert_pgm_to_png(input_dir, output_dir=None):
    """
    将指定目录下的所有.pgm文件转换为.png格式
    
    Args:
        input_dir: 输入目录，包含.pgm文件的目录
        output_dir: 输出目录，如果不指定则在原目录创建converted_png子目录
    """
    # 如果没有指定输出目录，则在输入目录下创建converted_png子目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'converted_png')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        # 获取相对路径
        rel_path = os.path.relpath(root, input_dir)
        # 在输出目录中创建对应的子目录
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        
        # 遍历所有文件
        for file in tqdm(files, desc=f'处理目录 {rel_path}'):
            if file.lower().endswith('.pgm'):
                # 构建输入和输出文件路径
                input_path = os.path.join(root, file)
                output_path = os.path.join(out_dir, file[:-4] + '.png')
                
                try:
                    # 打开并转换图像
                    with Image.open(input_path) as img:
                        # 转换为RGB模式（如果需要）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # 保存为PNG
                        img.save(output_path, 'PNG')
                except Exception as e:
                    print(f'处理文件 {input_path} 时出错: {str(e)}')

if __name__ == '__main__':
    # 设置输入目录（CroppedYale数据集路径）
    input_dir = r"C:\Users\32572\Downloads\CroppedYale"
    
    # 开始转换
    print('开始转换.pgm文件到.png格式...')
    convert_pgm_to_png(input_dir)
    print('转换完成！')