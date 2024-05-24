import rarfile
import zipfile
import os

def rar_to_zip(rar_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开RAR文件
    with rarfile.RarFile(rar_path) as opened_rar:
        # 创建ZIP文件，命名为原RAR文件名，但扩展名为.zip
        zip_path = os.path.join(output_folder, os.path.splitext(os.path.basename(rar_path))[0] + '.zip')
        with zipfile.ZipFile(zip_path, 'w') as myzip:
            # 遍历RAR文件中的每个文件
            for rar_info in opened_rar.infolist():
                # 从RAR文件中提取文件
                extracted_file = opened_rar.extract(rar_info, output_folder)
                # 将提取的文件添加到ZIP文件中
                myzip.write(extracted_file, rar_info.filename)
                # 删除提取出的文件以节省空间
                os.remove(extracted_file)
    print(f"Created ZIP file at: {zip_path}")

# 使用示例
rar_to_zip('MovieRecommendSystem.rar', 'MovieRecommendSystem.zip')
