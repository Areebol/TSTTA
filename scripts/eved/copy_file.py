import os
import shutil

def filter_and_copy_with_limit(src_root, dst_root, min_rows, max_files_per_dir=10):
    """
    min_rows: 行数阈值
    max_files_per_dir: 每个子目录下最多拷贝的文件数
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    for root, dirs, files in os.walk(src_root):
        # 筛选出当前目录下的所有 CSV 文件
        csv_files = [f for f in files if f.endswith('.csv')]
        
        copied_count = 0  # 计数器：记录当前文件夹已拷贝的数量
        
        for file in csv_files:
            # 如果当前子文件夹拷贝数量已达上限，跳过剩下的文件
            if copied_count >= max_files_per_dir:
                break
                
            src_file_path = os.path.join(root, file)
            
            # 统计行数
            try:
                with open(src_file_path, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for _ in f)
            except Exception as e:
                print(f"无法读取 {file}: {e}")
                continue

            # 逻辑判断：行数达标则拷贝
            if row_count > min_rows:
                rel_path = os.path.relpath(root, src_root)
                target_dir = os.path.join(dst_root, rel_path)
                
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                shutil.copy2(src_file_path, os.path.join(target_dir, file))
                copied_count += 1
                print(f"[{copied_count}/{max_files_per_dir}] 已拷贝: {file} ({row_count}行)")

# --- 参数设置 ---
SOURCE = '/wenzhiquan/dengzeshuai/codes/TSTTA_COBA/data/segmented_1s_eVED_v9/EV'
TARGET = '/wenzhiquan/dengzeshuai/codes/TSTTA_COBA/data/sved'
ROW_THRESHOLD = 1000  # 行数大于100
MAX_LIMIT = 10       # 每个目录最多10个

if __name__ == "__main__":
    filter_and_copy_with_limit(SOURCE, TARGET, ROW_THRESHOLD, MAX_LIMIT)
    print("\n任务已完成，符合条件的文件已成功同步。")