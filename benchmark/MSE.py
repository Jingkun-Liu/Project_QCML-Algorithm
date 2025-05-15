import pandas as pd
import numpy as np

def calculate_mse(csv_file_path):
  
    # 读取CSV文件，使用空格作为分隔符，跳过可能的非数据行（如列名）
    df = pd.read_csv(csv_file_path)
    
    # 将所有数据展平为一维数组并过滤非浮点数（如列名）
    errors = []
    for row in df.values:
        for item in row:
            try:
                errors.append(float(item))
            except ValueError:
                continue  # 忽略非浮点数的项
    
    # 计算MSE
    mse = np.mean(np.square(errors))
    return mse

# 示例用法
if __name__ == "__main__":
    mse_value = calculate_mse("difference_mackeyglass1500_hybrid_100.csv")
    print(f"MSE值为: {mse_value}")