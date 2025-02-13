import pandas as pd

# 读取新上传的 Excel 文件
file_path = "compare.xlsx"

# 读取 Excel（自动检测 sheet 名称）
df = pd.read_excel(file_path, engine="openpyxl")

# 确保 Well ID 和 Well ID Full 存在
if "Well ID" in df.columns and "Well ID Full" in df.columns:
    # 创建一个 Manual QC 列，默认填充 0
    df["Manual QC"] = 0

    # 查找所有 Well ID 中的唯一值
    well_id_set = set(df["Well ID"].dropna().astype(str))

    # 遍历 Well ID Full，检查是否在 Well ID 列中
    df.loc[df["Well ID Full"].astype(str).isin(well_id_set), "Manual QC"] = 1

    # 保存更新后的文件（为 Excel 格式）
    output_file = "compare_updated.xlsx"
    df.to_excel(output_file, index=False, engine="openpyxl")

    # 提供下载路径
    output_file
