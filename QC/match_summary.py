import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置 QC 结果目录
qc_results_folder = "qc_results"

# 读取所有 `_qc.csv` 结果文件
qc_files = [f for f in os.listdir(qc_results_folder) if f.endswith("_qc.csv")]

# 遍历所有 QC 结果文件
for file in qc_files:
    file_path = os.path.join(qc_results_folder, file)
    df = pd.read_csv(file_path)
    
    # 确保 `Auto QC` 和 `Manual QC` 存在
    if "Auto QC" in df.columns and "Manual QC" in df.columns:
        total_count = len(df)
        match_count = (df["Auto QC"] == df["Manual QC"]).sum()
        manual_0_auto_1 = ((df["Manual QC"] == 0) & (df["Auto QC"] == 1)).sum()
        manual_1_auto_0 = ((df["Manual QC"] == 1) & (df["Auto QC"] == 0)).sum()

        # 计算百分比
        match_pct = (match_count / total_count) * 100 if total_count > 0 else 0
        manual_0_auto_1_pct = (manual_0_auto_1 / total_count) * 100 if total_count > 0 else 0
        manual_1_auto_0_pct = (manual_1_auto_0 / total_count) * 100 if total_count > 0 else 0

        # 生成柱状图
        plt.figure(figsize=(6, 4))
        categories = ["Match", "Manual 0, Auto 1", "Manual 1, Auto 0"]
        percentages = [match_pct, manual_0_auto_1_pct, manual_1_auto_0_pct]
        colors = ['green', 'red', 'blue']

        plt.bar(categories, percentages, color=colors)
        plt.ylabel("Percentage (%)")
        plt.title(f"QC Comparison for {file.replace('_qc.csv', '')}")

        # 添加百分比标签
        for i, v in enumerate(percentages):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')

        plt.ylim(0, 100)
        plt.tight_layout()

        # 保存图像
        output_img_path = os.path.join(qc_results_folder, f"{file.replace('_qc.csv', '')}_match.png")
        plt.savefig(output_img_path)
        plt.close()

        print(f"Saved plot: {output_img_path}")
