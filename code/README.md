# 代码说明

- `final_analysis.py`：统一实现问题一至问题四，生成论文中的表格、数值和图1—12。
- `requirements.txt`：程序运行所需的 Python 依赖及版本。
- `generated/results.json`：论文关键结果汇总。
- `generated/tables/`：程序生成的全部结果数据表。
- `generated/figures/`：程序生成的论文图1—12。

## 运行方法

本支撑材料包不重复收录题目给定的原始数据。解压后，请将题目附件
`data.xlsx` 放入以下目录：

```text
A题：机械设备故障检测/data.xlsx
```

在支撑材料根目录依次执行：

```text
python -m pip install -r code/requirements.txt
python code/final_analysis.py
```

程序将在 `generated/` 下重新生成全部结果。运行环境为 Windows 11、
Python 3.13.13；随机试验种子固定为 `20260622`，结果可复现。
