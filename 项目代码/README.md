# 校园通知处理 Agent

在 PowerShell 中运行 `python .\notice_agent.py`。粘贴通知，输入空行结束，按提示确认事项。确认保存后，记录追加到 `todos_v2.csv`。

## 训练与评估

使用 Python 3.8 或兼容版本创建虚拟环境，安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\train_classifier.py .\notices.jsonl
.\.venv\Scripts\python.exe .\notice_agent.py
```

数据使用 UTF-8 JSONL，每行一条记录，包含 text（通知正文）、label（考试、竞赛、课程、活动、事务之一）、source（可追溯来源）、group（同源通知的共同编号）和 split（train 或 test）。训练集与测试集均需覆盖五类。同一通知的转发、改写以及同一事件的近似版本应放在同一组。现有45条分类数据划分为30条训练和15条测试；另外5条样例用于开发验收，不参与分类训练或测试。

训练脚本生成 notice_classifier.joblib 和 classification_results.json。结果包含关键词规则及模型在同一测试集上的准确率、宏平均 F1、各类别结果和逐条预测。Agent 默认使用关键词规则。要演示模型分类，在运行命令后加 `--model`。缺少本地模型时回退到规则。请只使用本项目训练得到的模型文件。

日期缺少发布年份或相对时间依据时请人工确认。当前版本按分句提取事项，跨句指代可能需要人工修正。

## 当前验证结果

首轮独立测试共 15 条。关键词规则正确 8 条，宏平均 F1 为 0.531。TF-IDF 模型正确 6 条，宏平均 F1 为 0.314。另在 30 条训练数据上做了三折分组交叉验证，模型候选结果见 classification_cv.json。开发样例 5 条从分类训练和测试中留出，用于验收提取、确认和保存。发布部门与日期来自用户附件，链接未核验。

运行 python compare_models.py 可复现训练集三折交叉验证。它生成 classification_cv.json。

运行 python extend_training.py 可复现新增的全文、标题、线性 SVM 和补集朴素贝叶斯训练集交叉验证。结果见 extended_cv.json。

## 补充功能验证与演示

运行 `python evaluate_boundaries.py` 复现十条人工构造边界回归案例，输出 boundary_results.json。修复前七条通过，修复后十条通过，原结果保留于boundary_results_before.json。这不是随机抽样的真实场景成功率。新增八条措辞检查通过，结果见boundary_variants.json。隐式年份导致日期早于发布日时留空人工确认；相对星期保留钟点；取消表述所在句被跳过并提示核对，不自动删除已有CSV记录。复杂否定与跨句指代仍有局限。

报告附录五的用户实际终端截图使用默认关键词规则模式。`python notice_agent.py --model` 为可选机器学习模式，两种模式应明确区分。完整流程实测.txt 是子进程预置输入的实际运行记录，实验截图中的07和09为用户手动操作后提供的真实截图。

数据来源字段仅记录用户提供的部门信息，不表示外部来源已核验。每次确认保存会追加记录，重复运行请检查重复事项。
