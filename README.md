# agentic_datascience_system

面向中文数据分析场景的智能数据科学原型平台。

## 当前能力

- 数据导入与 SQLite 历史回读
- 字段类型自动识别与人工标注
- 数据预处理与字段级规则配置
- EDA 可视化、统计检验与统计诊断
- 任务中心
- 经典统计建模与轻量机器学习建模

## 环境准备

```powershell
conda create -n rag310 python=3.10 -y
conda activate rag310
pip install -r requirements.txt
```

## 启动

```powershell
conda run -n rag310 streamlit run app/ui/streamlit_app.py
```

## 测试

```powershell
conda run -n rag310 python tests\smoke_check.py
```

## 技术栈

- Python
- Streamlit
- Pandas
- scikit-learn
- statsmodels
- SQLite

