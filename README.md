# agentic_datascience_system

面向中文数据分析场景的数据科学智能平台原型。当前版本已经从“单纯的 Streamlit 原型”演进到“FastAPI 后端 + React 工作台 + SQLite 原型存储”的可联调形态，适合继续完善预处理、统计诊断和建模主链路。

## 当前能力

- 数据集上传、预览、历史记录查询
- 字段类型自动识别与人工修正
- 字段字典持久化，支持语义类型、单位、软范围、备注修订
- 原始数据 `raw_dataset` 与预处理结果 `preprocessed_data` 落库
- 预处理任务创建与执行
- EDA、统计检验、统计诊断
- 经典统计建模与轻量机器学习建模
- 任务中心、日志、运行监控
- React 中文工作台接入 FastAPI 真实接口

## 当前技术形态

- 后端：Python + FastAPI
- 原型前端：Streamlit
- 正式前端方向：React + TypeScript + Vite + ECharts
- 数据处理：Pandas、scikit-learn、statsmodels、SciPy
- 存储：SQLite

## 项目结构

```text
statistic_platform/
├─ app/
│  ├─ api/                  # FastAPI 接口
│  ├─ core/                 # 预处理、EDA、统计诊断、建模核心逻辑
│  ├─ schemas/              # 请求与配置模型
│  ├─ services/             # 服务编排层
│  ├─ storage/              # SQLite 持久化
│  └─ ui/                   # Streamlit 原型界面
├─ data/                    # SQLite 数据库与导出产物
├─ docs/                    # 产品与模块文档
├─ frontend/                # React 工作台
└─ tests/                   # 冒烟测试
```

## 环境准备

### Python 环境

```powershell
conda create -n rag310 python=3.10 -y
conda activate rag310
pip install -r requirements.txt
```

### 前端环境

```powershell
cd frontend
npm install
```

## 启动方式

### 1. 启动 FastAPI 后端

请优先使用下面这条命令，不要直接用 `uvicorn app.api.fastapi_app:app ...`，否则在 Conda 环境下可能调用到错误的 `uvicorn` 入口。

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform
conda run -n rag310 python -m uvicorn app.api.fastapi_app:app --reload --host 127.0.0.1 --port 8000
```

接口文档地址：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/health`

### 2. 启动 React 工作台

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform\frontend
npm run dev
```

访问地址：

- `http://127.0.0.1:5173/overview`
- `http://127.0.0.1:5173/workbench`
- `http://127.0.0.1:5173/tasks`
- `http://127.0.0.1:5173/monitoring`

### 3. 启动 Streamlit 原型

Streamlit 现在更适合作为内部调试台或算法验证台，不再是唯一前端。

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform
conda run -n rag310 streamlit run app/ui/streamlit_app.py
```

## 数据库与持久化

默认 SQLite 数据库位置：

- `data/statistic_platform.db`

当前核心表：

- `raw_dataset`
- `preprocessed_data`
- `task_run`
- `task_step_log`
- `dataset_dictionary_override`
- `dataset_dictionary_revision`

其中：

- `raw_dataset` 保存原始导入数据
- `preprocessed_data` 保存预处理后的建模数据版本
- 任务与字典表为后续 agent、审计、重跑和监控提供依据

## 已接入的核心接口

- `GET /api/health`
- `GET /api/overview`
- `GET /api/workbench`
- `GET /api/datasets`
- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}/preview`
- `GET /api/datasets/{dataset_id}/dictionary`
- `PATCH /api/datasets/{dataset_id}/dictionary`
- `POST /api/preprocess/run`
- `GET /api/tasks/summary`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/logs`
- `GET /api/tasks/{task_id}/monitoring`
- `POST /api/modeling/run`

## 测试

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform
conda run -n rag310 python tests\smoke_check.py
conda run -n rag310 python tests\api_smoke_check.py
```

前端检查：

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform\frontend
npm run lint
npm run build
```

## 常见启动问题

### 1. 前端提示 `ECONNREFUSED 127.0.0.1:8000`

说明后端没有正常监听 `8000` 端口。先单独访问：

- `http://127.0.0.1:8000/api/health`

如果访问失败，先重新启动 FastAPI。

### 2. 直接执行 `uvicorn ...` 报环境依赖错乱

这是 Conda 环境里较常见的问题。请改用：

```powershell
conda run -n rag310 python -m uvicorn app.api.fastapi_app:app --reload --host 127.0.0.1 --port 8000
```

如需排查路径：

```powershell
conda run -n rag310 where python
conda run -n rag310 where uvicorn
conda run -n rag310 python -c "import sys, fastapi, uvicorn; print(sys.executable); print(fastapi.__file__); print(uvicorn.__file__)"
```

### 3. 出现 `python-multipart` 相关报错

重新在 `rag310` 环境安装依赖：

```powershell
conda run -n rag310 python -m pip install -r requirements.txt
```

### 4. 端口 `8000` 被占用

Windows 可先检查端口占用：

```powershell
netstat -ano | findstr :8000
```

## 当前开发主线

- 继续完善预处理细粒度策略
- 增强统计诊断与结果解释
- 扩充建模中心
- 逐步将任务执行改造成异步任务模式
- 后续从 SQLite 迁移到 PostgreSQL
- 最终以 React 前端 + Python API + 容器化部署作为正式产品形态
