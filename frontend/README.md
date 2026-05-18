# 前端工作台说明

这是当前项目的正式前端方向原型，技术栈为 `React + TypeScript + Vite + ECharts`。它不再只是静态展示页，而是已经承担真实的接口联调、任务创建、日志查看和运行监控职责。

## 当前页面

- `overview`：平台总览
- `workbench`：工作台，包含上传、预览、字段字典、预处理、建模、接口联调
- `tasks`：任务中心
- `monitoring`：运行监控与日志查看

## 当前联调状态

### 已接入真实 FastAPI 的能力

- 总览数据读取
- 工作台数据读取
- 数据集目录读取
- 数据集上传
- 原始数据与建模数据预览
- 字段字典读取与修订
- 预处理任务创建
- 建模任务创建
- 任务中心读取
- 单任务详情、日志、监控读取

### Mock 回退策略

当前前端服务层采用“优先真实接口，失败时部分页面回退 mock”的策略：

- 读取类接口存在回退能力，方便后端暂时不可用时继续看界面
- 写入类接口不回退 mock，包括上传、预处理运行、建模运行

实现位置：

- `src/services/platformService.ts`

## 本地启动

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform\frontend
npm install
npm run dev
```

默认地址：

- `http://127.0.0.1:5173`

## 后端代理

开发环境通过 Vite 代理将 `/api` 转发到：

- `http://127.0.0.1:8000`

配置位置：

- `vite.config.ts`

因此联调时需要先启动后端：

```powershell
cd E:\OneDrive\Desktop\data_science\statistic_platform
conda run -n rag310 python -m uvicorn app.api.fastapi_app:app --reload --host 127.0.0.1 --port 8000
```

## 前端命令

```powershell
npm run dev
npm run build
npm run lint
npm run preview
```

## 当前目录结构

```text
frontend/
├─ src/
│  ├─ components/      # 通用界面组件
│  ├─ pages/           # 页面级组件
│  ├─ services/        # 前端接口访问层
│  ├─ types/           # 类型定义
│  ├─ mocks/           # 本地 mock 数据
│  └─ assets/          # 静态资源
├─ package.json
└─ vite.config.ts
```

## 下一步建议

- 将工作台中的更多分析模块拆成独立页面
- 为预处理、统计诊断、建模增加统一任务表单模型
- 引入更清晰的错误提示与请求状态管理
- 后续在接口稳定后再接入鉴权、角色权限和正式设计系统
