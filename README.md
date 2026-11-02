# Understand KDA

Kimi Delta Attention 中文交互讲解网页：逐通道衰减、Delta Rule 状态更新、Decode / Prefill 优化与 PyPTO 实现思路。

## 本地预览

```bash
python -m http.server 8000
```

打开 http://localhost:8000 。网页无需安装依赖或构建。

## GitHub Pages

在 Settings → Pages → Build and deployment 中选择 GitHub Actions。主分支推送后由 Pages workflow 发布。

示例交互和数据流仅用于教学，不代表 NPU 实测性能。

## 交互微课

[开始 20 分钟微课](https://yushouxixiwo.github.io/Understand_KDA/)

7 节：标准 Attention / FlashAttention → 归一化线性 Attention → Delta Rule → Gated DeltaNet → KDA → Chunk 等价变换 → Decode / Prefill 与 PyPTO 优化。

每节包含目标、公式推导、小维度数值实验、讲解稿、自测和本机学习进度。网页无需登录，不上传学习数据。

公式检查：`node tests/course-math.test.cjs`（在线 Softmax、状态法、Delta 更新、非零初始状态与尾块的 chunk 对照）。部署工作流也会运行这些检查。


首页现在默认从第 1 节开始，保存自测进度不会跳过前置知识。原 KDA 实验室移至 [lab.html](https://yushouxixiwo.github.io/Understand_KDA/lab.html)，旧微课地址 course.html 仍可使用。
