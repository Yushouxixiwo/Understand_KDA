# Operator Learning Notes

交互式算子与高性能计算学习站，当前包含 Kimi Delta Attention（KDA）和昇腾 CANN 通算融合算子（MC²）两个专题。

## 在线目录

[打开专题学习目录](https://yushouxixiwo.github.io/operator-learning-notes/)

- **KDA 微课**：从 Attention / FlashAttention、线性 Attention 到 Delta Rule、Gated DeltaNet 和 Kimi Delta Attention。
- **KDA 实验室**：交互探索逐通道衰减、状态更新，以及 Decode / Prefill 优化思路。
- **MC² 微课**：理解通信与计算的流水重叠、MoE Dispatch / Combine 应用与性能边界。

## 本地预览

运行 `python -m http.server 8000`，然后打开 http://localhost:8000 。网页无需安装依赖或构建。

## GitHub Pages

在 Settings → Pages → Build and deployment 中选择 GitHub Actions。主分支推送后由 Pages workflow 发布。

## 验证与说明

KDA 微课公式检查：`node tests/course-math.test.cjs`。

交互中的小维度计算和数据流示意仅用于教学，不代表模型质量评测或目标硬件实测性能。MC² 的 API、支持设备与版本请以昇腾官方文档为准。