# Market Pulse AI · 市场脉搏智能看板

> 以标普500为基准，把美股整体风险压缩成 **L1–L6 六档压力评级**，每周一页纸讲清"现在处于什么风险状态"；明细版另提供 MSI 情绪评分与七大模块的完整驾驶舱。

**在线演示**：https://feifei-aidev.github.io/market-pulse-ai/market-pulse-dashboard.html （每个美股交易日收盘后自动更新）
**完整明细版**：https://feifei-aidev.github.io/market-pulse-ai/market-pulse-detail.html

![周报截图](docs/screenshot-desktop.png)

## 两页结构

| 页面 | 内容 |
|---|---|
| 主页 · 周度风险报告 | 五统计卡（压力评估/风险档位L1-L6/较上期/持续期/市场基准）· 周度变化 · 26周评估轨迹热图 · 本期简评（LLM或规则引擎） |
| 明细版 · 市场驾驶舱 | 状态总览 · MSI情绪环与因子贡献 · 情绪趋势7/30/90天 · 核心指数 · 市场宽度 · 行业热度地图 · 风险雷达 · AI解读 |

## MSI 情绪评分模型

```
MSI = 30%×趋势 + 25%×资金 + 20%×情绪 + 15%×波动 + 10%×宏观
```

- **趋势**：均线位置（MA20/50/200）+ 20日动量 + 新高新低比
- **资金**：量能热度 + 量价代理（估算，UI 标注"估"）
- **情绪**：大盘股样本涨跌比 + VIX 反向代理（估算，UI 标注"估"）
- **波动**：VIX 一年分位（反向）+ VIX3M/VIX 期限结构
- **宏观**：10Y 利率月变化 + 信用利差（FRED 真实 OAS 或 HYG/LQD 代理）+ 美元月变化

状态映射：0-30 悲观 · 30-50 谨慎 · 50-70 偏强 · 70-85 乐观 · 85-100 过热。
每个分数都可拆解到因子贡献（"为什么是这个分"），这是区别于黑箱评分的关键设计。完整公式见 [market-pulse-plan.md](market-pulse-plan.md)。

## 六档压力指数（主页模型）

```
压力 = 30%×趋势回撤 + 25%×波动 + 20%×宽度 + 15%×动能 + 10%×信用
```

- **趋势回撤**：标普500 距一年高点回撤 + 均线下方计数（MA20/50/200）
- **波动**：VIX 一年分位 + VIX3M/VIX 期限结构
- **宽度**：大盘股样本涨跌比 + 新高/新低占比
- **动能**：标普500 周/月涨幅反向
- **信用**：FRED HY OAS 月变化（无Key时 HYG/LQD 代理）

档位映射：L1 极低 <15 · L2 偏低 <30 · L3 中性 <45 · L4 警戒 <60 · L5 偏高 <75 · L6 极高。
每周采样一次形成 26 周评估轨迹；持续期统计连续处于同档的周数。

## 数据架构与透明度

```
Yahoo Finance (yfinance, 免费无需Key)
        │  每日收盘后 GitHub Actions 运行 fetch_data.py
        ▼
data/snapshot.js  ──►  看板 HTML（真实模式）
        缺失时          └─ 无快照自动回退内置模拟数据（演示模式）
```

- **真实数据**：指数、VIX、行业 ETF、利率、美元、信用代理、MSI 历史
- **透明代理**：资金流与期权情绪无免费源，采用量价代理并在 UI 标注"（估）"
- **可选增强**（在仓库 Secrets 配置后自动启用，不配置则回退）：
  - `FRED_API_KEY`：真实 HY 信用利差（OAS）
  - `OPENAI_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`：LLM 生成每日解读（OpenAI 兼容接口）

## 快速开始

```bash
pip install yfinance pandas numpy
python3 fetch_data.py        # 生成 data/snapshot.js
# 双击 market-pulse-dashboard.html 即可（真实数据模式）
# 删除 data/snapshot.js 则回退模拟数据演示模式
```

自动部署：推送到 GitHub → 开启 Pages → Actions 每交易日 UTC 21:00 自动更新快照并提交。

## 项目结构

```
market-pulse-dashboard.html   主页 · 周度风险报告（六档压力，简洁一页纸）
market-pulse-detail.html      明细版 · 完整市场驾驶舱（真实/模拟双模式）
fetch_data.py                 采集 + MSI/压力指数计算 + 风险规则 + LLM解读生成
market-pulse-plan.md          产品规划设计文档（模型公式/权重/路线图）
data/snapshot.js              数据快照（脚本生成，勿手改）
.github/workflows/            每日自动更新
docs/                         截图
```

## 免责声明

本项目为产品作品集演示，所有输出为基于历史数据的统计判断，不构成投资建议。
