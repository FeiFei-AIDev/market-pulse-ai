#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Pulse AI — 真实数据采集与计算脚本
数据源：Yahoo Finance（via yfinance，免费、无需 API Key）
用法：python3 fetch_data.py
输出：data/snapshot.js （window.MARKET_DATA = {...}）

说明：
- 价格类指标（指数/波动率/行业ETF/宏观）为真实数据；
- 资金流、期权情绪等无免费源的部分采用量价代理模型，并在 UI 标注"估算"；
- 宽度数据基于 30 只大盘股样本估算，UI 已标注。
"""
import json
import os
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------- 符号配置 ----------------
IDX = [
    ("S&P 500", "SPX", "^GSPC", "price"),
    ("纳斯达克", "IXIC", "^IXIC", "price"),
    ("道琼斯", "DJI", "^DJI", "price"),
    ("罗素2000", "RUT", "^RUT", "price"),
    ("VIX 恐慌指数", "VIX", "^VIX", "vix"),
    ("黄金", "GOLD", "GC=F", "price"),
    ("美债10Y", "US10Y", "^TNX", "yield"),
    ("比特币", "BTC", "BTC-USD", "btc"),
]
SECTORS = [
    ("AI & 云计算", "XLK", ["NVDA", "MSFT", "AVGO", "ORCL", "CRWD"]),
    ("半导体", "SMH", ["NVDA", "TSM", "AMD", "ASML", "MU"]),
    ("通信服务", "XLC", ["GOOGL", "META", "NFLX"]),
    ("医疗保健", "XLV", ["LLY", "UNH", "JNJ"]),
    ("金融", "XLF", ["JPM", "GS", "V"]),
    ("工业", "XLI", ["CAT", "GE", "HON"]),
    ("能源", "XLE", ["XOM", "CVX", "OXY"]),
    ("可选消费", "XLY", ["AMZN", "TSLA", "HD"]),
    ("房地产", "XLRE", ["PLD", "AMT", "EQIX"]),
    ("公用事业", "XLU", ["NEE", "VST", "CEG"]),
]
BASKET = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
          "JPM", "BAC", "WMT", "HD", "DIS", "NFLX", "ADBE", "CRM", "ORCL",
          "CSCO", "PEP", "KO", "JNJ", "UNH", "LLY", "XOM", "CVX", "BA",
          "CAT", "GE", "V", "MA"]
EXTRA = ["^VIX3M", "DX-Y.NYB", "HYG", "LQD", "SPY"]

def clamp(v, a=0.0, b=100.0):
    return float(max(a, min(b, v)))

def lin(v, lo, hi):
    """线性映射 lo→0, hi→100"""
    return clamp((v - lo) / (hi - lo) * 100.0)

# ---------------- 采集 ----------------
print("[1/4] 下载行情数据（约需 10-30 秒）…")
syms = [s for _, _, s, _ in IDX] + [s for _, s, _ in SECTORS] + BASKET + EXTRA
raw = yf.download(syms, period="1y", auto_adjust=True, progress=False, threads=True)
closes = raw["Close"].ffill()
volumes = raw["Volume"].replace(0, np.nan).ffill()

last = closes.iloc[-1]
prev = closes.iloc[-2]
as_of = closes.index[-1].strftime("%Y-%m-%d")

REQUIRED = [s for _, _, s, _ in IDX] + [s for _, s, _ in SECTORS] + ["^VIX3M", "DX-Y.NYB", "HYG", "LQD", "SPY"]
miss = [s for s in REQUIRED if s not in closes.columns or closes[s].dropna().empty]
if miss:
    raise SystemExit("关键符号数据缺失：%s（Yahoo 可能对 CI IP 限流，稍后重试）" % ", ".join(miss))

# ---------------- 指数卡片 ----------------
print("[2/4] 计算指数与行业…")
def spark(sym, n=14):
    return [round(float(x), 2) for x in closes[sym].dropna().tail(n)]

indices = []
for name, sym, ysym, kind in IDX:
    v, p = float(last[ysym]), float(prev[ysym])
    if kind == "yield":
        chg_bp = (v - p) * 100
        indices.append(dict(name=name, sym=sym, val="%.2f%%" % v, chg=round(chg_bp, 1),
                            fmt="%+.0fbp" % chg_bp, neutral=True,
                            note="收益率上行压制成长股估值", spark=spark(ysym)))
    else:
        chg = (v / p - 1) * 100
        if kind == "vix":
            fmt_v = "%.1f" % v
            extra = dict(goodDown=True, note="下跌为积极信号")
        elif kind == "btc":
            fmt_v = "{:,.0f}".format(v)
            extra = {}
        else:
            fmt_v = "{:,.1f}".format(v)
            extra = {}
        indices.append(dict(name=name, sym=sym, val=fmt_v, chg=round(chg, 2),
                            fmt="%+.2f%%" % chg, spark=spark(ysym), **extra))

# ---------------- 因子序列（向量化） ----------------
spx = closes["^GSPC"]
vix = closes["^VIX"]
vix3m = closes["^VIX3M"]
tnx = closes["^TNX"]
dxy = closes["DX-Y.NYB"]
credit = closes["HYG"] / closes["LQD"]
spv = volumes["SPY"] if "SPY" in volumes.columns and volumes["SPY"].notna().sum() > 100 else volumes["^GSPC"]

ma20, ma50, ma200 = spx.rolling(20).mean(), spx.rolling(50).mean(), spx.rolling(200).mean()
ma_pos = (25 * (spx > ma20) + 25 * (spx > ma50) + 25 * (ma20 > ma50) + 25 * (spx > ma200)).astype(float)
r20 = (spx / spx.shift(20) - 1) * 100
momentum = lin(r20, -8, 8) if False else ((r20 + 8) / 16 * 100).clip(0, 100)

bk = closes[BASKET].dropna(axis=1, how="all")
adv_pct = (bk > bk.shift(1)).mean(axis=1) * 100
rmax = bk.rolling(252).max()
rmin = bk.rolling(252).min()
nh = (bk >= rmax * 0.999).sum(axis=1)
nl = (bk <= rmin * 1.001).sum(axis=1)
nhnl = (nh / (nh + nl).replace(0, np.nan) * 100).fillna(50)

f_trend = 0.4 * ma_pos + 0.3 * momentum + 0.3 * nhnl

def roll_pctile(s, w=252):
    return s.rolling(w).apply(lambda x: (x[-1] > x).mean() * 100, raw=True)
vix_pct = roll_pctile(vix)
term_ratio = vix3m / vix
term_score = pd.Series(np.select([term_ratio > 1.05, term_ratio >= 0.95],
                                 [100, 60], default=20), index=term_ratio.index)
f_vol = 0.6 * (100 - vix_pct) + 0.4 * term_score

vol_ratio = spv.rolling(5).mean() / spv.rolling(20).mean()
vol_heat = ((vol_ratio - 0.6) / 0.8 * 100).clip(0, 100)
r5 = (spx / spx.shift(5) - 1) * 100
flow_proxy = (50 + r5 * 25 + (vol_heat - 50) * 0.5).clip(0, 100)   # 量价代理
f_flow = flow_proxy

adv_score = ((adv_pct - 30) / 40 * 100).clip(0, 100)
pc_proxy = ((30 - vix) / 18 * 100).clip(0, 100)                     # 以VIX反向代理期权情绪
f_sent = 0.5 * adv_score + 0.5 * pc_proxy

tnx_1m = (tnx - tnx.shift(21)) * 100                                # bp
rate_score = ((40 - tnx_1m) / 80 * 100).clip(0, 100)
cred_1m = (credit / credit.shift(21) - 1) * 100
cred_score = ((cred_1m + 2) / 4 * 100).clip(0, 100)
dxy_1m = (dxy / dxy.shift(21) - 1) * 100
dxy_score = ((3 - dxy_1m) / 6 * 100).clip(0, 100)
f_macro = 0.4 * rate_score + 0.3 * cred_score + 0.3 * dxy_score

msi = (0.30 * f_trend + 0.25 * f_flow + 0.20 * f_sent + 0.15 * f_vol + 0.10 * f_macro)
msi = msi.dropna()

def state_of(v):
    return "过热" if v >= 85 else "乐观" if v >= 70 else "偏强" if v >= 50 else "谨慎" if v >= 30 else "悲观"

def regime_label(v):
    return "过热" if v >= 85 else "乐观偏强" if v >= 70 else "震荡偏强" if v >= 50 else "谨慎震荡" if v >= 30 else "悲观防御"

def lastval(s, name):
    s = s.dropna()
    if s.empty:
        raise SystemExit("因子序列 %s 为空：Yahoo 数据缺失或被限流，请稍后重试" % name)
    return float(s.iloc[-1])

if len(msi) < 90:
    raise SystemExit("有效历史不足 90 天（%d 天），数据源可能不完整" % len(msi))

cur = {k: lastval(s, k) for k, s in dict(
    trend=f_trend, flow=f_flow, sent=f_sent, vol=f_vol, macro=f_macro, msi=msi).items()}
msi_now = cur["msi"]
msi_prev = float(msi.dropna().iloc[-2])
regime_now = round(0.5 * cur["trend"] + 0.25 * cur["flow"] + 0.25 * cur["vol"])

# 情绪历史（7/30/90）
def hist(n):
    tail = msi.tail(n)
    return dict(values=[round(float(x), 1) for x in tail],
                dates=[d.strftime("%-m/%-d") for d in tail.index])
msi_history = {"7": hist(7), "30": hist(30), "90": hist(90)}

# ---------------- 行业热度 ----------------
sectors = []
for name, sym, tickers in SECTORS:
    c1 = (float(last[sym]) / float(prev[sym]) - 1) * 100
    c5 = (float(last[sym]) / float(closes[sym].iloc[-6]) - 1) * 100
    score = int(round(clamp(50 + 6 * c1 + 2.5 * c5, 5, 98)))
    heat = "极高" if score >= 80 else "高" if score >= 65 else "中" if score >= 45 else "低"
    sent = "非常乐观" if score >= 80 else "偏乐观" if score >= 60 else "中性" if score >= 45 else "偏弱"
    if score >= 75:
        view = "%s 板块动能强劲（1日 %+.1f%% / 5日 %+.1f%%），趋势与资金共振，但需留意拥挤度。" % (name, c1, c5)
    elif score >= 55:
        view = "%s 板块震荡偏强（1日 %+.1f%% / 5日 %+.1f%%），价格结构健康。" % (name, c1, c5)
    elif score >= 45:
        view = "%s 板块方向不明（1日 %+.1f%% / 5日 %+.1f%%），资金观望。" % (name, c1, c5)
    else:
        view = "%s 板块偏弱（1日 %+.1f%% / 5日 %+.1f%%），资金流出压力仍存。" % (name, c1, c5)
    sectors.append(dict(name=name, score=score, chg=round(c1, 2),
                        flow="5D %+.1f%%" % c5, flowLabel="资金动能（估算）",
                        news=heat, sent=sent, view=view, tickers=tickers))

# ---------------- 宽度与拆解 ----------------
adv_now = float(adv_pct.iloc[-1])
up_n = int((bk.iloc[-1] > bk.iloc[-2]).sum())
dec_n = len(BASKET) - up_n
nh_now, nl_now = int(nh.iloc[-1]), int(nl.iloc[-1])
vr_now = float(vol_ratio.iloc[-1])

breadth = dict(adv_pct=round(adv_now, 1), adv=up_n, dec=dec_n, flat=0,
               new_high=nh_now, new_low=nl_now, vol_ratio=round(vr_now * 100),
               scope_note="大盘股样本（30只）估算")

breakdown = [
    dict(name="涨跌情绪", score=int(round(float(adv_score.iloc[-1]))),
         desc="%.1f%% 样本股上涨，宽度%s" % (adv_now, "健康" if adv_now > 55 else "一般" if adv_now > 45 else "偏弱"),
         color="#0ea5a4"),
    dict(name="成交热度", score=int(round(float(vol_heat.iloc[-1]))),
         desc="两日均量为20日均量的 %d%%" % round(vr_now * 100), color="#f59e0b"),
    dict(name="资金强度", score=int(round(float(flow_proxy.iloc[-1]))),
         desc="量价代理估算 · 5日涨幅 %+.1f%%" % float(r5.iloc[-1]), color="#4361ee", est=True),
    dict(name="波动舒适度", score=int(round(float(f_vol.iloc[-1]))),
         desc="VIX %.1f，处一年 %d%% 分位" % (float(vix.iloc[-1]), int(float(vix_pct.iloc[-1]))),
         color="#8b5cf6"),
]

factors = [
    dict(name="趋势", w="30%", score=int(round(cur["trend"])), color="#4361ee"),
    dict(name="资金", w="25%", score=int(round(cur["flow"])), color="#0ea5a4", est=True),
    dict(name="情绪", w="20%", score=int(round(cur["sent"])), color="#f59e0b", est=True),
    dict(name="波动", w="15%", score=int(round(cur["vol"])), color="#8b5cf6"),
    dict(name="宏观", w="10%", score=int(round(cur["macro"])), color="#64748b"),
]

# ---------------- 风险雷达（规则引擎） ----------------
risks = []
vix_now, vix_p = float(vix.iloc[-1]), float(vix_pct.iloc[-1])
if vix_p < 35:
    risks.append(dict(level="green", label="利好", tag="波动率", title="波动率处于低位",
                      desc="VIX %.1f，处一年 %d%% 分位，市场定价平静" % (vix_now, int(vix_p))))
elif vix_p > 85:
    risks.append(dict(level="red", label="预警", tag="波动率", title="波动率异常抬升",
                      desc="VIX %.1f，处一年 %d%% 分位" % (vix_now, int(vix_p))))
else:
    risks.append(dict(level="yellow", label="关注", tag="波动率", title="波动率中性",
                      desc="VIX %.1f，处一年 %d%% 分位" % (vix_now, int(vix_p))))
if float(vol_ratio.iloc[-1]) > 1.05 and adv_now > 55:
    risks.append(dict(level="green", label="利好", tag="资金流", title="量价配合良好",
                      desc="放量上涨，两日均量为20日均量 %d%%" % round(vr_now * 100)))
tnx_wk = float((tnx.iloc[-1] - tnx.iloc[-6]) * 100)
if abs(tnx_wk) > 12:
    risks.append(dict(level="yellow", label="关注", tag="宏观", title="利率快速波动",
                      desc="10Y 美债收益率一周变动 %+.0fbp 至 %.2f%%" % (tnx_wk, float(tnx.iloc[-1]))))
semi_top = max([s for s in sectors if s["name"] in ("半导体", "AI & 云计算")], key=lambda x: x["score"])
if semi_top["score"] >= 88:
    risks.append(dict(level="yellow", label="关注", tag="估值", title="AI/半导体处高位",
                      desc="%s 热度 %d，5日 %+.1f%%，拥挤度偏高" % (semi_top["name"], semi_top["score"],
                            (float(last["SMH"]) / float(closes["SMH"].iloc[-6]) - 1) * 100)))
rut20 = (float(last["^RUT"]) / float(closes["^RUT"].iloc[-21]) - 1) * 100
spx20 = (float(last["^GSPC"]) / float(closes["^GSPC"].iloc[-21]) - 1) * 100
rel = rut20 - spx20
if rel < -4:
    risks.append(dict(level="red", label="预警", tag="结构", title="大小盘分化扩大",
                      desc="罗素2000 相对标普500 的20日超额 %.1f%%，宽度集中于大盘" % rel))
cred_wk = float((credit.iloc[-1] / credit.iloc[-6] - 1) * 100)
if cred_wk < -1:
    risks.append(dict(level="red", label="预警", tag="信用", title="信用利差走阔",
                      desc="HYG/LQD 一周 %.1f%%，信用环境收紧" % cred_wk))
greens = sum(1 for r in risks if r["level"] == "green")
yellows = sum(1 for r in risks if r["level"] == "yellow")
reds = sum(1 for r in risks if r["level"] == "red")
if reds >= 2:
    overall = "高风险"
elif reds == 1 or yellows >= 3:
    overall = "中风险"
elif yellows >= 1:
    overall = "中低风险"
else:
    overall = "低风险"

# ---------------- 规则化 AI 解读 ----------------
top_sectors = sorted(sectors, key=lambda x: -x["score"])[:2]
bot_sectors = sorted(sectors, key=lambda x: x["score"])[:2]
spc1 = float(indices[0]["chg"])
drivers = []
drivers.append("标普500 日涨跌 %+.2f%%，趋势因子得分 %d/100" % (spc1, round(cur["trend"])))
drivers.append("%s、%s 领涨行业（热度 %d / %d）" % (top_sectors[0]["name"], top_sectors[1]["name"],
               top_sectors[0]["score"], top_sectors[1]["score"]))
drivers.append("VIX %.1f（一年 %d%% 分位），波动因子得分 %d/100" % (vix_now, int(vix_p), round(cur["vol"])))
ai_risks = [r["title"] + "：" + r["desc"] for r in risks if r["level"] != "green"] or ["当前无显著风险信号触发"]
watch = ["情绪指数 MSI 当前 %.0f，%s区间上沿为 85（过热警示）" % (msi_now, state_of(msi_now)),
         "10Y 美债收益率走势（当前 %.2f%%）" % float(tnx.iloc[-1]),
         "行业宽度能否从 %s 扩散至弱势板块（%s）" % (top_sectors[0]["name"], bot_sectors[0]["name"])]
strat_map = {"悲观": "市场处于防御状态，控制仓位，关注避险资产与超跌修复。",
             "谨慎": "方向不明，降低操作频率，等待趋势确认。",
             "偏强": "趋势健康，维持正常配置，避免追高单一拥挤板块。",
             "乐观": "情绪乐观，注意板块拥挤度，可用再平衡代替加仓。",
             "过热": "情绪拥挤，警惕回调，考虑降低波动敞口并保留尾部对冲。"}
ai = [dict(
    overall="MSI 情绪指数 %.0f（%s），市场状态「%s」。标普500 %+.2f%%，%d%% 样本股上涨，VIX %.1f。"
            % (msi_now, state_of(msi_now), regime_label(regime_now), spc1, adv_now, vix_now),
    drivers=drivers, risks=ai_risks, watch=watch, strategy=strat_map[state_of(msi_now)])]

# ---------------- 输出 ----------------
print("[3/4] 生成 snapshot.js …")
snapshot = dict(
    mode="real", source="Yahoo Finance (yfinance)", as_of=as_of,
    generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    msi=round(msi_now, 1), msi_state=state_of(msi_now),
    msi_delta=round(msi_now - msi_prev, 1),
    regime=dict(score=regime_now, label=regime_label(regime_now), confidence=78),
    risk=dict(level=overall, greens=greens, yellows=yellows, reds=reds),
    factors=factors, msi_history=msi_history, indices=indices,
    breadth=breadth, breakdown=breakdown, sectors=sectors, risks=risks, ai=ai,
)
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "snapshot.js"), "w", encoding="utf-8") as f:
    f.write("window.MARKET_DATA = " + json.dumps(snapshot, ensure_ascii=False, indent=1) + ";\n")

print("[4/4] 完成 ✔")
print("  数据截至：%s ｜ MSI %.1f（%s）｜ 状态指数 %d（%s）｜ 风险 %s"
      % (as_of, msi_now, state_of(msi_now), regime_now, regime_label(regime_now), overall))
print("  输出：data/snapshot.js")
