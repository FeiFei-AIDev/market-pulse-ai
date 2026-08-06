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

# ---------------- 可选增强：FRED 真实宏观数据（无Key回退代理） ----------------
def fred_series(sid, limit=60):
    """FRED_API_KEY 存在时拉取日频序列（pandas Series），否则/失败返回 None"""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    try:
        import urllib.request
        url = ("https://api.stlouisfed.org/fred/series/observations?series_id=%s"
               "&api_key=%s&file_type=json&sort_order=desc&limit=%d" % (sid, key, limit))
        with urllib.request.urlopen(url, timeout=30) as r:
            obs = json.loads(r.read())["observations"]
        vals = {pd.Timestamp(o["date"]): float(o["value"])
                for o in obs if o["value"] not in (".", "")}
        s = pd.Series(vals).sort_index()
        return s if len(s) >= 25 else None
    except Exception as e:
        print("  FRED 拉取失败（%s），使用代理: %s" % (sid, e))
        return None

# ---------------- 可选增强：LLM 生成解读（无Key回退规则版） ----------------
def _llm_json(system, user_payload, keys):
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    try:
        import urllib.request
        body = json.dumps({
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
        }).encode("utf-8")
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        txt = txt.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            if txt.startswith("json"):
                txt = txt[4:]
        d = json.loads(txt)
        if all(k in d for k in keys):
            return d
        return None
    except Exception as e:
        print("  LLM 调用失败，回退规则版: %s" % e)
        return None

def llm_interpret(payload):
    return _llm_json(
        "你是专业投研助手。根据市场指标JSON生成中文市场解读，严格输出JSON对象，"
        "键为 overall(总体判断2-3句), drivers(主要驱动3条数组), risks(风险提示2条数组), "
        "watch(关注方向3条数组), strategy(策略提示1句)。"
        "字段说明：msi=情绪指数0-100；msi_state/regime=状态标签；spx_chg=标普日涨跌%；"
        "breadth_adv_pct=样本股上涨占比%；vix/tnx=波动率与10Y收益率；"
        "top_sectors/weak_sectors=强弱势板块[名称,热度]；risk_signals=已触发风险。"
        "只依据给定字段，不得声称数据缺失。不提供个股买卖建议，不预测具体点位。",
        payload, ("overall", "drivers", "risks", "watch", "strategy"))

def llm_report(payload):
    return _llm_json(
        "你是专业风险监测助手。根据美股压力指数JSON生成中文周报，严格输出JSON对象，键为 "
        "summary(本期总评1-2句), commentary(周度变化分析2-3段、每段1-3句、用||分隔), "
        "strategy(操作建议1句)。风格冷静克制、善守者先为不可胜。"
        "只依据给定字段，不得声称数据缺失。不提供个股买卖建议，不预测具体点位。",
        payload, ("summary", "commentary", "strategy"))

# ---------------- 采集（批量 + 逐符号重试，抗间歇限流） ----------------
import time

def _align(idx, target):
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    if target.tz is None and idx.tz is not None:
        idx = idx.tz_localize(None)
    if target.tz is not None and idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx

def _fetch_one(sym, target_index, min_rows=200, tries=3):
    for attempt in range(tries):
        try:
            h = yf.Ticker(sym).history(period="2y", auto_adjust=True)
        except Exception:
            h = None
        if h is not None and len(h) >= min_rows:
            h.index = _align(h.index, target_index)
            return h
        time.sleep(2 * (attempt + 1))
    return None

print("[1/4] 下载行情数据（约需 10-30 秒）…")
syms = [s for _, _, s, _ in IDX] + [s for _, s, _ in SECTORS] + BASKET + EXTRA
raw = yf.download(syms, period="2y", auto_adjust=True, progress=False, threads=True)
close_raw = raw["Close"]

# 锚定符号 ^GSPC：批量失败时单独重试
if "^GSPC" not in close_raw.columns or close_raw["^GSPC"].notna().sum() < 200:
    h = _fetch_one("^GSPC", close_raw.index)
    if h is not None:
        close_raw["^GSPC"] = h["Close"].reindex(close_raw.index)
        raw["Volume"]["^GSPC"] = h["Volume"].reindex(close_raw.index)

# 截断尾部未生成的空行（CI 时区下 Yahoo 会返回当日空行）；
# 保留中间日历日索引（BTC 七日交易撑起周末行，滚动分位窗口依赖它）
last_valid = close_raw["^GSPC"].last_valid_index()
if last_valid is not None:
    close_raw = close_raw.loc[:last_valid]
    raw["Volume"] = raw["Volume"].loc[:last_valid]
closes = close_raw.ffill()
volumes = raw["Volume"].replace(0, np.nan).ffill()

last = closes.iloc[-1]
prev = closes.iloc[-2]
as_of = closes.index[-1].strftime("%Y-%m-%d")

REQUIRED = [s for _, _, s, _ in IDX] + [s for _, s, _ in SECTORS] + ["^VIX3M", "DX-Y.NYB", "HYG", "LQD", "SPY"]
miss = [s for s in REQUIRED if s not in closes.columns or closes[s].dropna().empty]
if miss:
    print("  补取缺失符号：%s" % ", ".join(miss))
    for s in miss:
        h = _fetch_one(s, closes.index)
        if h is not None:
            closes[s] = h["Close"].reindex(closes.index).ffill()
            volumes[s] = h["Volume"].reindex(closes.index).replace(0, np.nan).ffill()
    miss = [s for s in REQUIRED if s not in closes.columns or closes[s].dropna().empty]
if miss:
    raise SystemExit("关键符号数据缺失：%s（Yahoo 可能对 CI IP 限流，稍后重试）" % ", ".join(miss))

# 篮子成分股软补取（宽度用；仍缺失则下方中性回退）
for s in [x for x in BASKET if x not in closes.columns or closes[x].dropna().empty]:
    _hb = _fetch_one(s, closes.index, tries=2)
    if _hb is not None:
        closes[s] = _hb["Close"].reindex(closes.index).ffill()

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
if spv.notna().sum() < 100:
    _h = _fetch_one("SPY", closes.index)
    if _h is not None:
        volumes["SPY"] = _h["Volume"].reindex(closes.index).replace(0, np.nan).ffill()
        spv = volumes["SPY"]
if spv.notna().sum() < 100:
    spv = pd.Series(1.0, index=closes.index)                      # 量能缺失→热度中性

ma20, ma50, ma200 = spx.rolling(20).mean(), spx.rolling(50).mean(), spx.rolling(200).mean()
ma_pos = (25 * (spx > ma20) + 25 * (spx > ma50) + 25 * (ma20 > ma50) + 25 * (spx > ma200)).astype(float)
r20 = (spx / spx.shift(20) - 1) * 100
momentum = lin(r20, -8, 8) if False else ((r20 + 8) / 16 * 100).clip(0, 100)

bk_cols = [s for s in BASKET if s in closes.columns and closes[s].notna().sum() > 100]
bk = closes[bk_cols]
if len(bk_cols) >= 10:
    adv_pct = (bk > bk.shift(1)).mean(axis=1) * 100
    rmax = bk.rolling(252).max()
    rmin = bk.rolling(252).min()
    nh = (bk >= rmax * 0.999).sum(axis=1)
    nl = (bk <= rmin * 1.001).sum(axis=1)
    nhnl = (nh / (nh + nl).replace(0, np.nan) * 100).fillna(50)
    up_n = int((bk.iloc[-1] > bk.iloc[-2]).sum())
else:
    print("  宽度样本不足（%d 只），宽度分量取中性 50" % len(bk_cols))
    adv_pct = pd.Series(50.0, index=closes.index)
    nh = pd.Series(0, index=closes.index)
    nl = pd.Series(0, index=closes.index)
    nhnl = pd.Series(50.0, index=closes.index)
    up_n = 0

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
oas = fred_series("BAMLH0A0HYM2", 80)
if oas is not None:
    oas_a = oas.reindex(closes.index, method="ffill")
    dbp_1m = (oas_a - oas_a.shift(21)) * 100                      # bp
    cred_score = (50 - dbp_1m).clip(0, 100)
    cred_src = "FRED HY OAS（真实）"
else:
    cred_1m = (credit / credit.shift(21) - 1) * 100
    cred_score = ((cred_1m + 2) / 4 * 100).clip(0, 100)
    cred_src = "HYG/LQD 价格代理"
dxy_1m = (dxy / dxy.shift(21) - 1) * 100
dxy_score = ((3 - dxy_1m) / 6 * 100).clip(0, 100)
f_macro = 0.4 * rate_score + 0.3 * cred_score + 0.3 * dxy_score

msi = (0.30 * f_trend + 0.25 * f_flow + 0.20 * f_sent + 0.15 * f_vol + 0.10 * f_macro)
msi = msi.dropna()

# ---------------- 压力指数（标普500基准 · 六档风险 L1-L6） ----------------
LV_NAMES = {1: "极低", 2: "偏低", 3: "中性", 4: "警戒", 5: "偏高", 6: "极高"}
LV_BANDS = [(15, 1), (30, 2), (45, 3), (60, 4), (75, 5), (101, 6)]

def level_of(p):
    for hi, lv in LV_BANDS:
        if p < hi:
            return lv
    return 6

spx_dd = (spx / spx.rolling(252).max() - 1) * 100                 # 距一年高点回撤%
dd_score = ((-spx_dd) / 20 * 100).clip(0, 100)                   # 回撤20%→100
below_ma = ((spx < ma20).astype(float) + (spx < ma50) + (spx < ma200))
p_trend = 0.6 * dd_score + 0.4 * (below_ma / 3 * 100)

p_vol = 0.7 * vix_pct + 0.3 * ((1.05 - term_ratio) / 0.15 * 100).clip(0, 100)

nl_share = (nl / (nh + nl).replace(0, np.nan) * 100).fillna(50)
p_breadth = 0.5 * (100 - adv_pct.clip(0, 100)) + 0.5 * nl_share

p_mom = 0.5 * ((5 - r5) / 10 * 100).clip(0, 100) + 0.5 * ((10 - r20) / 20 * 100).clip(0, 100)

if oas is not None:
    p_credit = ((dbp_1m + 20) / 70 * 100).clip(0, 100)           # -20bp→0, +50bp→100
else:
    p_credit = ((cred_1m + 1) / 3 * 100).clip(0, 100)            # -1%→0, +2%→100

pressure = (0.30 * p_trend + 0.25 * p_vol + 0.20 * p_breadth
            + 0.15 * p_mom + 0.10 * p_credit).dropna()

weekly_p = pressure.resample("W-FRI").last().dropna()
p_now = float(pressure.iloc[-1])
p_week_ago = float(pressure.iloc[-6]) if len(pressure) > 6 else p_now
lv_now = level_of(p_now)
dur = 0
for v in reversed(weekly_p.values):
    if level_of(float(v)) == lv_now:
        dur += 1
    else:
        break
weeks_hist = [dict(w=d.strftime("%-m/%-d"), p=round(float(v), 1), lv=level_of(float(v)))
              for d, v in weekly_p.tail(26).items()]

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
adv_now = float(adv_pct.dropna().iloc[-1])
dec_n = len(bk_cols) - up_n
nh_now, nl_now = int(nh.dropna().iloc[-1]), int(nl.dropna().iloc[-1])
vr_now = float(vol_ratio.dropna().iloc[-1])

breadth = dict(adv_pct=round(adv_now, 1), adv=up_n, dec=dec_n, flat=0,
               new_high=nh_now, new_low=nl_now, vol_ratio=round(vr_now * 100),
               scope_note="大盘股样本（30只）估算")

breakdown = [
    dict(name="涨跌情绪", score=int(round(float(adv_score.dropna().iloc[-1]))),
         desc="%.1f%% 样本股上涨，宽度%s" % (adv_now, "健康" if adv_now > 55 else "一般" if adv_now > 45 else "偏弱"),
         color="#0ea5a4"),
    dict(name="成交热度", score=int(round(float(vol_heat.dropna().iloc[-1]))),
         desc="两日均量为20日均量的 %d%%" % round(vr_now * 100), color="#f59e0b"),
    dict(name="资金强度", score=int(round(float(flow_proxy.dropna().iloc[-1]))),
         desc="量价代理估算 · 5日涨幅 %+.1f%%" % float(r5.iloc[-1]), color="#4361ee", est=True),
    dict(name="波动舒适度", score=int(round(float(f_vol.dropna().iloc[-1]))),
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
vix_now, vix_p = float(vix.dropna().iloc[-1]), float(vix_pct.dropna().iloc[-1])
if vix_p < 35:
    risks.append(dict(level="green", label="利好", tag="波动率", title="波动率处于低位",
                      desc="VIX %.1f，处一年 %d%% 分位，市场定价平静" % (vix_now, int(vix_p))))
elif vix_p > 85:
    risks.append(dict(level="red", label="预警", tag="波动率", title="波动率异常抬升",
                      desc="VIX %.1f，处一年 %d%% 分位" % (vix_now, int(vix_p))))
else:
    risks.append(dict(level="yellow", label="关注", tag="波动率", title="波动率中性",
                      desc="VIX %.1f，处一年 %d%% 分位" % (vix_now, int(vix_p))))
if float(vol_ratio.dropna().iloc[-1]) > 1.05 and adv_now > 55:
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

ai_engine = "rules"
_llm = llm_interpret(dict(
    date=as_of, msi=round(msi_now, 1), msi_state=state_of(msi_now),
    regime=regime_label(regime_now), spx_chg=round(spc1, 2),
    breadth_adv_pct=round(adv_now, 1), vix=round(vix_now, 1),
    tnx=round(float(tnx.iloc[-1]), 2),
    top_sectors=[[s["name"], s["score"]] for s in top_sectors],
    weak_sectors=[[s["name"], s["score"]] for s in bot_sectors],
    risk_signals=[r["title"] + "：" + r["desc"] for r in risks if r["level"] != "green"]))
if _llm:
    ai = [_llm]
    ai_engine = "llm"
    print("  AI 解读：LLM 生成")
else:
    print("  AI 解读：规则引擎")

# ---------------- 周度风险报告（六档压力） ----------------
c_trend = lastval(p_trend, "p_trend")
c_volp = lastval(p_vol, "p_vol")
c_breadth = lastval(p_breadth, "p_breadth")
c_momp = lastval(p_mom, "p_mom")
c_credit = lastval(p_credit, "p_credit")
wow = p_now - p_week_ago
bench = float(r5.iloc[-1])
dd_now = float(spx_dd.iloc[-1])
wow_dir = "回升" if wow > 1.5 else "回落" if wow < -1.5 else "持平"
ADVICE = {1: "风险偏好正常，保持标准配置。", 2: "压力偏低，保持标准配置并跟踪趋势。",
          3: "维持中性仓位，等待方向确认。", 4: "维持「警戒」，适度降低高波动敞口。",
          5: "转向防御姿态，提高现金与对冲比例。", 6: "极端压力期，本金安全优先。"}
LV_SUB = {1: "压力指数偏低", 2: "压力指数偏低", 3: "压力指数中性",
          4: "压力指数偏高", 5: "压力指数高位", 6: "压力指数高位"}

rep_summary = ("本期压力指数 %.0f，风险档位 %s（L%d）；较上期%s %.1f，标普500 本周 %+.1f%%。"
               % (p_now, LV_NAMES[lv_now], lv_now, wow_dir, abs(wow), bench))
rep_commentary = ("压力指数较上期%s，距一年高点回撤 %.1f%%，VIX 处一年 %d%% 分位。"
                  % (wow_dir, dd_now, int(float(vix_pct.dropna().iloc[-1])))) \
    + "||" + ("样本宽度 %.0f%% 上涨，周度动能分量 %.0f；%s。"
              % (adv_now, c_momp, "上涨动能趋于均衡" if 40 <= adv_now <= 65 else
                 ("动能偏强" if adv_now > 65 else "动能偏弱"))) \
    + "||" + ("信用分量 %.0f（%s），利率与信用环境%s。"
              % (c_credit, cred_src, "平稳" if c_credit < 55 else "边际收紧"))
rep_strategy = ADVICE[lv_now]

report = dict(
    pressure=round(p_now, 1), level=lv_now, level_name=LV_NAMES[lv_now],
    level_sub=LV_SUB[lv_now], wow=round(wow, 1), wow_dir=wow_dir,
    duration=dur, advice=ADVICE[lv_now], benchmark=round(bench, 2),
    drawdown=round(dd_now, 1),
    components=dict(trend=round(c_trend), vol=round(c_volp), breadth=round(c_breadth),
                    mom=round(c_momp), credit=round(c_credit)),
    weeks=weeks_hist,
    summary=rep_summary, commentary=rep_commentary, strategy=rep_strategy,
    engine="rules")

_llmrep = llm_report(dict(
    date=as_of, pressure=round(p_now, 1), level="L%d %s" % (lv_now, LV_NAMES[lv_now]),
    wow=round(wow, 1), duration_weeks=dur, benchmark_spx=round(bench, 2),
    drawdown_from_1y_high=round(dd_now, 1),
    components=report["components"],
    recent_weeks=[[w["w"], w["p"], "L%d" % w["lv"]] for w in weeks_hist[-8:]]))
if _llmrep:
    report["summary"] = _llmrep["summary"]
    report["commentary"] = _llmrep["commentary"]
    report["strategy"] = _llmrep["strategy"]
    report["engine"] = "llm"
    print("  周报简评：LLM 生成")

# ---------------- 输出 ----------------
print("[3/4] 生成 snapshot.js …")
snapshot = dict(
    mode="real", source="Yahoo Finance (yfinance)", as_of=as_of,
    generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    ai_engine=ai_engine, sources=dict(credit=cred_src),
    msi=round(msi_now, 1), msi_state=state_of(msi_now),
    msi_delta=round(msi_now - msi_prev, 1),
    regime=dict(score=regime_now, label=regime_label(regime_now), confidence=78),
    risk=dict(level=overall, greens=greens, yellows=yellows, reds=reds),
    factors=factors, msi_history=msi_history, indices=indices,
    breadth=breadth, breakdown=breakdown, sectors=sectors, risks=risks, ai=ai,
    report=report,
)
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "snapshot.js"), "w", encoding="utf-8") as f:
    f.write("window.MARKET_DATA = " + json.dumps(snapshot, ensure_ascii=False, indent=1) + ";\n")

print("[4/4] 完成 ✔")
print("  数据截至：%s ｜ 压力 %.0f（L%d %s）｜ MSI %.1f（%s）｜ 风险 %s"
      % (as_of, p_now, lv_now, LV_NAMES[lv_now], msi_now, state_of(msi_now), overall))
print("  输出：data/snapshot.js")
