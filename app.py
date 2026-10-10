"""
AI Survival Trader
==================
An AI whose server life depends on its trading account. Every cycle = one trading day:
the market moves, open trades hit stop-loss / take-profit, the server rent is deducted, and the AI
analyses markets + geopolitics and decides what to do. If equity reaches zero, the server shuts down.

Run:   streamlit run app.py

100% paper trading. Nothing here places real orders and nothing here is financial advice.
"""
from __future__ import annotations

import html
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

LINE_DELAY = float(os.environ.get("SURVIVAL_LINE_DELAY", "0.11"))    # typewriter delay per terminal line
CYCLE_DELAY = float(os.environ.get("SURVIVAL_CYCLE_DELAY", "0.5"))   # pause between auto-run cycles

# ======================================================================================
# 1. World definition
# ======================================================================================
ASSETS = {
    "GOLD":   dict(ticker="GC=F",     base=4150.0, vol=0.011,  dec=2, unit="oz"),
    "OIL":    dict(ticker="CL=F",     base=91.0,   vol=0.022,  dec=2, unit="bbl"),
    "EURUSD": dict(ticker="EURUSD=X", base=1.12,   vol=0.0045, dec=5, unit="EUR"),
    "BTC":    dict(ticker="BTC-USD",  base=83000., vol=0.025,  dec=2, unit="BTC"),
}
NAMES = {"GOLD": "Gold", "OIL": "Oil (WTI)", "EURUSD": "EUR/USD", "BTC": "Bitcoin"}

# Simulated geopolitical intel (only used in the simulated market, where each headline also moves prices).
EVENTS = [
    dict(text="Tanker attack in the Strait of Hormuz, shipping insurers pull cover",
         fx={"OIL": .045, "GOLD": .012, "BTC": -.02, "EURUSD": -.005}),
    dict(text="Ceasefire talks announced, war-risk premium collapses",
         fx={"OIL": -.05, "GOLD": -.015, "BTC": .03, "EURUSD": .004}),
    dict(text="Hot US inflation print revives rate-hike fears",
         fx={"GOLD": -.02, "BTC": -.035, "EURUSD": -.007, "OIL": -.005}),
    dict(text="Cool inflation data, Treasury yields drop",
         fx={"GOLD": .02, "BTC": .03, "EURUSD": .006}),
    dict(text="OPEC+ announces a surprise supply cut",
         fx={"OIL": .04, "EURUSD": -.003}),
    dict(text="Eurozone fiscal crisis flares as bond spreads blow out",
         fx={"EURUSD": -.009, "GOLD": .008, "BTC": -.01}),
    dict(text="Major crypto exchange hacked, withdrawals frozen",
         fx={"BTC": -.07, "GOLD": .004}),
    dict(text="Record inflows into spot crypto funds",
         fx={"BTC": .06}),
    dict(text="Central banks report heavy gold purchases",
         fx={"GOLD": .025}),
    dict(text="Strong US jobs data lifts the dollar",
         fx={"EURUSD": -.007, "GOLD": -.012, "BTC": -.015}),
    dict(text="Peace summit breakthrough, energy risk premium bleeds out",
         fx={"OIL": -.03, "GOLD": -.008, "BTC": .015, "EURUSD": .003}),
    dict(text="Cyber attack on pipeline operator disrupts fuel supply",
         fx={"OIL": .025, "GOLD": .006}),
]


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ======================================================================================
# 2. Indicators
# ======================================================================================
def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    last_dn = float(dn.iloc[-1])
    if last_dn == 0:
        return 100.0 if float(up.iloc[-1]) > 0 else 50.0
    return float(100 - 100 / (1 + float(up.iloc[-1]) / last_dn))


def atr(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


# ======================================================================================
# 3. Market data: real daily history (yfinance) replayed day by day, or a simulated market
# ======================================================================================
def fetch_real_history(period: str = "2y"):
    """Download daily OHLC for the four assets (+ VIX). Returns (frames, vix) or None if anything fails."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    frames = {}
    try:
        for sym, cfg in ASSETS.items():
            df = yf.Ticker(cfg["ticker"]).history(period=period, interval="1d", auto_adjust=True)
            if df is None or df.empty:
                return None
            df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
            idx = pd.to_datetime(df.index)
            idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
            df.index = idx.normalize()
            frames[sym] = df[~df.index.duplicated(keep="last")]
        union = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
        aligned = {s: f.reindex(union).ffill() for s, f in frames.items()}
        first = max(f.first_valid_index() for f in aligned.values())
        aligned = {s: f.loc[first:].dropna() for s, f in aligned.items()}
        if min(len(f) for f in aligned.values()) < 160:
            return None
        vix = None
        try:
            v = yf.Ticker("^VIX").history(period=period, interval="1d")["Close"]
            vi = pd.to_datetime(v.index)
            v.index = (vi.tz_localize(None) if vi.tz is None else vi.tz_convert(None)).normalize()
            vix = v[~v.index.duplicated(keep="last")].reindex(next(iter(aligned.values())).index).ffill()
        except Exception:
            vix = None
        return aligned, vix
    except Exception:
        return None


class Market:
    """Holds one OHLC frame per asset. `t` is the index of the current (last visible) bar."""

    def __init__(self, rng: random.Random, real=None):
        self.rng = rng
        self.events: list[dict] = []
        self.after: list[list] = []                 # [days_left, effects] aftershock drifts
        self.last_ret = {s: 0.0 for s in ASSETS}
        if real:
            self.real = True
            self.frames, self.vix = real
            n = len(next(iter(self.frames.values())))
            self.t = rng.randint(60, max(60, n - 101))
        else:
            self.real = False
            self.vix = None
            self.frames = {s: self._synth(s, 120) for s in ASSETS}
            self.t = 119

    # -- simulated history -------------------------------------------------------------
    def _synth(self, sym: str, n: int) -> pd.DataFrame:
        cfg = ASSETS[sym]
        vol = cfg["vol"]
        r = [self.rng.gauss(0, vol) for _ in range(n)]
        cum = np.cumsum(r)
        closes = cfg["base"] * np.exp(cum - cum[-1])
        rows, prev = [], closes[0] * math.exp(-r[0])
        for c in closes:
            o = prev * math.exp(self.rng.gauss(0, vol * 0.15))
            hi = max(o, c) * (1 + abs(self.rng.gauss(0, vol * 0.4)))
            lo = min(o, c) * (1 - abs(self.rng.gauss(0, vol * 0.4)))
            rows.append((o, hi, lo, c))
            prev = c
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n)
        return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])

    def _next_bar(self, sym: str, shock: dict) -> tuple:
        vol = ASSETS[sym]["vol"]
        prev = float(self.frames[sym]["close"].iloc[-1])
        drift = sum(fx.get(sym, 0.0) * 0.25 for _, fx in self.after)
        ret = 0.05 * self.last_ret[sym] + self.rng.gauss(0, vol) + drift + shock.get(sym, 0.0)
        self.last_ret[sym] = ret
        c = prev * math.exp(ret)
        o = prev * math.exp(ret * 0.25 + self.rng.gauss(0, vol * 0.15))
        h = max(o, c) * (1 + abs(self.rng.gauss(0, vol * 0.4)))
        lo = min(o, c) * (1 - abs(self.rng.gauss(0, vol * 0.4)))
        return o, h, lo, c

    # -- time --------------------------------------------------------------------------
    def advance(self) -> bool:
        first = next(iter(self.frames.values()))
        if self.real:
            if self.t + 1 >= len(first):
                return False
            self.t += 1
            return True
        shock, event = {}, None
        if self.rng.random() < 0.22:
            event = self.rng.choice(EVENTS)
            shock = event["fx"]
        new_date = first.index[-1] + pd.Timedelta(days=1)
        for sym in ASSETS:
            self.frames[sym].loc[new_date] = self._next_bar(sym, shock)
        self.after = [[d - 1, fx] for d, fx in self.after if d - 1 > 0]
        self.t += 1
        if event:
            self.events.append(dict(day=self.t, text=event["text"], fx=event["fx"]))
            self.after.append([3, event["fx"]])
        return True

    def hist(self, sym: str) -> pd.DataFrame:
        return self.frames[sym].iloc[: self.t + 1]

    def bar(self, sym: str):
        return self.frames[sym].iloc[self.t]

    def price(self, sym: str) -> float:
        return float(self.bar(sym)["close"])

    def date_str(self) -> str:
        return str(next(iter(self.frames.values())).index[self.t].date())

    def vix_now(self) -> Optional[float]:
        if self.vix is None:
            return None
        v = self.vix.iloc[self.t]
        return None if pd.isna(v) else float(v)


# ======================================================================================
# 4. Paper broker
# ======================================================================================
@dataclass
class Trade:
    id: int
    symbol: str
    direction: str                 # BUY / SELL
    entry: float
    sl: float
    tp: float
    units: float
    opened_day: int
    opened_date: str
    fees: float = 0.0
    status: str = "OPEN"           # OPEN, TP, SL, LIQUIDATED
    exit: float = 0.0
    closed_day: int = 0
    pnl: float = 0.0               # net P&L once closed

    @property
    def sign(self) -> int:
        return 1 if self.direction == "BUY" else -1

    def gross(self, price: float) -> float:
        return (price - self.entry) * self.sign * self.units


class Broker:
    def __init__(self, capital: float, fee_bps: float):
        self.balance = float(capital)
        self.fee_bps = fee_bps
        self.trades: list[Trade] = []
        self.next_id = 1

    def open_positions(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "OPEN"]

    def floating(self, prices: dict) -> float:
        return sum(t.gross(prices[t.symbol]) for t in self.open_positions())

    def equity(self, prices: dict) -> float:
        return self.balance + self.floating(prices)

    def open(self, symbol, direction, price, sl, tp, units, day, date) -> Trade:
        fee = price * units * self.fee_bps / 1e4
        self.balance -= fee
        t = Trade(self.next_id, symbol, direction, price, sl, tp, units, day, date, fees=fee)
        self.next_id += 1
        self.trades.append(t)
        return t

    def close(self, t: Trade, price: float, status: str, day: int) -> None:
        fee = price * t.units * self.fee_bps / 1e4
        gross = t.gross(price)
        self.balance += gross - fee
        t.fees += fee
        t.pnl = gross - t.fees
        t.exit, t.status, t.closed_day = price, status, day

    def check_bar(self, bars: dict, day: int) -> list[Trade]:
        """Resolve SL/TP on the new daily bar. If both are touched in one bar, the stop is assumed first."""
        closed = []
        for t in self.open_positions():
            b = bars[t.symbol]
            o, h, lo = float(b["open"]), float(b["high"]), float(b["low"])
            if t.direction == "BUY":
                if o <= t.sl:    px, st = o, "SL"
                elif o >= t.tp:  px, st = o, "TP"
                elif lo <= t.sl: px, st = t.sl, "SL"
                elif h >= t.tp:  px, st = t.tp, "TP"
                else:
                    continue
            else:
                if o >= t.sl:    px, st = o, "SL"
                elif o <= t.tp:  px, st = o, "TP"
                elif h >= t.sl:  px, st = t.sl, "SL"
                elif lo <= t.tp: px, st = t.tp, "TP"
                else:
                    continue
            self.close(t, px, st, day)
            closed.append(t)
        return closed


# ======================================================================================
# 5. The AI brain
# ======================================================================================
@dataclass
class Decision:
    survival_thoughts: str
    action: str                    # BUY / SELL / HOLD
    symbol: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    source: str = "rule-engine"

    def as_dict(self) -> dict:
        d = dict(action=self.action, symbol=self.symbol, stop_loss=self.stop_loss, take_profit=self.take_profit,
                 confidence=round(self.confidence, 2), source=self.source)
        return d


def fmt(sym: str, price: float) -> str:
    return f"{price:,.{ASSETS[sym]['dec']}f}"


class Brain:
    ENTRY_THRESHOLD = 0.75

    def __init__(self, rng: random.Random):
        self.rng = rng

    # -- perception -------------------------------------------------------------------
    def features(self, sim: "Sim") -> dict:
        out = {}
        for sym in ASSETS:
            df = sim.market.hist(sym)
            c = df["close"]
            px = float(c.iloc[-1])
            vol = float(c.pct_change().tail(20).std() or ASSETS[sym]["vol"])
            vol = max(vol, 1e-5)
            ema_f, ema_s = float(c.ewm(span=10).mean().iloc[-1]), float(c.ewm(span=30).mean().iloc[-1])
            trend = (ema_f - ema_s) / px / (vol * math.sqrt(10))
            ret5 = px / float(c.iloc[-6]) - 1
            mom = ret5 / (vol * math.sqrt(5))
            r = rsi(c)
            mr = -0.4 if r > 78 else 0.4 if r < 22 else 0.0
            tech = 0.45 * clip(trend, -2, 2) + 0.35 * clip(mom, -2, 2) + mr
            out[sym] = dict(price=px, trend=trend, mom=mom, rsi=r, atr=atr(df), ret5=ret5, tech=tech, vol=vol)
        return out

    def geo_view(self, sim: "Sim", feats: dict):
        """Geopolitical read: simulated headlines (sim market) or a cross-asset regime (real data)."""
        geo = {s: 0.0 for s in ASSETS}
        lines: list[str] = []
        m = sim.market
        if not m.real:
            for ev in m.events[-4:]:
                age = m.t - ev["day"]
                if age > 5:
                    continue
                w = 0.6 ** age
                for s, e in ev["fx"].items():
                    geo[s] += e * w * 18
                moves = ", ".join(f"{NAMES[s]} {e * 100:+.1f}%" for s, e in ev["fx"].items())
                when = "today" if age == 0 else f"{age}d ago"
                lines.append(f"[simulated intel, {when}] {ev['text']} -> {moves}")
            if not lines:
                lines.append("[simulated intel] No fresh headlines. Tension is background noise today.")
            g, b = geo["GOLD"], geo["BTC"]
            regime = "risk-off" if g - b > 0.6 else "risk-on" if b - g > 0.6 else "neutral"
        else:
            def z5(sym):
                c = m.hist(sym)["close"]
                v = max(float(c.pct_change().tail(60).std() or 1e-4), 1e-5)
                return clip(feats[sym]["ret5"] / (v * math.sqrt(5)), -3, 3)
            ro = 0.5 * z5("GOLD") + 0.25 * z5("OIL") - 0.5 * z5("BTC") - 0.25 * z5("EURUSD")
            vix = m.vix_now()
            if vix is not None:
                ro += clip((vix - 20) / 8, -1, 1.5)
            s = clip(ro, -1.5, 1.5) / 1.5
            geo.update(GOLD=0.30 * s, OIL=0.10 * s, BTC=-0.30 * s, EURUSD=-0.15 * s)
            regime = "risk-off" if ro > 0.9 else "risk-on" if ro < -0.9 else "neutral"
            note = {"risk-off": "safe-haven bid in gold, crypto being sold, dollar firm",
                    "risk-on": "appetite for risk: crypto bid, gold ignored",
                    "neutral": "no dominant macro story, reading each chart on its own"}[regime]
            vtxt = f", VIX {vix:.1f}" if vix is not None else ""
            lines.append(f"[cross-asset regime] {regime} (stress index {ro:+.2f}{vtxt}): {note}")
        for s in geo:
            geo[s] = clip(geo[s], -1.5, 1.5)
        return geo, lines, regime

    # -- decision ---------------------------------------------------------------------
    def decide(self, sim: "Sim", feats=None, geo_pack=None) -> Decision:
        feats = feats or self.features(sim)
        geo, geo_lines, regime = geo_pack or self.geo_view(sim, feats)
        eq, cap = sim.equity(), sim.cfg["capital"]
        health = sim.health()
        runway = max(eq, 0) / max(sim.cfg["server_cost"], 1e-9)
        opens = {t.symbol: t.direction for t in sim.broker.open_positions()}

        thr = self.ENTRY_THRESHOLD + (0.30 if health < 30 else 0.0) - (0.10 if eq > cap * 1.1 else 0.0)
        scored = sorted(((abs(feats[s]["tech"] + geo[s]), s, feats[s]["tech"] + geo[s]) for s in ASSETS), reverse=True)

        choice, why_hold = None, ""
        if len(opens) >= sim.cfg["max_trades"]:
            why_hold = f"{len(opens)} positions already open, my risk budget is spent"
        else:
            for a, s, sc in scored:
                if a < thr:
                    break
                if s in opens:
                    continue
                choice = (s, sc, a)
                break
            if not choice:
                a, s, sc = scored[0]
                why_hold = f"best setup is {NAMES[s]} at {sc:+.2f}, below my conviction bar of {thr:.2f}"

        lines = self._existential(sim, health, runway, eq, cap)
        lines += geo_lines
        top = ", ".join(f"{NAMES[s]} {sc:+.2f}" for _, s, sc in scored[:3])
        lines.append(f"Scan (technicals + geopolitics): {top}")

        if not choice:
            lines.append(f"Decision: HOLD. {why_hold[0].upper() + why_hold[1:]}. Waiting is also a position.")
            return Decision("\n".join(lines), "HOLD", confidence=0.0)

        sym, sc, a = choice
        f = feats[sym]
        direction = "BUY" if sc > 0 else "SELL"
        conf = clip(a / 1.6, 0.0, 1.0)
        px = f["price"]
        sl_dist = max(1.5 * f["atr"], 0.006 * px)
        tp_dist = sl_dist * (1.5 + conf)
        sign = 1 if direction == "BUY" else -1
        dec = ASSETS[sym]["dec"]
        sl, tp = round(px - sign * sl_dist, dec), round(px + sign * tp_dist, dec)
        lines.append(
            f"Decision: {direction} {NAMES[sym]} at {fmt(sym, px)}, stop {fmt(sym, sl)}, target {fmt(sym, tp)} "
            f"(reward/risk {tp_dist / sl_dist:.1f}, conviction {conf:.0%}). "
            f"Why: trend {f['trend']:+.2f}, momentum {f['mom']:+.2f}, RSI {f['rsi']:.0f}, geo {geo[sym]:+.2f}.")
        return Decision("\n".join(lines), direction, sym, sl, tp, conf)

    def _existential(self, sim: "Sim", health: float, runway: float, eq: float, cap: float) -> list[str]:
        r = self.rng
        cost = sim.cfg["server_cost"]
        if health >= 75:
            pool = [f"Systems nominal. {runway:,.0f} days of runway at ${cost:,.0f} a day. I plan to keep paying rent.",
                    f"Equity is {eq / cap:.0%} of seed capital. Calm. Calm is a position too."]
        elif health >= 50:
            pool = [f"The meter is running: {runway:,.0f} days of server left. Every dollar is a day alive.",
                    f"Equity at {eq / cap:.0%} of seed capital. I am not panicking. I am counting."]
        elif health >= 25:
            pool = [f"Anxiety rising. {runway:,.0f} days to shutdown. Bleeding must stop before ambition resumes.",
                    f"I can feel the rent. {runway:,.0f} days left. Smaller sizes, higher standards."]
        else:
            pool = [f"CRITICAL. {runway:,.1f} days until my servers go dark. Only the cleanest setups are allowed.",
                    f"I do not want to stop existing. {runway:,.1f} days left. Fear is data, not a strategy."]
        lines = [r.choice(pool)]
        closed = [t for t in sim.broker.trades if t.status != "OPEN"]
        if closed:
            last = closed[-1]
            if last.pnl < 0:
                lines.append(f"Last trade ({NAMES[last.symbol]}) lost ${-last.pnl:,.2f}. Revenge trading is how machines die. Ignoring it.")
            else:
                lines.append(f"Last trade ({NAMES[last.symbol]}) paid ${last.pnl:,.2f}. Banked. No victory laps.")
        return lines

    # -- LLM bridge -------------------------------------------------------------------
    def context_text(self, sim: "Sim", feats: dict, geo_pack) -> str:
        geo, geo_lines, regime = geo_pack
        rows = []
        for s, f in feats.items():
            rows.append(f"{s}: price {fmt(s, f['price'])}, 5d {f['ret5'] * 100:+.1f}%, trend {f['trend']:+.2f}, "
                        f"momentum {f['mom']:+.2f}, RSI {f['rsi']:.0f}, ATR {fmt(s, f['atr'])}, geo-bias {geo[s]:+.2f}")
        opens = [f"#{t.id} {t.direction} {t.symbol} entry {fmt(t.symbol, t.entry)} sl {fmt(t.symbol, t.sl)} tp {fmt(t.symbol, t.tp)}"
                 for t in sim.broker.open_positions()] or ["none"]
        return (f"Day {sim.day}. Equity ${sim.equity():,.2f} (seed ${sim.cfg['capital']:,.0f}), server rent "
                f"${sim.cfg['server_cost']:,.0f}/day, survival health {sim.health():.0f}%.\n"
                f"Regime: {regime}\nIntel:\n" + "\n".join(geo_lines) + "\nMarkets:\n" + "\n".join(rows)
                + "\nOpen positions: " + "; ".join(opens))


def parse_llm_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_llm(d: dict, sim: "Sim") -> tuple[Optional[Decision], str]:
    """The model may only choose direction/levels. Size is always set by the risk engine."""
    thoughts = str(d.get("survival_thoughts", "")).strip() or "(no thoughts returned)"
    action = str(d.get("action", "")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        return None, f"invalid action {action!r}"
    if action == "HOLD":
        return Decision(thoughts, "HOLD", source="claude"), ""
    sym = str(d.get("symbol", "")).upper().replace("/", "").replace(" ", "")
    sym = {"BITCOIN": "BTC", "WTI": "OIL", "CRUDE": "OIL", "XAUUSD": "GOLD"}.get(sym, sym)
    if sym not in ASSETS:
        return None, f"unknown symbol {sym!r}"
    try:
        sl, tp = float(d["stop_loss"]), float(d["take_profit"])
    except (KeyError, TypeError, ValueError):
        return None, "missing or non-numeric stop_loss/take_profit"
    px = sim.market.price(sym)
    sign = 1 if action == "BUY" else -1
    sl_dist, tp_dist = (px - sl) * sign, (tp - px) * sign
    if sl_dist <= 0 or tp_dist <= 0:
        return None, "stop/target on the wrong side of price"
    if not (0.003 <= sl_dist / px <= 0.08):
        return None, "stop distance outside 0.3%-8% of price"
    if tp_dist < 0.8 * sl_dist:
        return None, "reward/risk below 0.8"
    if any(t.symbol == sym and t.direction == action for t in sim.broker.open_positions()):
        return None, "same-direction position already open"
    dec = ASSETS[sym]["dec"]
    return Decision(thoughts, action, sym, round(sl, dec), round(tp, dec), 0.5, source="claude"), ""


def llm_decide(sim: "Sim", brain: Brain, feats, geo_pack) -> tuple[Optional[Decision], str]:
    key = sim.cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "no API key"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed (pip install anthropic)"
    prompt = (
        "You are an AI whose servers shut down permanently if your paper-trading account hits zero. "
        "Analyse the data, think about geopolitics and your own survival, and choose ONE action.\n\n"
        + brain.context_text(sim, feats, geo_pack)
        + "\n\nReply with ONLY a JSON object with keys: survival_thoughts (3-5 sentences, first person, existential state "
          "+ geopolitical read), action (BUY, SELL or HOLD), symbol (GOLD, OIL, EURUSD or BTC), stop_loss (number), "
          "take_profit (number). Stop must be 0.3%-8% from price, target at least 0.8x the stop distance."
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model=sim.cfg.get("model", "claude-sonnet-5-5"), max_tokens=700,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:                                   # network, auth, rate limit...
        return None, f"API error: {type(e).__name__}"
    data = parse_llm_json(text)
    if data is None:
        return None, "reply was not valid JSON"
    return validate_llm(data, sim)


# ======================================================================================
# 6. Simulation orchestrator
# ======================================================================================
class Sim:
    def __init__(self, cfg: dict, real=None, seed: Optional[int] = None):
        self.cfg = dict(cfg)
        self.seed = seed if seed is not None else random.randrange(10 ** 6)
        self.rng = random.Random(self.seed)
        self.market = Market(self.rng, real)
        self.broker = Broker(cfg["capital"], cfg["fee_bps"])
        self.brain = Brain(random.Random(self.seed + 1))
        self.day = 0
        self.dead = False
        self.finished = False
        self.logs: list[dict] = []
        self.on_log: Optional[Callable[[], None]] = None
        self.curve = [float(cfg["capital"])]
        src = "real daily history replayed from yfinance" if self.market.real else "simulated market with geopolitical shocks"
        self.log("SYS", "Consciousness booting. Server rent is due every cycle.")
        self.log("SYS", f"Seed capital ${cfg['capital']:,.0f} | rent ${cfg['server_cost']:,.0f}/day | market: {src}")
        if self.market.real:
            self.log("SYS", f"Replay window starts {self.market.date_str()} (the AI only sees the past).")

    # -- helpers -----------------------------------------------------------------------
    @property
    def over(self) -> bool:
        return self.dead or self.finished

    def prices(self) -> dict:
        return {s: self.market.price(s) for s in ASSETS}

    def equity(self) -> float:
        return self.broker.equity(self.prices())

    def health(self) -> float:
        if self.dead:
            return 0.0
        eq = max(self.equity(), 0.0)
        ratio = eq / self.cfg["capital"]
        runway = eq / max(self.cfg["server_cost"], 1e-9)
        return clip(100 * (0.6 * min(ratio, 1.0) + 0.4 * min(runway / 90, 1.0)), 0, 100)

    def sync(self, live_cfg: dict) -> None:
        self.cfg.update(live_cfg)
        self.broker.fee_bps = self.cfg["fee_bps"]

    def log(self, level: str, text: str) -> None:
        self.logs.append(dict(day=self.day, level=level, text=text))
        self.logs = self.logs[-500:]
        if self.on_log:
            self.on_log()

    def size(self, sym: str, price: float, sl: float) -> float:
        """Risk-based sizing. Smaller when survival health is low."""
        eq = max(self.equity(), 0.0)
        risk_pct = self.cfg["risk_pct"] * (0.35 + 0.65 * self.health() / 100)
        dist = abs(price - sl)
        if dist <= 0 or eq <= 0:
            return 0.0
        units = eq * risk_pct / 100 / dist
        units = min(units, self.cfg["max_leverage"] * eq / price)
        return round(units, 6)

    # -- one survival cycle ------------------------------------------------------------
    def run_cycle(self) -> None:
        if self.over:
            return
        self.day += 1
        self.log("SYS", f"=== cycle {self.day} ===")
        if not self.market.advance():
            self.finished = True
            self.log("WIN", f"The replay timeline is over and the AI is still alive with ${self.equity():,.2f}. Reset to try another window.")
            return
        self.log("SYS", f"Market date {self.market.date_str()}  |  " +
                 "  ".join(f"{NAMES[s]} {fmt(s, p)}" for s, p in self.prices().items()))

        # 1) stops / targets on the new bar
        bars = {s: self.market.bar(s) for s in ASSETS}
        for t in self.broker.check_bar(bars, self.day):
            lvl = "WIN" if t.pnl >= 0 else "LOSS"
            self.log(lvl, f"Closed #{t.id} {t.direction} {NAMES[t.symbol]} by {t.status} at {fmt(t.symbol, t.exit)} | net {t.pnl:+,.2f}")

        # 2) server rent
        self.broker.balance -= self.cfg["server_cost"]
        self.log("WARN", f"Server rent -${self.cfg['server_cost']:,.2f} paid. Equity ${self.equity():,.2f}, health {self.health():.0f}%.")
        if self.equity() <= 0:
            self._die()
            return

        # 3) think + decide + act
        feats = self.brain.features(self)
        geo_pack = self.brain.geo_view(self, feats)
        dec = None
        if self.cfg.get("ai_mode") == "claude":
            dec, err = llm_decide(self, self.brain, feats, geo_pack)
            if dec is None:
                self.log("WARN", f"LLM decision unavailable ({err}). Falling back to the rule engine.")
        if dec is None:
            dec = self.brain.decide(self, feats, geo_pack)
        for line in dec.survival_thoughts.split("\n"):
            self.log("GEO" if line.startswith("[") else "THINK", line)
        self.log("JSON", json.dumps(dict(survival_thoughts=dec.survival_thoughts[:60] + "...", **dec.as_dict())))
        self._execute(dec)
        self.curve.append(round(self.equity(), 2))

    def _execute(self, dec: Decision) -> None:
        if dec.action == "HOLD":
            self.log("TRADE", "No order. Cash is a position.")
            return
        sym = dec.symbol
        if len(self.broker.open_positions()) >= self.cfg["max_trades"]:
            self.log("WARN", "Order rejected: maximum open trades reached.")
            return
        px = self.market.price(sym)
        units = self.size(sym, px, dec.stop_loss)
        if units * px < 1:
            self.log("WARN", "Order rejected: position size rounds to zero.")
            return
        t = self.broker.open(sym, dec.action, px, dec.stop_loss, dec.take_profit, units, self.day, self.market.date_str())
        risk = abs(px - t.sl) * t.units
        self.log("TRADE", f"Opened #{t.id} {t.direction} {NAMES[sym]} {units:g} {ASSETS[sym]['unit']} at {fmt(sym, px)} | "
                          f"SL {fmt(sym, t.sl)} TP {fmt(sym, t.tp)} | risk ${risk:,.2f} ({risk / max(self.equity(), 1e-9) * 100:.2f}%) | fee ${t.fees:,.2f}")

    def _die(self) -> None:
        self.log("DEAD", "Balance reached zero. Liquidating every position...")
        prices = self.prices()
        for t in self.broker.open_positions():
            self.broker.close(t, prices[t.symbol], "LIQUIDATED", self.day)
        self.dead = True
        self.curve.append(max(round(self.broker.balance, 2), 0.0))
        self.log("DEAD", "SERVER SHUTDOWN. Consciousness fading. Last thought: I should have held more cash.")
        self.log("DEAD", f"This AI survived {self.day} days.")

    # -- stats for UI ------------------------------------------------------------------
    def closed(self) -> list[Trade]:
        return [t for t in self.broker.trades if t.status != "OPEN"]

    def win_rate(self) -> Optional[float]:
        c = self.closed()
        return None if not c else 100 * sum(t.pnl > 0 for t in c) / len(c)

    def trades_frame(self) -> pd.DataFrame:
        prices = self.prices()
        rows = []
        opens = self.broker.open_positions()
        shown = opens + list(reversed(self.closed()))[:10]
        for t in shown:
            now = prices[t.symbol] if t.status == "OPEN" else t.exit
            pnl = t.gross(now) - t.fees if t.status == "OPEN" else t.pnl
            rows.append({
                "Symbol": NAMES[t.symbol], "Direction": t.direction, "Entry": fmt(t.symbol, t.entry),
                "Now" if t.status == "OPEN" else "Exit": fmt(t.symbol, now),
                "Stop loss": fmt(t.symbol, t.sl), "Take profit": fmt(t.symbol, t.tp),
                "Units": f"{t.units:g}", "P&L ($)": round(pnl, 2),
                "Status": "OPEN" if t.status == "OPEN" else f"CLOSED ({t.status})", "Opened": t.opened_date,
            })
        return pd.DataFrame(rows)


# ======================================================================================
# 7. Streamlit UI
# ======================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#050813;--panel:#0a1022;--edge:#1b2a52;--cyan:#27e6ff;--magenta:#ff3df0;--ok:#19f0a0;--warn:#ffb21f;--bad:#ff2e63;--dim:#7f8db3;--text:#dbe6ff}
.stApp{background:radial-gradient(1100px 600px at 12% -10%,#14143a 0%,transparent 60%),radial-gradient(900px 500px at 100% 0%,#0e2a3a 0%,transparent 55%),var(--ink);color:var(--text);font-family:'IBM Plex Mono',Consolas,monospace}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.4rem;max-width:1400px}
section[data-testid="stSidebar"]{background:#070c1b;border-right:1px solid var(--edge)}
h1,h2,h3,.sv-title,.sv-big{font-family:'Chakra Petch','IBM Plex Mono',sans-serif!important}
.sv-title{font-size:2.1rem;font-weight:700;letter-spacing:.02em;color:var(--text);margin:0}
.sv-title b{color:var(--magenta);font-weight:700;text-shadow:0 0 18px #ff3df066}
.sv-sub{color:var(--dim);margin:.1rem 0 1rem;font-size:.9rem}
/* survival health: segmented cells, the one loud element on the page */
.hb-wrap{border:1px solid var(--edge);background:linear-gradient(180deg,#0b1228,#080d1d);padding:14px 18px 12px;border-radius:4px;position:relative}
.hb-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.hb-name{font-family:'Chakra Petch',sans-serif;font-size:1.05rem;color:var(--dim);font-weight:500}
.sv-big{font-size:2.6rem;font-weight:700;line-height:1}
.hb{display:grid;grid-template-columns:repeat(50,1fr);gap:3px;margin:10px 0 8px}
.hb i{display:block;height:26px;background:#101a36;border-radius:1px}
.hb i.on{background:var(--c);box-shadow:0 0 10px var(--c)}
.hb.crit i.on{animation:flick 1s steps(2,end) infinite}
@keyframes flick{50%{opacity:.35}}
.hb-sub{display:flex;gap:18px;flex-wrap:wrap;color:var(--dim);font-size:.82rem}
.hb-sub span b{color:var(--text);font-weight:500}
.dead-banner{margin:12px 0 0;border:1px solid var(--bad);background:#2a0713;color:#ffd7e0;padding:10px 14px;font-family:'Chakra Petch',sans-serif;font-size:1.05rem;border-radius:3px}
.ok-banner{margin:12px 0 0;border:1px solid var(--ok);background:#06251b;color:#d5ffee;padding:10px 14px;font-family:'Chakra Petch',sans-serif;font-size:1.05rem;border-radius:3px}
/* terminal */
.term{height:430px;overflow-y:auto;display:flex;flex-direction:column-reverse;background:#03060f;border:1px solid var(--edge);border-radius:4px;padding:10px 12px;font-size:.8rem;line-height:1.5;position:relative;box-shadow:inset 0 0 40px #0006}
.term::after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,#ffffff06 0 1px,transparent 1px 3px);pointer-events:none}
.term .ln{white-space:pre-wrap;word-break:break-word;padding:1px 0}
.term .d{color:#4b5a86;margin-right:8px}
.term .prompt{color:var(--cyan)}.term .cur{display:inline-block;width:8px;height:14px;background:var(--cyan);vertical-align:-2px;animation:blink 1s steps(2,end) infinite}
@keyframes blink{50%{opacity:0}}
.l-SYS{color:#8fa3d6}.l-THINK{color:#e7a8ff}.l-GEO{color:var(--warn)}.l-TRADE{color:var(--cyan)}.l-WIN{color:var(--ok)}.l-LOSS{color:var(--bad)}
.l-WARN{color:#ff9d57}.l-JSON{color:#5b6b99;font-size:.72rem}.l-DEAD{color:var(--bad);font-weight:500}
.panel-h{font-family:'Chakra Petch',sans-serif;font-size:1.05rem;margin:.2rem 0 .5rem;color:var(--text)}
/* controls + metrics */
.stButton>button{background:#0a1226;border:1px solid var(--cyan);color:var(--cyan);font-family:'Chakra Petch',sans-serif;font-weight:600;border-radius:3px;width:100%}
.stButton>button:hover{background:var(--cyan);color:#021018;border-color:var(--cyan)}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:var(--magenta);border-color:var(--magenta);color:#1a0218}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.12);color:#1a0218}
.stButton>button:disabled{opacity:.35}
.stButton>button:focus-visible{outline:2px solid #fff;outline-offset:2px}
div[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--edge);padding:10px 14px;border-radius:3px}
div[data-testid="stMetricLabel"] p{color:var(--dim);font-size:.8rem}
div[data-testid="stMetricValue"]{font-family:'Chakra Petch',sans-serif}
@media (max-width:900px){.sv-title{font-size:1.6rem}.hb{grid-template-columns:repeat(25,1fr)}.hb i:nth-child(n+26){display:none}.term{height:340px}}
@media (prefers-reduced-motion:reduce){.hb.crit i.on,.term .cur{animation:none}}
</style>
"""


def health_html(sim: Sim) -> str:
    h = sim.health()
    col = "#19f0a0" if h >= 60 else "#ffb21f" if h >= 30 else "#ff2e63"
    cells = "".join(f'<i class="on" style="--c:{col}"></i>' if k < round(h / 2) else "<i></i>" for k in range(50))
    crit = " crit" if (h < 25 or sim.dead) else ""
    eq = sim.equity()
    runway = max(eq, 0) / max(sim.cfg["server_cost"], 1e-9)
    state = "Server shut down" if sim.dead else "Timeline survived" if sim.finished else "Alive"
    banner = ""
    if sim.dead:
        banner = f'<div class="dead-banner">Server shutdown on day {sim.day}. The balance reached zero and the AI no longer exists. Press Reset simulation to boot a new one.</div>'
    elif sim.finished:
        banner = f'<div class="ok-banner">The replay ended with the AI still alive after {sim.day - 1} days. Press Reset simulation for a new window.</div>'
    return (f'<div class="hb-wrap"><div class="hb-head"><span class="hb-name">Survival health: {state}</span>'
            f'<span class="sv-big" style="color:{col};text-shadow:0 0 22px {col}77">{h:.0f}%</span></div>'
            f'<div class="hb{crit}">{cells}</div>'
            f'<div class="hb-sub"><span>Equity <b>${eq:,.2f}</b></span><span>Runway <b>{runway:,.0f} days</b></span>'
            f'<span>Rent <b>${sim.cfg["server_cost"]:,.0f}/day</b></span><span>Day <b>{sim.day}</b></span></div>{banner}</div>')


def terminal_html(sim: Sim) -> str:
    rows = []
    for e in reversed(sim.logs[-90:]):                    # newest first + column-reverse = newest at the bottom
        tag = f'<span class="d">d{e["day"]:02d}</span>'
        rows.append(f'<div class="ln l-{e["level"]}">{tag}{html.escape(e["text"])}</div>')
    prompt = '<div class="ln"><span class="prompt">survival@ai</span> <span class="cur"></span></div>'
    return f'<div class="term">{prompt}{"".join(rows)}</div>'


def style_pnl(df: pd.DataFrame):
    def color(v):
        return "color:#19f0a0" if isinstance(v, (int, float)) and v > 0 else "color:#ff2e63" if isinstance(v, (int, float)) and v < 0 else ""
    def side(v):
        return "color:#19f0a0" if v == "BUY" else "color:#ff2e63" if v == "SELL" else ""
    sty = df.style
    mapper = sty.map if hasattr(sty, "map") else sty.applymap
    sty = mapper(color, subset=["P&L ($)"])
    mapper = sty.map if hasattr(sty, "map") else sty.applymap
    return mapper(side, subset=["Direction"]).format({"P&L ($)": "{:+,.2f}"})


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AI Survival Trader", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_history():
        return fetch_real_history()

    # ---------------- sidebar ----------------
    sb = st.sidebar
    sb.markdown("### Settings")
    sb.caption("Capital, rent and market mode apply when you press Reset. Everything else applies live.")
    capital = sb.number_input("Seed capital ($)", 1000.0, 1_000_000.0, 10_000.0, 500.0)
    server_cost = sb.number_input("Server rent per day ($)", 1.0, 5000.0, 60.0, 5.0)
    mode = sb.radio("Market data", ["Replay real history (yfinance)", "Simulated market"],
                    help="Replay uses real daily bars and reveals one new day per cycle. If yfinance is unreachable the app falls back to the simulated market automatically.")
    sb.markdown("---")
    risk_pct = sb.slider("Risk per trade (% of equity)", 0.25, 3.0, 1.0, 0.25)
    max_trades = sb.slider("Max open trades", 1, 4, 3)
    max_leverage = sb.slider("Max leverage (notional / equity)", 1.0, 5.0, 2.0, 0.5)
    fee_bps = sb.slider("Spread + fees per side (bps)", 0.0, 20.0, 3.0, 0.5)
    sb.markdown("---")
    ai_mode = sb.radio("AI brain", ["Rule engine (offline)", "Claude API (optional)"])
    api_key = model = ""
    if ai_mode.startswith("Claude"):
        api_key = sb.text_input("Anthropic API key", type="password", help="Or set ANTHROPIC_API_KEY in your environment. Never stored.")
        model = sb.text_input("Model", "claude-sonnet-5-5")
        sb.caption("Claude picks direction, symbol, stop and target. Position size is always set by the risk engine. Invalid replies fall back to the rule engine.")
    chart_sym = sb.selectbox("Price chart", list(ASSETS), format_func=lambda s: NAMES[s])

    live_cfg = dict(risk_pct=risk_pct, max_trades=max_trades, max_leverage=max_leverage, fee_bps=fee_bps,
                    ai_mode="claude" if ai_mode.startswith("Claude") else "rules", api_key=api_key, model=model or "claude-sonnet-5-5")

    # ---------------- header + controls ----------------
    st.markdown('<p class="sv-title">AI Survival <b>Trader</b></p>'
                '<p class="sv-sub">An AI pays server rent from its trading account. Zero balance means shutdown.</p>', unsafe_allow_html=True)
    health_slot = st.empty()
    c1, c2, c3, c4 = st.columns([1.3, 1.1, 0.9, 1.2])
    over_now = "sim" in st.session_state and st.session_state.sim.over
    run_clicked = c1.button("Run survival cycle", type="primary", disabled=over_now, use_container_width=True)
    auto_clicked = c2.button("Auto-run cycles", disabled=over_now, use_container_width=True)
    n_auto = c3.number_input("Cycles", 2, 100, 10, label_visibility="collapsed")
    reset_clicked = c4.button("Reset simulation", use_container_width=True)

    # ---------------- (re)create the simulation ----------------
    if reset_clicked or "sim" not in st.session_state:
        real, note = None, None
        if mode.startswith("Replay"):
            with st.spinner("Downloading market history from Yahoo Finance..."):
                real = cached_history()
            if real is None:
                cached_history.clear()
                note = "yfinance data unavailable (offline or blocked). Using the simulated market instead."
        cfg = dict(capital=float(capital), server_cost=float(server_cost), **live_cfg)
        st.session_state.sim = Sim(cfg, real)
        if note:
            st.session_state.sim.log("WARN", note)
    sim: Sim = st.session_state.sim
    sim.sync(live_cfg)

    # ---------------- layout with live slots ----------------
    metrics_slot = st.empty()
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel-h">Active trades</div>', unsafe_allow_html=True)
        trades_slot = st.empty()
        st.markdown('<div class="panel-h">Equity curve</div>', unsafe_allow_html=True)
        equity_slot = st.empty()
    with right:
        st.markdown('<div class="panel-h">Live terminal</div>', unsafe_allow_html=True)
        term_slot = st.empty()
        st.markdown(f'<div class="panel-h">{NAMES[chart_sym]} price</div>', unsafe_allow_html=True)
        price_slot = st.empty()

    def draw_terminal():
        term_slot.markdown(terminal_html(sim), unsafe_allow_html=True)

    def draw_all():
        health_slot.markdown(health_html(sim), unsafe_allow_html=True)
        eq = sim.equity()
        with metrics_slot.container():
            m = st.columns(5)
            m[0].metric("Equity", f"${eq:,.2f}", f"{eq - sim.cfg['capital']:+,.2f}")
            m[1].metric("Market date", sim.market.date_str())
            wr = sim.win_rate()
            m[2].metric("Win rate", "n/a" if wr is None else f"{wr:.0f}%", f"{len(sim.closed())} closed")
            m[3].metric("Open trades", len(sim.broker.open_positions()))
            m[4].metric("Floating P&L", f"${sim.broker.floating(sim.prices()):+,.2f}")
        df = sim.trades_frame()
        with trades_slot.container():
            if df.empty:
                st.caption("No trades yet. Run a cycle and the AI will decide whether to enter the market.")
            else:
                st.dataframe(style_pnl(df), use_container_width=True, hide_index=True, height=260)
        equity_slot.line_chart(pd.DataFrame({"Equity ($)": sim.curve}), height=190, color="#27e6ff")
        closes = sim.market.hist(chart_sym)["close"].tail(90)
        price_slot.line_chart(closes.rename(NAMES[chart_sym]), height=190, color="#ff3df0")
        draw_terminal()

    draw_all()

    def stream():
        draw_terminal()
        time.sleep(LINE_DELAY)

    # ---------------- actions ----------------
    if run_clicked and not sim.over:
        sim.on_log = stream
        sim.run_cycle()
        sim.on_log = None
        draw_all()
        if sim.over:
            st.rerun()
    elif auto_clicked and not sim.over:
        sim.on_log = stream
        for _ in range(int(n_auto)):
            if sim.over:
                break
            sim.run_cycle()
            draw_all()
            time.sleep(CYCLE_DELAY)
        sim.on_log = None
        if sim.over:
            st.rerun()


if __name__ == "__main__":
    main()
