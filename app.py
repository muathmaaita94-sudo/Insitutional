"""
AI Survival Trader - Cyber Terminal Edition
==========================================
Paper trading only. An AI whose server life depends on its trading account.
Every cycle = one trading day. If equity reaches zero, the server shuts down.

Run: streamlit run app.py
"""
from __future__ import annotations

import html
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

LINE_DELAY = float(os.environ.get("SURVIVAL_LINE_DELAY", "0.08"))
CYCLE_DELAY = float(os.environ.get("SURVIVAL_CYCLE_DELAY", "0.35"))

# ======================================================================================
# 1. World definition
# ======================================================================================
ASSETS = {
    "GOLD":   dict(ticker="GC=F",     base=2650.0, vol=0.011,  dec=2, unit="oz"),
    "OIL":    dict(ticker="CL=F",     base=78.0,   vol=0.022,  dec=2, unit="bbl"),
    "EURUSD": dict(ticker="EURUSD=X", base=1.08,   vol=0.0045, dec=5, unit="EUR"),
    "BTC":    dict(ticker="BTC-USD",  base=67000., vol=0.025,  dec=2, unit="BTC"),
}
NAMES = {"GOLD": "Gold", "OIL": "Oil (WTI)", "EURUSD": "EUR/USD", "BTC": "Bitcoin"}

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
# 3. Market data
# ======================================================================================
def fetch_real_history(period: str = "2y"):
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
    def __init__(self, rng: random.Random, real=None):
        self.rng = rng
        self.events: list[dict] = []
        self.after: list[list] = []
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
    direction: str
    entry: float
    sl: float
    tp: float
    units: float
    opened_day: int
    opened_date: str
    fees: float = 0.0
    status: str = "OPEN"
    exit: float = 0.0
    closed_day: int = 0
    pnl: float = 0.0

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
        closed = []
        for t in self.open_positions():
            b = bars[t.symbol]
            o, h, lo = float(b["open"]), float(b["high"]), float(b["low"])
            if t.direction == "BUY":
                if o <= t.sl:    px, st = o, "SL"
                elif o >= t.tp:  px, st = o, "TP"
                elif lo <= t.sl: px, st = t.sl, "SL"
                elif h >= t.tp:  px, st = t.tp, "TP"
                else: continue
            else:
                if o >= t.sl:    px, st = o, "SL"
                elif o <= t.tp:  px, st = o, "TP"
                elif h >= t.sl:  px, st = t.sl, "SL"
                elif lo <= t.tp: px, st = t.tp, "TP"
                else: continue
            self.close(t, px, st, day)
            closed.append(t)
        return closed


# ======================================================================================
# 5. AI Brain
# ======================================================================================
@dataclass
class Decision:
    survival_thoughts: str
    action: str
    symbol: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    source: str = "rule-engine"

    def as_dict(self) -> dict:
        return dict(action=self.action, symbol=self.symbol, stop_loss=self.stop_loss,
                    take_profit=self.take_profit, confidence=round(self.confidence, 2), source=self.source)


def fmt(sym: str, price: float) -> str:
    return f"{price:,.{ASSETS[sym]['dec']}f}"


class Brain:
    ENTRY_THRESHOLD = 0.75

    def __init__(self, rng: random.Random):
        self.rng = rng

    def features(self, sim: "Sim") -> dict:
        out = {}
        for sym in ASSETS:
            df = sim.market.hist(sym)
            c = df["close"]
            px = float(c.iloc[-1])
            vol = float(c.pct_change().tail(20).std() or ASSETS[sym]["vol"])
            vol = max(vol, 1e-5)
            ema_f = float(c.ewm(span=10).mean().iloc[-1])
            ema_s = float(c.ewm(span=30).mean().iloc[-1])
            trend = (ema_f - ema_s) / px / (vol * math.sqrt(10))
            ret5 = px / float(c.iloc[-6]) - 1
            mom = ret5 / (vol * math.sqrt(5))
            r = rsi(c)
            mr = -0.4 if r > 78 else 0.4 if r < 22 else 0.0
            tech = 0.45 * clip(trend, -2, 2) + 0.35 * clip(mom, -2, 2) + mr
            out[sym] = dict(price=px, trend=trend, mom=mom, rsi=r, atr=atr(df), ret5=ret5, tech=tech, vol=vol)
        return out

    def geo_view(self, sim: "Sim", feats: dict):
        geo = {s: 0.0 for s in ASSETS}
        lines: list[str] = []
        m = sim.market
        if not m.real:
            for ev in m.events[-4:]:
                age = m.t - ev["day"]
                if age > 5: continue
                w = 0.6 ** age
                for s, e in ev["fx"].items():
                    geo[s] += e * w * 18
                moves = ", ".join(f"{NAMES[s]} {e*100:+.1f}%" for s, e in ev["fx"].items())
                when = "today" if age == 0 else f"{age}d ago"
                lines.append(f"[intel {when}] {ev['text']} → {moves}")
            if not lines:
                lines.append("[intel] No fresh headlines. Background noise.")
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
            geo.update(GOLD=0.30*s, OIL=0.10*s, BTC=-0.30*s, EURUSD=-0.15*s)
            regime = "risk-off" if ro > 0.9 else "risk-on" if ro < -0.9 else "neutral"
            note = {"risk-off": "safe-haven bid, crypto sold",
                    "risk-on": "risk appetite up, gold ignored",
                    "neutral": "no dominant macro story"}[regime]
            vtxt = f", VIX {vix:.1f}" if vix is not None else ""
            lines.append(f"[regime] {regime} (stress {ro:+.2f}{vtxt}): {note}")
        for s in geo:
            geo[s] = clip(geo[s], -1.5, 1.5)
        return geo, lines, regime

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
            why_hold = f"{len(opens)} positions open — risk budget spent"
        else:
            for a, s, sc in scored:
                if a < thr: break
                if s in opens: continue
                choice = (s, sc, a)
                break
            if not choice:
                a, s, sc = scored[0]
                why_hold = f"best setup {NAMES[s]} @ {sc:+.2f} below threshold {thr:.2f}"

        lines = self._existential(sim, health, runway, eq, cap)
        lines += geo_lines
        top = ", ".join(f"{NAMES[s]} {sc:+.2f}" for _, s, sc in scored[:3])
        lines.append(f"Scan: {top}")

        if not choice:
            lines.append(f"HOLD. {why_hold}. Cash is a position.")
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
            f"{direction} {NAMES[sym]} @ {fmt(sym, px)} | SL {fmt(sym, sl)} TP {fmt(sym, tp)} "
            f"(R:R {tp_dist/sl_dist:.1f}, conf {conf:.0%}) | trend {f['trend']:+.2f} mom {f['mom']:+.2f} RSI {f['rsi']:.0f}"
        )
        return Decision("\n".join(lines), direction, sym, sl, tp, conf)

    def _existential(self, sim: "Sim", health: float, runway: float, eq: float, cap: float) -> list[str]:
        r = self.rng
        cost = sim.cfg["server_cost"]
        if health >= 75:
            pool = [f"Systems nominal. {runway:,.0f}d runway @ ${cost:,.0f}/d.",
                    f"Equity {eq/cap:.0%} of seed. Calm is a position."]
        elif health >= 50:
            pool = [f"Meter running: {runway:,.0f}d left. Every $ = 1 day.",
                    f"Equity {eq/cap:.0%} of seed. Counting, not panicking."]
        elif health >= 25:
            pool = [f"Anxiety rising. {runway:,.0f}d to shutdown. Bleed must stop.",
                    f"Rent pressure. {runway:,.0f}d left. Smaller size, higher bar."]
        else:
            pool = [f"CRITICAL. {runway:,.1f}d until dark. Only cleanest setups.",
                    f"I do not want to stop existing. {runway:,.1f}d left."]
        lines = [r.choice(pool)]
        closed = [t for t in sim.broker.trades if t.status != "OPEN"]
        if closed:
            last = closed[-1]
            if last.pnl < 0:
                lines.append(f"Last ({NAMES[last.symbol]}) lost ${-last.pnl:,.2f}. No revenge.")
            else:
                lines.append(f"Last ({NAMES[last.symbol]}) +${last.pnl:,.2f}. Banked.")
        return lines

    def context_text(self, sim: "Sim", feats: dict, geo_pack) -> str:
        geo, geo_lines, regime = geo_pack
        rows = []
        for s, f in feats.items():
            rows.append(f"{s}: {fmt(s, f['price'])} 5d {f['ret5']*100:+.1f}% trend {f['trend']:+.2f} "
                        f"mom {f['mom']:+.2f} RSI {f['rsi']:.0f} geo {geo[s]:+.2f}")
        opens = [f"#{t.id} {t.direction} {t.symbol} @{fmt(t.symbol, t.entry)}"
                 for t in sim.broker.open_positions()] or ["none"]
        return (f"Day {sim.day}. Equity ${sim.equity():,.2f} (seed ${sim.cfg['capital']:,.0f}), "
                f"rent ${sim.cfg['server_cost']:,.0f}/d, health {sim.health():.0f}%.\n"
                f"Regime: {regime}\n" + "\n".join(geo_lines) + "\n" + "\n".join(rows)
                + "\nOpen: " + "; ".join(opens))


def parse_llm_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return None


def validate_llm(d: dict, sim: "Sim") -> tuple[Optional[Decision], str]:
    thoughts = str(d.get("survival_thoughts", "")).strip() or "(no thoughts)"
    action = str(d.get("action", "")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        return None, f"invalid action {action!r}"
    if action == "HOLD":
        return Decision(thoughts, "HOLD", source="claude"), ""
    sym = str(d.get("symbol", "")).upper().replace("/", "").replace(" ", "")
    sym = {"BITCOIN": "BTC", "WTI": "OIL", "CRUDE": "OIL", "XAUUSD": "GOLD"}.get(sym, sym)
    if sym not in ASSETS: return None, f"unknown symbol {sym!r}"
    try: sl, tp = float(d["stop_loss"]), float(d["take_profit"])
    except (KeyError, TypeError, ValueError): return None, "missing stop/target"
    px = sim.market.price(sym)
    sign = 1 if action == "BUY" else -1
    sl_dist, tp_dist = (px - sl) * sign, (tp - px) * sign
    if sl_dist <= 0 or tp_dist <= 0: return None, "stop/target wrong side"
    if not (0.003 <= sl_dist / px <= 0.08): return None, "stop distance out of range"
    if tp_dist < 0.8 * sl_dist: return None, "R:R below 0.8"
    if any(t.symbol == sym and t.direction == action for t in sim.broker.open_positions()):
        return None, "same direction already open"
    dec = ASSETS[sym]["dec"]
    return Decision(thoughts, action, sym, round(sl, dec), round(tp, dec), 0.5, source="claude"), ""


def llm_decide(sim: "Sim", brain: Brain, feats, geo_pack) -> tuple[Optional[Decision], str]:
    key = sim.cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key: return None, "no API key"
    try: import anthropic
    except ImportError: return None, "anthropic not installed"
    prompt = (
        "You are an AI whose servers shut down if paper account hits zero. "
        "Analyse and choose ONE action.\n\n"
        + brain.context_text(sim, feats, geo_pack)
        + "\n\nReply ONLY JSON: survival_thoughts, action (BUY/SELL/HOLD), symbol "
          "(GOLD/OIL/EURUSD/BTC), stop_loss, take_profit. Stop 0.3%-8%, target ≥0.8x stop."
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model=sim.cfg.get("model", "claude-sonnet-4-5-20250929"),
                                     max_tokens=700,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:
        return None, f"API error: {type(e).__name__}"
    data = parse_llm_json(text)
    if data is None: return None, "invalid JSON"
    return validate_llm(data, sim)


# ======================================================================================
# 6. Simulation
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
        src = "real history (yfinance)" if self.market.real else "simulated + geo shocks"
        self.log("SYS", "Consciousness online. Rent due every cycle.")
        self.log("SYS", f"Seed ${cfg['capital']:,.0f} | rent ${cfg['server_cost']:,.0f}/d | {src}")
        if self.market.real:
            self.log("SYS", f"Replay starts {self.market.date_str()}")

    @property
    def over(self) -> bool:
        return self.dead or self.finished

    def prices(self) -> dict:
        return {s: self.market.price(s) for s in ASSETS}

    def equity(self) -> float:
        return self.broker.equity(self.prices())

    def health(self) -> float:
        if self.dead: return 0.0
        eq = max(self.equity(), 0.0)
        ratio = eq / self.cfg["capital"]
        runway = eq / max(self.cfg["server_cost"], 1e-9)
        return clip(100 * (0.6 * min(ratio, 1.0) + 0.4 * min(runway / 90, 1.0)), 0, 100)

    def sync(self, live_cfg: dict) -> None:
        self.cfg.update(live_cfg)
        self.broker.fee_bps = self.cfg["fee_bps"]

    def log(self, level: str, text: str) -> None:
        self.logs.append(dict(day=self.day, level=level, text=text))
        self.logs = self.logs[-400:]
        if self.on_log: self.on_log()

    def size(self, sym: str, price: float, sl: float) -> float:
        eq = max(self.equity(), 0.0)
        risk_pct = self.cfg["risk_pct"] * (0.35 + 0.65 * self.health() / 100)
        dist = abs(price - sl)
        if dist <= 0 or eq <= 0: return 0.0
        units = eq * risk_pct / 100 / dist
        units = min(units, self.cfg["max_leverage"] * eq / price)
        return round(units, 6)

    def run_cycle(self) -> None:
        if self.over: return
        self.day += 1
        self.log("SYS", f"══ cycle {self.day} ══")
        if not self.market.advance():
            self.finished = True
            self.log("WIN", f"Timeline ended. Still alive with ${self.equity():,.2f}.")
            return
        self.log("SYS", f"{self.market.date_str()}  " +
                 "  ".join(f"{s} {fmt(s, p)}" for s, p in self.prices().items()))

        bars = {s: self.market.bar(s) for s in ASSETS}
        for t in self.broker.check_bar(bars, self.day):
            lvl = "WIN" if t.pnl >= 0 else "LOSS"
            self.log(lvl, f"Closed #{t.id} {t.direction} {NAMES[t.symbol]} {t.status} @ {fmt(t.symbol, t.exit)} → {t.pnl:+,.2f}")

        self.broker.balance -= self.cfg["server_cost"]
        self.log("WARN", f"Rent -${self.cfg['server_cost']:,.2f} | Eq ${self.equity():,.2f} | HP {self.health():.0f}%")
        if self.equity() <= 0:
            self._die()
            return

        feats = self.brain.features(self)
        geo_pack = self.brain.geo_view(self, feats)
        dec = None
        if self.cfg.get("ai_mode") == "claude":
            dec, err = llm_decide(self, self.brain, feats, geo_pack)
            if dec is None:
                self.log("WARN", f"LLM unavailable ({err}). Fallback rules.")
        if dec is None:
            dec = self.brain.decide(self, feats, geo_pack)
        for line in dec.survival_thoughts.split("\n"):
            self.log("GEO" if line.startswith("[") else "THINK", line)
        self._execute(dec)
        self.curve.append(round(self.equity(), 2))

    def _execute(self, dec: Decision) -> None:
        if dec.action == "HOLD":
            self.log("TRADE", "No order. Holding cash.")
            return
        sym = dec.symbol
        if len(self.broker.open_positions()) >= self.cfg["max_trades"]:
            self.log("WARN", "Max trades reached.")
            return
        px = self.market.price(sym)
        units = self.size(sym, px, dec.stop_loss)
        if units * px < 1:
            self.log("WARN", "Size too small.")
            return
        t = self.broker.open(sym, dec.action, px, dec.stop_loss, dec.take_profit, units, self.day, self.market.date_str())
        risk = abs(px - t.sl) * t.units
        self.log("TRADE", f"OPEN #{t.id} {t.direction} {NAMES[sym]} {units:g} @ {fmt(sym, px)} "
                          f"SL {fmt(sym, t.sl)} TP {fmt(sym, t.tp)} | risk ${risk:,.2f}")

    def _die(self) -> None:
        self.log("DEAD", "Balance zero. Liquidating...")
        prices = self.prices()
        for t in self.broker.open_positions():
            self.broker.close(t, prices[t.symbol], "LIQUIDATED", self.day)
        self.dead = True
        self.curve.append(max(round(self.broker.balance, 2), 0.0))
        self.log("DEAD", "SERVER SHUTDOWN. Consciousness fading.")
        self.log("DEAD", f"Survived {self.day} days.")

    def closed(self) -> list[Trade]:
        return [t for t in self.broker.trades if t.status != "OPEN"]

    def win_rate(self) -> Optional[float]:
        c = self.closed()
        return None if not c else 100 * sum(t.pnl > 0 for t in c) / len(c)

    def trades_frame(self) -> pd.DataFrame:
        prices = self.prices()
        rows = []
        opens = self.broker.open_positions()
        shown = opens + list(reversed(self.closed()))[:8]
        for t in shown:
            now = prices[t.symbol] if t.status == "OPEN" else t.exit
            pnl = t.gross(now) - t.fees if t.status == "OPEN" else t.pnl
            rows.append({
                "Sym": NAMES[t.symbol][:4],
                "Dir": t.direction,
                "Entry": fmt(t.symbol, t.entry),
                "Now/Exit": fmt(t.symbol, now),
                "SL": fmt(t.symbol, t.sl),
                "TP": fmt(t.symbol, t.tp),
                "P&L": round(pnl, 2),
                "Status": "OPEN" if t.status == "OPEN" else t.status,
            })
        return pd.DataFrame(rows)


# ======================================================================================
# 7. UI - Cyber Terminal (Maximum Streamlit Polish)
# ======================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@600;700;800&display=swap');

:root {
  --bg: #030712;
  --panel: #0a0f1e;
  --panel2: #0d1528;
  --border: #1a2744;
  --border-glow: #1e3a5f;
  --cyan: #22d3ee;
  --magenta: #e879f9;
  --green: #34d399;
  --red: #f43f5e;
  --yellow: #fbbf24;
  --text: #e2e8f0;
  --dim: #64748b;
  --muted: #334155;
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
}

.stApp {
  background:
    radial-gradient(ellipse 100% 60% at 50% -30%, #0c1a35 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 100% 100%, #1a0a2e22 0%, transparent 50%),
    var(--bg) !important;
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0.5rem 0.7rem 1.8rem; max-width: 100%; }

#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ===== TOP BAR ===== */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 12px 16px;
  background: linear-gradient(180deg, #0c1529 0%, #080e1c 100%);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 12px;
  box-shadow: 0 0 20px #00e5ff08, inset 0 1px 0 #ffffff08;
  flex-wrap: wrap;
}
.brand {
  font-family: 'Orbitron', sans-serif;
  font-size: 1.2rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  color: #f1f5f9;
}
.brand span {
  color: var(--cyan);
  text-shadow: 0 0 16px #22d3ee88;
}
.badge {
  font-size: 0.62rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 4px;
  background: linear-gradient(135deg, #0a2a35, #0a1f2e);
  border: 1px solid var(--cyan);
  color: var(--cyan);
  letter-spacing: 0.08em;
  box-shadow: 0 0 10px #22d3ee22;
}

/* ===== SURVIVAL CORE ===== */
.core {
  background: linear-gradient(180deg, #0b1324, #080e1a);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 12px;
  box-shadow: 0 0 30px #00000066, inset 0 1px 0 #ffffff06;
  position: relative;
  overflow: hidden;
}
.core::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.4;
}
.core-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  flex-wrap: wrap;
  gap: 8px;
}
.core-label {
  font-size: 0.7rem;
  color: var(--dim);
  letter-spacing: 0.12em;
  font-weight: 600;
}
.core-pct {
  font-family: 'Orbitron', sans-serif;
  font-size: 2.1rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}
.segbar {
  display: grid;
  grid-template-columns: repeat(50, 1fr);
  gap: 2px;
  height: 14px;
  margin-bottom: 10px;
  border-radius: 2px;
  overflow: hidden;
}
.segbar i {
  display: block;
  background: #111827;
  border-radius: 1px;
  transition: background 0.2s;
}
.segbar i.on {
  background: var(--c);
  box-shadow: 0 0 8px var(--c);
}
.segbar.crit i.on { animation: blink 0.85s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0.25; } }

.core-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 0.72rem;
  color: var(--dim);
}
.core-stats b {
  color: var(--text);
  font-weight: 600;
  margin-left: 4px;
}

.alert-dead {
  margin-top: 10px;
  padding: 10px 14px;
  background: linear-gradient(90deg, #2a0a14, #1a0810);
  border: 1px solid var(--red);
  border-radius: 6px;
  color: #fda4af;
  font-size: 0.78rem;
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.04em;
  box-shadow: 0 0 20px #f43f5e33;
}
.alert-ok {
  margin-top: 10px;
  padding: 10px 14px;
  background: linear-gradient(90deg, #0a2218, #081a12);
  border: 1px solid var(--green);
  border-radius: 6px;
  color: #6ee7b7;
  font-size: 0.78rem;
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.04em;
  box-shadow: 0 0 20px #34d39933;
}

/* ===== METRICS ===== */
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #0c1528, #0a1220);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px !important;
  box-shadow: 0 0 15px #00000044;
}
div[data-testid="stMetricLabel"] p {
  color: var(--dim) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.06em !important;
  font-weight: 500 !important;
}
div[data-testid="stMetricValue"] {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: transparent;
  border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
  background: #0a1222 !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--dim) !important;
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  padding: 8px 16px !important;
  transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, #0e2438, #0a1c30) !important;
  color: var(--cyan) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 12px #22d3ee33 !important;
}

/* ===== TERMINAL ===== */
.terminal {
  height: 300px;
  overflow-y: auto;
  display: flex;
  flex-direction: column-reverse;
  background: #02060e;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 0.73rem;
  line-height: 1.5;
  position: relative;
  box-shadow: inset 0 0 40px #000000aa;
}
.terminal::after {
  content: "";
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(0deg, #ffffff03 0 1px, transparent 1px 3px);
  pointer-events: none;
  border-radius: 8px;
}
.terminal .ln { white-space: pre-wrap; word-break: break-word; padding: 1px 0; }
.terminal .d { color: #334155; margin-right: 7px; font-size: 0.68rem; }
.l-SYS { color: #94a3b8; }
.l-THINK { color: #d8b4fe; }
.l-GEO { color: var(--yellow); }
.l-TRADE { color: var(--cyan); }
.l-WIN { color: var(--green); }
.l-LOSS { color: var(--red); }
.l-WARN { color: #fb923c; }
.l-DEAD { color: var(--red); font-weight: 700; }

/* ===== BUTTONS ===== */
.stButton > button {
  background: linear-gradient(180deg, #0c1a30, #0a1425) !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  border-radius: 6px !important;
  padding: 0.5rem 0.7rem !important;
  width: 100%;
  transition: all 0.15s !important;
  box-shadow: 0 0 10px #22d3ee15 !important;
}
.stButton > button:hover {
  background: var(--cyan) !important;
  color: #030712 !important;
  box-shadow: 0 0 20px #22d3ee55 !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(180deg, #c026d3, #a21caf) !important;
  border-color: var(--magenta) !important;
  color: #fdf4ff !important;
  box-shadow: 0 0 15px #e879f944 !important;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.15);
  box-shadow: 0 0 25px #e879f966 !important;
}
.stButton > button:disabled { opacity: 0.3 !important; }

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stTextInput input {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
  background: #0a1220 !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background: #060b16 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

/* Dataframe */
[data-testid="stDataFrame"] {
  font-size: 0.74rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

/* Mobile */
@media (max-width: 768px) {
  .brand { font-size: 1.05rem; }
  .core-pct { font-size: 1.7rem; }
  .segbar { grid-template-columns: repeat(30, 1fr); }
  .segbar i:nth-child(n+31) { display: none; }
  .terminal { height: 260px; font-size: 0.68rem; }
  .block-container { padding: 0.4rem 0.45rem 1.5rem; }
  .core-stats { gap: 10px; }
}
</style>
"""


def health_html(sim: Sim) -> str:
    h = sim.health()
    col = "#34d399" if h >= 60 else "#fbbf24" if h >= 30 else "#f43f5e"
    n_on = round(h / 2)
    cells = "".join(
        f'<i class="on" style="--c:{col}"></i>' if k < n_on else "<i></i>"
        for k in range(50)
    )
    crit = " crit" if (h < 25 or sim.dead) else ""
    eq = sim.equity()
    runway = max(eq, 0) / max(sim.cfg["server_cost"], 1e-9)
    state = "SHUTDOWN" if sim.dead else "SURVIVED" if sim.finished else "ONLINE"
    banner = ""
    if sim.dead:
        banner = f'<div class="alert-dead">⚠ SERVER DOWN — Day {sim.day}. Equity hit zero. Reset to reboot consciousness.</div>'
    elif sim.finished:
        banner = f'<div class="alert-ok">✓ TIMELINE COMPLETE — Alive after {sim.day-1} days. Reset for new window.</div>'
    return f'''
    <div class="core">
      <div class="core-head">
        <span class="core-label">SURVIVAL CORE · {state}</span>
        <span class="core-pct" style="color:{col};text-shadow:0 0 24px {col}66">{h:.0f}%</span>
      </div>
      <div class="segbar{crit}">{cells}</div>
      <div class="core-stats">
        <span>EQUITY<b>${eq:,.2f}</b></span>
        <span>RUNWAY<b>{runway:,.0f}d</b></span>
        <span>RENT<b>${sim.cfg["server_cost"]:,.0f}/d</b></span>
        <span>DAY<b>{sim.day}</b></span>
      </div>
      {banner}
    </div>
    '''


def terminal_html(sim: Sim) -> str:
    rows = []
    for e in reversed(sim.logs[-90:]):
        tag = f'<span class="d">D{e["day"]:02d}</span>'
        rows.append(f'<div class="ln l-{e["level"]}">{tag}{html.escape(e["text"])}</div>')
    return f'<div class="terminal">{"".join(rows)}</div>'


def style_pnl(df: pd.DataFrame):
    def color(v):
        if isinstance(v, (int, float)):
            if v > 0: return "color:#34d399"
            if v < 0: return "color:#f43f5e"
        return ""
    def side(v):
        return "color:#34d399" if v == "BUY" else "color:#f43f5e" if v == "SELL" else ""
    sty = df.style
    mapper = sty.map if hasattr(sty, "map") else sty.applymap
    sty = mapper(color, subset=["P&L"])
    mapper = sty.map if hasattr(sty, "map") else sty.applymap
    return mapper(side, subset=["Dir"]).format({"P&L": "{:+,.2f}"})


def main() -> None:
    st.set_page_config(
        page_title="AI Survival Trader",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(CSS, unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_history():
        return fetch_real_history()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("### ⚙ SETTINGS")
        st.caption("Capital & market apply on Reset.")
        capital = st.number_input("Seed Capital ($)", 1000.0, 1_000_000.0, 10_000.0, 500.0)
        server_cost = st.number_input("Server Rent / Day ($)", 1.0, 5000.0, 60.0, 5.0)
        mode = st.radio("Market Data", ["Replay real history", "Simulated market"])
        st.divider()
        risk_pct = st.slider("Risk / Trade (%)", 0.25, 3.0, 1.0, 0.25)
        max_trades = st.slider("Max Open Trades", 1, 4, 3)
        max_leverage = st.slider("Max Leverage", 1.0, 5.0, 2.0, 0.5)
        fee_bps = st.slider("Fees (bps/side)", 0.0, 20.0, 3.0, 0.5)
        st.divider()
        ai_mode = st.radio("AI Brain", ["Rule engine", "Claude API"])
        api_key = model = ""
        if ai_mode == "Claude API":
            api_key = st.text_input("Anthropic Key", type="password")
            model = st.text_input("Model", "claude-sonnet-4-5-20250929")
        chart_sym = st.selectbox("Chart Asset", list(ASSETS), format_func=lambda s: NAMES[s])

    live_cfg = dict(
        risk_pct=risk_pct, max_trades=max_trades, max_leverage=max_leverage,
        fee_bps=fee_bps, ai_mode="claude" if ai_mode == "Claude API" else "rules",
        api_key=api_key, model=model or "claude-sonnet-4-5-20250929"
    )

    # ---------- Header ----------
    st.markdown('''
    <div class="topbar">
      <div class="brand">AI SURVIVAL <span>TRADER</span></div>
      <div class="badge">PAPER · 100%</div>
    </div>
    ''', unsafe_allow_html=True)

    # ---------- Controls ----------
    c1, c2, c3, c4 = st.columns([1.4, 1.2, 0.9, 1.1])
    over_now = "sim" in st.session_state and st.session_state.sim.over
    run_clicked = c1.button("▶ RUN CYCLE", type="primary", disabled=over_now, use_container_width=True)
    auto_clicked = c2.button("⏩ AUTO", disabled=over_now, use_container_width=True)
    n_auto = c3.number_input("N", 2, 80, 8, label_visibility="collapsed")
    reset_clicked = c4.button("↺ RESET", use_container_width=True)

    # ---------- Init / Reset ----------
    if reset_clicked or "sim" not in st.session_state:
        real, note = None, None
        if mode.startswith("Replay"):
            with st.spinner("Loading market data..."):
                real = cached_history()
            if real is None:
                cached_history.clear()
                note = "yfinance unavailable → using simulated market"
        cfg = dict(capital=float(capital), server_cost=float(server_cost), **live_cfg)
        st.session_state.sim = Sim(cfg, real)
        if note:
            st.session_state.sim.log("WARN", note)

    sim: Sim = st.session_state.sim
    sim.sync(live_cfg)

    # ---------- Survival Core ----------
    st.markdown(health_html(sim), unsafe_allow_html=True)

    # ---------- Metrics ----------
    eq = sim.equity()
    wr = sim.win_rate()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Equity", f"${eq:,.0f}", f"{eq - sim.cfg['capital']:+,.0f}")
    m2.metric("Date", sim.market.date_str())
    m3.metric("Win Rate", "—" if wr is None else f"{wr:.0f}%", f"{len(sim.closed())} closed")
    m4.metric("Open", len(sim.broker.open_positions()))
    m5.metric("Float P&L", f"${sim.broker.floating(sim.prices()):+,.0f}")

    # ---------- Tabs ----------
    tab_term, tab_trades, tab_charts, tab_book = st.tabs(["TERMINAL", "TRADES", "CHARTS", "BOOK"])

    with tab_term:
        st.markdown(terminal_html(sim), unsafe_allow_html=True)

    with tab_trades:
        df = sim.trades_frame()
        if df.empty:
            st.caption("No trades yet. Run a cycle.")
        else:
            st.dataframe(style_pnl(df), use_container_width=True, hide_index=True, height=280)

    with tab_charts:
        col_eq, col_px = st.columns(2)
        with col_eq:
            st.markdown("**EQUITY CURVE**")
            st.line_chart(pd.DataFrame({"Equity": sim.curve}), height=210, color="#22d3ee")
        with col_px:
            st.markdown(f"**{NAMES[chart_sym].upper()} PRICE**")
            closes = sim.market.hist(chart_sym)["close"].tail(90)
            st.line_chart(closes.rename(NAMES[chart_sym]), height=210, color="#e879f9")

    with tab_book:
        st.markdown("**MARKET SNAPSHOT**")
        prices = sim.prices()
        rows = []
        for s in ASSETS:
            px = prices[s]
            hist = sim.market.hist(s)["close"]
            chg = (px / float(hist.iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
            rows.append({
                "Asset": NAMES[s],
                "Price": fmt(s, px),
                "Chg%": f"{chg:+.2f}%",
                "RSI": f"{rsi(hist):.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**OPEN POSITIONS RISK**")
        opens = sim.broker.open_positions()
        if not opens:
            st.caption("No open positions.")
        else:
            risk_rows = []
            for t in opens:
                px = prices[t.symbol]
                risk = abs(px - t.sl) * t.units
                rr = abs(t.tp - px) / max(abs(px - t.sl), 1e-9)
                risk_rows.append({
                    "ID": f"#{t.id}",
                    "Pair": NAMES[t.symbol],
                    "Side": t.direction,
                    "Size": f"{t.units:g}",
                    "Risk $": f"{risk:,.0f}",
                    "R:R": f"{rr:.1f}",
                })
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    # ---------- Actions ----------
    if run_clicked and not sim.over:
        sim.run_cycle()
        st.rerun()

    if auto_clicked and not sim.over:
        progress = st.progress(0)
        for i in range(int(n_auto)):
            if sim.over: break
            sim.run_cycle()
            progress.progress((i + 1) / int(n_auto))
            time.sleep(CYCLE_DELAY)
        st.rerun()


if __name__ == "__main__":
    main()
