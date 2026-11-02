from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from app.config import Settings
from app.indicators import RollingIndicators, OrderBookImbalance, spread_pct, regime_from_indicators


Side = Literal["LONG", "SHORT"]
Regime = Literal["TREND_UP", "TREND_DOWN", "RANGE", "NO_TRADE"]


@dataclass
class Signal:
    symbol: str
    side: Side
    setup: str  # "A" | "B" | "C" | "D"
    score: float  # 0..100
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    atr: float
    regime: Regime
    reasons: list[str] = field(default_factory=list)
    ai_prob: float = 0.0
    delta: float = 0.0
    imbalance: float = 0.0
    spread: float = 0.0


@dataclass
class MarketSnapshot:
    symbol: str
    last_price: float
    bid: float
    ask: float
    ind_1m: RollingIndicators
    ind_3m: RollingIndicators
    ind_5m: RollingIndicators
    ind_15m: RollingIndicators
    order_book_bids: list[list[str]] = field(default_factory=list)
    order_book_asks: list[list[str]] = field(default_factory=list)
    funding_rate: float = 0.0
    oi_change_pct: float = 0.0
    tape_speed: float = 0.0
    large_print: float = 0.0
    data_age_seconds: float = 0.0


class StrategyEngine:
    """Regime-adaptive scalping strategy with 4 setups + order-flow scoring."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ob_imbalance = OrderBookImbalance(levels=10)
        self.enabled_setups: dict[str, bool] = {"A": True, "B": True, "C": True, "D": True}

    def _cost_gate(self, tp_distance: float, price: float, spread: float, fee_rate: float = 0.0004) -> bool:
        rt_fees = 2 * fee_rate
        expected_slippage = self.settings.MAX_SLIPPAGE_PCT / 100
        cost_pct = rt_fees + (spread / 100) + expected_slippage
        cost_abs = cost_pct * price
        return tp_distance >= self.settings.COST_GATE_MULTIPLIER * cost_abs

    def _flow_score(self, snap: MarketSnapshot, side: Side) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        
        delta_30 = snap.ind_1m.cvd(30)
        slope_30 = snap.ind_1m.cvd_slope(30)
        if side == "LONG" and delta_30 > 0 and slope_30 > 0:
            score += 0.25
            reasons.append("delta+")
        elif side == "SHORT" and delta_30 < 0 and slope_30 < 0:
            score += 0.25
            reasons.append("delta-")

        imb = self.ob_imbalance.compute(snap.order_book_bids, snap.order_book_asks)
        if side == "LONG" and imb > 0.15:
            score += 0.20
            reasons.append(f"imb+{imb:.2f}")
        elif side == "SHORT" and imb < -0.15:
            score += 0.20
            reasons.append(f"imb{imb:.2f}")

        if snap.tape_speed > 3.0:
            score += 0.10
            reasons.append("tape")

        if snap.large_print > 0:
            score += 0.10
            reasons.append("print")

        if side == "LONG" and snap.oi_change_pct > 0.5:
            score += 0.10
            reasons.append("oi+")
        elif side == "SHORT" and snap.oi_change_pct < -0.5:
            score += 0.10
            reasons.append("oi-")

        if side == "LONG" and snap.funding_rate > 0.0005:
            score -= 0.10
            reasons.append("funding-warn")
        elif side == "SHORT" and snap.funding_rate < -0.0005:
            score -= 0.10
            reasons.append("funding-warn")

        return max(0.0, min(1.0, score)), reasons

    def _higher_tf_alignment(self, snap: MarketSnapshot, side: Side) -> float:
        ema21 = snap.ind_15m.ema(21)
        ema50 = snap.ind_15m.ema(50)
        if ema21 is None or ema50 is None:
            return 0.0
        if side == "LONG" and ema21 > ema50:
            return 1.0
        if side == "SHORT" and ema21 < ema50:
            return 1.0
        return 0.0

    def _setup_A_pullback(self, snap: MarketSnapshot, regime: Regime) -> Signal | None:
        if regime not in ("TREND_UP", "TREND_DOWN"):
            return None
        side: Side = "LONG" if regime == "TREND_UP" else "SHORT"
        price = snap.last_price
        ema21 = snap.ind_5m.ema(21)
        vwap = snap.ind_5m.vwap()
        rsi = snap.ind_5m.rsi(7)
        atr = snap.ind_5m.atr(14)
        if None in (ema21, vwap, rsi, atr) or atr == 0:
            return None
        ref = (ema21 + vwap) / 2
        if abs(price - ref) / price > 0.004:
            return None
        reasons = ["A:pullback"]
        if side == "LONG" and rsi < 40:
            reasons.append(f"rsi-reset({rsi:.0f})")
        elif side == "SHORT" and rsi > 60:
            reasons.append(f"rsi-reset({rsi:.0f})")
        else:
            return None
        closes = list(snap.ind_5m.closes)
        if len(closes) < 2:
            return None
        if side == "LONG" and closes[-1] <= closes[-2]:
            return None
        if side == "SHORT" and closes[-1] >= closes[-2]:
            return None

        sl_dist = self.settings.ATR_SL_MULTIPLIER * atr
        sl = price - sl_dist if side == "LONG" else price + sl_dist
        tp1 = price + self.settings.TP1_R_MULTIPLE * sl_dist * (1 if side == "LONG" else -1)
        tp2 = price + self.settings.TP2_R_MULTIPLE * sl_dist * (1 if side == "LONG" else -1)
        return Signal(symbol=snap.symbol, side=side, setup="A", score=0.0, entry_price=price, sl_price=sl, tp1_price=tp1, tp2_price=tp2, atr=atr, regime=regime, reasons=reasons)

    def _setup_B_mean_reversion(self, snap: MarketSnapshot, regime: Regime) -> Signal | None:
        if regime != "RANGE":
            return None
        bb = snap.ind_5m.bollinger(20, 2.0)
        rsi = snap.ind_5m.rsi(7)
        vwap = snap.ind_5m.vwap()
        atr = snap.ind_5m.atr(14)
        if bb is None or rsi is None or vwap is None or atr is None or atr == 0:
            return None
        lower, mid, upper = bb
        price = snap.last_price
        side: Side | None = None
        reasons = ["B:range-mr"]
        if price <= lower and rsi < 30 and price < vwap * 0.998:
            side = "LONG"
            reasons.append(f"bb-touch+{rsi:.0f}")
        elif price >= upper and rsi > 70 and price > vwap * 1.002:
            side = "SHORT"
            reasons.append(f"bb-touch-{100 - rsi:.0f}")
        else:
            return None
        sl_dist = max(self.settings.ATR_SL_MULTIPLIER * atr, (upper - lower) * 0.6)
        sl = price - sl_dist if side == "LONG" else price + sl_dist
        tp1 = mid
        tp2 = upper if side == "LONG" else lower
        return Signal(symbol=snap.symbol, side=side, setup="B", score=0.0, entry_price=price, sl_price=sl, tp1_price=tp1, tp2_price=tp2, atr=atr, regime=regime, reasons=reasons)

    def _setup_C_breakout(self, snap: MarketSnapshot, regime: Regime) -> Signal | None:
        if regime == "NO_TRADE":
            return None
        closes = list(snap.ind_5m.closes)
        highs = list(snap.ind_5m.highs)
        lows = list(snap.ind_5m.lows)
        if len(closes) < 30:
            return None
        don_high = max(highs[-20:-1])
        don_low = min(lows[-20:-1])
        price = snap.last_price
        vol_z = snap.ind_5m.volume_zscore(20)
        atr_now = snap.ind_5m.atr(14)
        if vol_z is None or atr_now is None or atr_now == 0 or vol_z < 1.5:
            return None
        side: Side | None = None
        reasons = ["C:breakout", f"volZ={vol_z:.1f}"]
        if price > don_high and regime in ("TREND_UP", "RANGE"):
            side = "LONG"
        elif price < don_low and regime in ("TREND_DOWN", "RANGE"):
            side = "SHORT"
        else:
            return None
        sl_dist = self.settings.ATR_SL_MULTIPLIER * atr_now
        sl = price - sl_dist if side == "LONG" else price + sl_dist
        tp1 = price + self.settings.TP1_R_MULTIPLE * sl_dist * (1 if side == "LONG" else -1)
        tp2 = price + self.settings.TP2_R_MULTIPLE * sl_dist * (1 if side == "LONG" else -1)
        return Signal(symbol=snap.symbol, side=side, setup="C", score=0.0, entry_price=price, sl_price=sl, tp1_price=tp1, tp2_price=tp2, atr=atr_now, regime=regime, reasons=reasons)

    def _setup_D_liquidation_fade(self, snap: MarketSnapshot, regime: Regime) -> Signal | None:
        vol_z = snap.ind_5m.volume_zscore(20)
        bb = snap.ind_5m.bollinger(20, 2.5)
        rsi = snap.ind_1m.rsi(7)
        atr = snap.ind_5m.atr(14)
        if vol_z is None or bb is None or rsi is None or atr is None or atr == 0 or vol_z < 2.5:
            return None
        lower, mid, upper = bb
        price = snap.last_price
        side: Side | None = None
        reasons = ["D:liq-fade", f"volZ={vol_z:.1f}"]
        if price < lower and rsi < 20:
            side = "LONG"
            reasons.append("wick-low")
        elif price > upper and rsi > 80:
            side = "SHORT"
            reasons.append("wick-high")
        else:
            return None
        sl_dist = max(self.settings.ATR_SL_MULTIPLIER * atr, atr * 1.5)
        sl = price - sl_dist if side == "LONG" else price + sl_dist
        tp1 = mid
        tp2 = price + 2 * sl_dist * (1 if side == "LONG" else -1)
        return Signal(symbol=snap.symbol, side=side, setup="D", score=0.0, entry_price=price, sl_price=sl, tp1_price=tp1, tp2_price=tp2, atr=atr, regime=regime, reasons=reasons)

    def evaluate(self, snap: MarketSnapshot) -> Signal | None:
        if snap.data_age_seconds > 3.0 or snap.bid <= 0 or snap.ask <= 0:
            return None
        regime = regime_from_indicators(snap.ind_5m)
        if regime == "NO_TRADE":
            return None

        candidates: list[Signal] = []
        if self.enabled_setups.get("A"):
            s = self._setup_A_pullback(snap, regime)
            if s: candidates.append(s)
        if self.enabled_setups.get("B"):
            s = self._setup_B_mean_reversion(snap, regime)
            if s: candidates.append(s)
        if self.enabled_setups.get("C"):
            s = self._setup_C_breakout(snap, regime)
            if s: candidates.append(s)
        if self.enabled_setups.get("D"):
            s = self._setup_D_liquidation_fade(snap, regime)
            if s: candidates.append(s)

        best: Signal | None = None
        best_score = 0.0
        for sig in candidates:
            flow, flow_reasons = self._flow_score(snap, sig.side)
            if flow < 0.35: continue
            htf = self._higher_tf_alignment(snap, sig.side)
            
            setup_w = 0.30 if sig.setup == "A" else 0.25
            score = (setup_w * 100) + (0.15 * 100) + (0.30 * 100 * flow) + (0.15 * 100 * htf) + (0.10 * 100 * sig.ai_prob)
            
            sp = spread_pct(snap.bid, snap.ask)
            tp_distance = abs(sig.tp1_price - sig.entry_price)
            if not self._cost_gate(tp_distance, sig.entry_price, sp):
                sig.reasons.append("skip:cost-gate")
                continue
                
            sig.score = min(100.0, score)
            sig.reasons.extend(flow_reasons)
            sig.delta = snap.ind_1m.cvd(30)
            sig.imbalance = self.ob_imbalance.compute(snap.order_book_bids, snap.order_book_asks)
            sig.spread = sp
            
            if sig.score > best_score:
                best_score = sig.score
                best = sig

        if best and best.score >= self.settings.SIGNAL_MIN_SCORE:
            if best.ai_prob < self.settings.AI_MIN_PROB and best.ai_prob > 0:
                best.reasons.append("skip:ai-low")
                return None
            return best
        return None
