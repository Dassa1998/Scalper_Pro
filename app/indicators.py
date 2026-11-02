from __future__ import annotations

from collections import deque

import numpy as np


class RollingIndicators:
    """
    Rolling-buffer numpy indicators. No full-DataFrame recompute per tick.
    """

    def __init__(self, maxlen: int = 500):
        self.maxlen = maxlen
        self.closes: deque[float] = deque(maxlen=maxlen)
        self.highs: deque[float] = deque(maxlen=maxlen)
        self.lows: deque[float] = deque(maxlen=maxlen)
        self.volumes: deque[float] = deque(maxlen=maxlen)
        self.aggr_buys: deque[float] = deque(maxlen=maxlen)
        self.aggr_sells: deque[float] = deque(maxlen=maxlen)

    def update_candle(self, close: float, high: float, low: float, volume: float) -> None:
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)

    def update_trade(self, qty: float, is_buyer_maker: bool) -> None:
        if is_buyer_maker:
            self.aggr_sells.append(qty)
        else:
            self.aggr_buys.append(qty)

    # ---------- basics ----------
    def _arr(self, d: deque) -> np.ndarray:
        return np.fromiter(d, dtype=np.float64)

    def ema(self, period: int) -> float | None:
        arr = self._arr(self.closes)
        if len(arr) < period:
            return None
        alpha = 2 / (period + 1)
        ema = arr[0]
        for x in arr[1:]:
            ema = alpha * x + (1 - alpha) * ema
        return float(ema)

    def rsi(self, period: int = 7) -> float | None:
        arr = self._arr(self.closes)
        if len(arr) < period + 1:
            return None
        diffs = np.diff(arr[-(period + 1):])
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def atr(self, period: int = 14) -> float | None:
        if len(self.closes) < period + 1:
            return None
        highs = self._arr(self.highs)[-(period + 1):]
        lows = self._arr(self.lows)[-(period + 1):]
        closes = self._arr(self.closes)[-(period + 1):]
        trs = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        return float(trs.mean())

    def atr_pct(self, period: int = 14) -> float | None:
        a = self.atr(period)
        if a is None or not self.closes:
            return None
        last = self.closes[-1]
        return a / last * 100 if last else None

    def bollinger(self, period: int = 20, num_std: float = 2.0) -> tuple[float, float, float] | None:
        arr = self._arr(self.closes)
        if len(arr) < period:
            return None
        window = arr[-period:]
        mid = float(window.mean())
        std = float(window.std())
        return mid - num_std * std, mid, mid + num_std * std

    def bollinger_width_pct(self, period: int = 20, num_std: float = 2.0) -> float | None:
        bb = self.bollinger(period, num_std)
        if bb is None:
            return None
        lower, mid, upper = bb
        return (upper - lower) / mid * 100 if mid else None

    def adx(self, period: int = 14) -> float | None:
        if len(self.closes) < period * 2:
            return None
        highs = self._arr(self.highs)
        lows = self._arr(self.lows)
        closes = self._arr(self.closes)
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(
            np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        up = highs[1:] - highs[:-1]
        down = lows[:-1] - lows[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_s = np.convolve(tr, np.ones(period) / period, mode="valid")
        plus_di = 100 * np.convolve(plus_dm, np.ones(period) / period, mode="valid") / (atr_s + 1e-12)
        minus_di = 100 * np.convolve(minus_dm, np.ones(period) / period, mode="valid") / (atr_s + 1e-12)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        adx_arr = np.convolve(dx, np.ones(period) / period, mode="valid")
        return float(adx_arr[-1]) if len(adx_arr) else None

    def vwap(self) -> float | None:
        if not self.closes:
            return None
        closes = self._arr(self.closes)
        vols = self._arr(self.volumes)
        if vols.sum() == 0:
            return float(closes[-1])
        return float((closes * vols).sum() / vols.sum())

    def volume_zscore(self, period: int = 20) -> float | None:
        arr = self._arr(self.volumes)
        if len(arr) < period:
            return None
        window = arr[-period:]
        mu = window.mean()
        sd = window.std()
        if sd == 0:
            return 0.0
        return float((arr[-1] - mu) / sd)

    def cvd(self, seconds: int = 30) -> float:
        buys = sum(list(self.aggr_buys)[-seconds:])
        sells = sum(list(self.aggr_sells)[-seconds:])
        return buys - sells

    def cvd_slope(self, seconds: int = 30) -> float:
        if len(self.aggr_buys) < seconds:
            return 0.0
        buys = np.fromiter(list(self.aggr_buys)[-seconds:], dtype=float)
        sells = np.fromiter(list(self.aggr_sells)[-seconds:], dtype=float)
        cvd_series = np.cumsum(buys - sells)
        x = np.arange(len(cvd_series))
        slope = np.polyfit(x, cvd_series, 1)[0]
        return float(slope)


class OrderBookImbalance:
    def __init__(self, levels: int = 10):
        self.levels = levels

    def compute(self, bids: list[list[str]], asks: list[list[str]]) -> float:
        bid_vol = sum(float(q) for _, q in bids[: self.levels])
        ask_vol = sum(float(q) for _, q in asks[: self.levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 100 if mid else 0.0


def regime_from_indicators(ind: RollingIndicators) -> str:
    """Return TREND_UP / TREND_DOWN / RANGE / NO_TRADE."""
    adx = ind.adx(14)
    ema21 = ind.ema(21)
    ema50 = ind.ema(50)
    ema200 = ind.ema(200)
    atr_pct = ind.atr_pct(14)
    bbw = ind.bollinger_width_pct()

    if adx is None or ema21 is None or ema50 is None or ema200 is None:
        return "NO_TRADE"
    if atr_pct is not None and atr_pct < 0.15:
        return "NO_TRADE"
    if bbw is not None and bbw < 0.5:
        return "NO_TRADE"

    if adx > 25 and ema21 > ema50 > ema200:
        return "TREND_UP"
    if adx > 25 and ema21 < ema50 < ema200:
        return "TREND_DOWN"
    if adx < 20:
        return "RANGE"
    return "NO_TRADE"
