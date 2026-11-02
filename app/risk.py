from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from app.config import Settings
from app.strategy import Signal


@dataclass
class RiskDecision:
    approved: bool
    quantity: float = 0.0
    reason: str = ""
    leverage: int = 1


class RiskManager:
    """Enforces risk caps, position sizing, and limits."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.consecutive_losses = 0
        self.open_positions_count = 0

    def update_balance(self, balance: float) -> None:
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance

    def check_daily_loss(self) -> bool:
        if self.current_balance <= 0:
            return False
        loss_pct = (self.daily_pnl / self.current_balance) * 100
        return loss_pct > -self.settings.DAILY_LOSS_LIMIT_PCT

    def check_max_drawdown(self) -> bool:
        if self.peak_balance <= 0:
            return True
        dd_pct = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
        return dd_pct < self.settings.MAX_DRAWDOWN_PCT

    def calculate_position_size(
        self,
        signal: Signal,
        step_size: float,
        min_qty: float,
        min_notional: float,
        price: float,
    ) -> RiskDecision:
        """Calculate quantity based on risk %, SL distance, and exchange filters."""
        if self.open_positions_count >= self.settings.MAX_CONCURRENT_POSITIONS:
            return RiskDecision(approved=False, reason="Max concurrent positions reached")

        if not self.check_daily_loss():
            return RiskDecision(approved=False, reason="Daily loss limit hit")

        if not self.check_max_drawdown():
            return RiskDecision(approved=False, reason="Max drawdown hit")

        if self.consecutive_losses >= self.settings.CONSECUTIVE_LOSS_COOLDOWN:
            return RiskDecision(approved=False, reason="Consecutive loss cooldown active")

        risk_pct = self.settings.RISK_PER_TRADE_PCT
        # Small account mode
        if self.current_balance < self.settings.SMALL_ACCOUNT_THRESHOLD:
            # Allow up to SMALL_ACCOUNT_MAX_RISK_PCT but never exceed hard cap
            risk_pct = min(self.settings.SMALL_ACCOUNT_MAX_RISK_PCT, self.settings.MAX_RISK_PER_TRADE_PCT)

        risk_amount = self.current_balance * (risk_pct / 100)
        sl_distance = abs(signal.entry_price - signal.sl_price)

        if sl_distance == 0:
            return RiskDecision(approved=False, reason="SL distance is zero")

        raw_qty = risk_amount / sl_distance

        # Round to step size
        qty = math.floor(raw_qty / step_size) * step_size
        # Format to avoid float precision issues (assume 8 decimals max)
        qty = float(f"{qty:.8f}")

        if qty < min_qty:
            return RiskDecision(approved=False, reason=f"Qty {qty} < minQty {min_qty}")

        notional = qty * price
        if notional < min_notional:
            # Try to increase to minNotional if it doesn't exceed hard risk cap
            req_qty = math.ceil((min_notional / price) / step_size) * step_size
            req_qty = float(f"{req_qty:.8f}")
            req_risk = req_qty * sl_distance
            req_risk_pct = (req_risk / self.current_balance) * 100
            
            if req_risk_pct <= self.settings.MAX_RISK_PER_TRADE_PCT:
                qty = req_qty
                notional = qty * price
            else:
                return RiskDecision(approved=False, reason="Cannot meet minNotional within risk cap")

        # Calculate leverage needed (default settings, capped by liquidation safety)
        leverage = self.settings.DEFAULT_LEVERAGE
        # Ensure SL is far from liquidation
        # Liq price approx: entry * (1 - 1/lev) for long
        liq_dist_pct = (1 / leverage) * 100
        sl_dist_pct = (sl_distance / price) * 100
        if sl_dist_pct > (liq_dist_pct * 0.5):
            # Too close to liq, reduce leverage
            safe_lev = int(100 / (sl_dist_pct * 2))
            leverage = max(1, min(leverage, safe_lev))

        return RiskDecision(approved=True, quantity=qty, leverage=leverage)
