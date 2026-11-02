from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

from app.config import Settings
from app.binance_client import BinanceFuturesClient
from app.strategy import StrategyEngine, MarketSnapshot
from app.risk import RiskManager
from app.ai_model import AIModel
from app.indicators import RollingIndicators

log = logging.getLogger(__name__)

State = Literal["IDLE", "SIGNAL", "ENTRY_PENDING", "OPEN", "EXITING", "COOLDOWN"]

class TradingEngine:
    def __init__(self, settings: Settings, client: BinanceFuturesClient):
        self.settings = settings
        self.client = client
        self.strategy = StrategyEngine(settings)
        self.risk = RiskManager(settings)
        self.ai = AIModel()
        
        self.is_running = False
        self.state: State = "IDLE"
        self._task: asyncio.Task | None = None
        
        # Live in-memory data
        self.indicators: dict[str, dict[str, RollingIndicators]] = {}
        self.latest_prices: dict[str, float] = {}
        self.last_update: dict[str, float] = {}
        self.open_positions: dict[str, dict] = {}
        self.logs: list[str] = []

    def log_event(self, msg: str):
        log.info(msg)
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")
        if len(self.logs) > 100:
            self.logs.pop(0)

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.log_event("Engine starting...")
        self.ai.load()
        await self.client.start()
        
        # Start background tasks
        self._task = asyncio.create_task(self._run_loop())
        self.log_event(f"Engine started in {self.settings.APP_MODE.upper()} mode.")

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        self.log_event("Engine stopped.")

    async def panic(self):
        self.log_event("PANIC! Closing all positions and stopping engine.")
        self.is_running = False
        if self._task:
            self._task.cancel()
            
        # In real mode, we must close all positions and cancel orders via API
        if self.settings.is_live:
            try:
                positions = await self.client.position_risk()
                for pos in positions:
                    amt = float(pos["positionAmt"])
                    if amt != 0:
                        symbol = pos["symbol"]
                        side = "SELL" if amt > 0 else "BUY"
                        await self.client.new_order(
                            symbol=symbol, side=side, type="MARKET", quantity=abs(amt), reduceOnly=True
                        )
                        await self.client.cancel_all_orders(symbol)
                self.log_event("Panic execution complete. All positions flattened.")
            except Exception as e:
                self.log_event(f"PANIC ERROR: {e}")
        self.state = "IDLE"

    async def _run_loop(self):
        """Main loop: Fetch data, evaluate, execute."""
        # For demo/simplicity in this file, we simulate the market data loop.
        # A full production bot uses WebSockets (binance_client.stream_forever)
        # Here we use a polling REST approach to guarantee it works on first boot.
        
        while self.is_running:
            try:
                # 1. Fetch account balance
                if self.settings.is_live:
                    balances = await self.client.balance()
                    usdt = next((b for b in balances if b["asset"] == "USDT"), None)
                    if usdt:
                        self.risk.update_balance(float(usdt["balance"]))
                else:
                    self.risk.update_balance(1000.0) # Paper balance

                # 2. Fetch market data (BTCUSDT example)
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
                for sym in symbols:
                    klines = await self.client.klines(sym, "5m", limit=100)
                    if sym not in self.indicators:
                        self.indicators[sym] = {
                            "1m": RollingIndicators(), "5m": RollingIndicators(), "15m": RollingIndicators()
                        }
                    
                    ind = self.indicators[sym]["5m"]
                    for k in klines:
                        # k[4] = close, k[2] = high, k[3] = low, k[5] = volume
                        if len(ind.closes) == 0 or ind.closes[-1] != float(k[4]):
                            ind.update_candle(float(k[4]), float(k[2]), float(k[3]), float(k[5]))
                    
                    ticker = await self.client.mark_price(sym)
                    price = float(ticker["markPrice"])
                    self.latest_prices[sym] = price
                    self.last_update[sym] = time.time()
                    
                    # 3. Evaluate strategy
                    snap = MarketSnapshot(
                        symbol=sym, last_price=price, bid=price*0.999, ask=price*1.001,
                        ind_1m=self.indicators[sym]["1m"],
                        ind_3m=self.indicators[sym]["5m"],
                        ind_5m=ind,
                        ind_15m=self.indicators[sym]["15m"],
                        data_age_seconds=time.time() - self.last_update[sym]
                    )
                    
                    signal = self.strategy.evaluate(snap)
                    if signal and sym not in self.open_positions and self.risk.open_positions_count < 3:
                        self.log_event(f"Signal found: {signal.side} {sym} | Score: {signal.score:.1f}")
                        self.risk.open_positions_count += 1
                        self.open_positions[sym] = {
                            "side": signal.side, "entry": price, "qty": 0.1, "sl": signal.sl_price
                        }
                        self.log_event(f"Executed {signal.side} on {sym} (Paper: {not self.settings.is_live})")

                await asyncio.sleep(5) # Poll every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_event(f"Loop error: {e}")
                await asyncio.sleep(10)
