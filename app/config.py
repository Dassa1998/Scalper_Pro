from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- mode ----
    APP_MODE: Literal["paper", "testnet", "mainnet"] = "paper"
    CONFIRM_LIVE_TRADING: bool = False

    # ---- binance ----
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET_API_KEY: str = ""
    BINANCE_TESTNET_API_SECRET: str = ""
    BINANCE_BASE_URL_MAINNET: str = "https://fapi.binance.com"
    BINANCE_BASE_URL_TESTNET: str = "https://demo-fapi.binance.com"
    BINANCE_WS_BASE_MAINNET: str = "wss://fstream.binance.com"
    BINANCE_WS_BASE_TESTNET: str = "wss://demo-fstream.binance.com"

    # ---- db ----
    DATABASE_URL: str = "sqlite+aiosqlite:///./scalper.db"

    # ---- auth ----
    DASHBOARD_PASSWORD_HASH: str = ""
    SECRET_KEY: str = Field(default_factory=lambda: os.urandom(32).hex())
    SESSION_COOKIE_SECURE: bool = True

    # ---- risk ----
    RISK_PER_TRADE_PCT: float = 0.5
    MAX_RISK_PER_TRADE_PCT: float = 2.0
    SMALL_ACCOUNT_THRESHOLD: float = 20.0
    SMALL_ACCOUNT_MAX_RISK_PCT: float = 5.0
    MAX_CONCURRENT_POSITIONS: int = 3
    DAILY_LOSS_LIMIT_PCT: float = 3.0
    MAX_DRAWDOWN_PCT: float = 10.0
    CONSECUTIVE_LOSS_COOLDOWN: int = 3

    # ---- strategy ----
    DEFAULT_LEVERAGE: int = 7
    ATR_SL_MULTIPLIER: float = 1.2
    TP1_R_MULTIPLE: float = 1.0
    TP2_R_MULTIPLE: float = 2.0
    SIGNAL_MIN_SCORE: int = 60
    AI_MIN_PROB: float = 0.55
    MAX_SLIPPAGE_PCT: float = 0.15
    COST_GATE_MULTIPLIER: float = 3.0

    # ---- telegram ----
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    @field_validator("MAX_RISK_PER_TRADE_PCT")
    @classmethod
    def _cap_risk(cls, v: float) -> float:
        if v > 2.0:
            raise ValueError("MAX_RISK_PER_TRADE_PCT hard cap is 2.0")
        return v

    @property
    def is_live(self) -> bool:
        return self.APP_MODE in ("testnet", "mainnet")

    @property
    def is_mainnet(self) -> bool:
        return self.APP_MODE == "mainnet"

    @property
    def api_key(self) -> str:
        return self.BINANCE_TESTNET_API_KEY if self.APP_MODE == "testnet" else self.BINANCE_API_KEY

    @property
    def api_secret(self) -> str:
        return self.BINANCE_TESTNET_API_SECRET if self.APP_MODE == "testnet" else self.BINANCE_API_SECRET

    @property
    def rest_base(self) -> str:
        return self.BINANCE_BASE_URL_TESTNET if self.APP_MODE == "testnet" else self.BINANCE_BASE_URL_MAINNET

    @property
    def ws_base(self) -> str:
        return self.BINANCE_WS_BASE_TESTNET if self.APP_MODE == "testnet" else self.BINANCE_WS_BASE_MAINNET

    def validate_live(self) -> None:
        if self.APP_MODE == "mainnet":
            if not self.CONFIRM_LIVE_TRADING:
                raise RuntimeError("MAINNET requires CONFIRM_LIVE_TRADING=true")
            if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
                raise RuntimeError("MAINNET requires BINANCE_API_KEY and BINANCE_API_SECRET")
        if self.APP_MODE == "testnet":
            if not self.BINANCE_TESTNET_API_KEY:
                raise RuntimeError("TESTNET requires BINANCE_TESTNET_API_KEY")
        if self.RISK_PER_TRADE_PCT > self.MAX_RISK_PER_TRADE_PCT:
            raise RuntimeError("RISK_PER_TRADE_PCT exceeds hard cap")


@lru_cache
def get_settings() -> Settings:
    return Settings()
