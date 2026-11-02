import pytest
from app.config import Settings
from app.risk import RiskManager
from app.strategy import Signal

@pytest.fixture
def settings():
    return Settings(RISK_PER_TRADE_PCT=1.0, MAX_RISK_PER_TRADE_PCT=2.0, SMALL_ACCOUNT_THRESHOLD=20.0)

@pytest.fixture
def risk_mgr(settings):
    rm = RiskManager(settings)
    rm.update_balance(1000.0)
    return rm

def test_position_sizing_basic(risk_mgr):
    signal = Signal(
        symbol="BTCUSDT", side="LONG", setup="A", score=80,
        entry_price=50000, sl_price=49000, tp1_price=51000, tp2_price=52000,
        atr=500, regime="TREND_UP"
    )
    # Risk 1% of 1000 = $10. SL distance = 1000. Qty = 10 / 1000 = 0.01
    decision = risk_mgr.calculate_position_size(
        signal, step_size=0.001, min_qty=0.001, min_notional=5.0, price=50000
    )
    assert decision.approved == True
    assert decision.quantity == 0.01

def test_position_sizing_small_account(risk_mgr):
    risk_mgr.update_balance(10.0)
    signal = Signal(
        symbol="BTCUSDT", side="LONG", setup="A", score=80,
        entry_price=50000, sl_price=49000, tp1_price=51000, tp2_price=52000,
        atr=500, regime="TREND_UP"
    )
    # Small account: risk can go up to 5% = $0.50. Qty = 0.50 / 1000 = 0.0005
    # But min_qty is 0.001, which requires $1 risk (10% of balance -> exceeds max cap).
    # So it should be blocked because it exceeds MAX_RISK_PER_TRADE_PCT (2%).
    decision = risk_mgr.calculate_position_size(
        signal, step_size=0.001, min_qty=0.001, min_notional=5.0, price=50000
    )
    assert decision.approved == False
    assert "Cannot meet minNotional" in decision.reason or "Qty" in decision.reason

def test_daily_loss_limit(risk_mgr):
    risk_mgr.daily_pnl = -50.0 # -5% of 1000
    assert risk_mgr.check_daily_loss() == False
