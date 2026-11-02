from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Callable
from urllib.parse import urlencode

import aiohttp

log = logging.getLogger(__name__)


class BinanceError(Exception):
    def __init__(self, code: int, msg: str, status: int = 0):
        self.code = code
        self.msg = msg
        self.status = status
        super().__init__(f"[{code}] {msg} (HTTP {status})")


class BinanceFuturesClient:
    """Thin async USDⓈ-M Futures client: REST (HMAC) + WebSocket streams."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        rest_base: str,
        ws_base: str,
        recv_window: int = 5000,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_base = rest_base.rstrip("/")
        self.ws_base = ws_base.rstrip("/")
        self.recv_window = recv_window
        self._session: aiohttp.ClientSession | None = None
        self._time_offset = 0
        self._weight_used = 0
        self._order_count_10s = 0
        self._order_count_1m = 0
        self._lock = asyncio.Lock()

    # ---------- lifecycle ----------
    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {},
            )
        await self.sync_time()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def sync_time(self) -> None:
        data = await self._request("GET", "/fapi/v1/time", signed=False)
        server_ms = data["serverTime"]
        local_ms = int(time.time() * 1000)
        self._time_offset = server_ms - local_ms
        log.info("time offset synced: %d ms", self._time_offset)

    def _ts(self) -> int:
        return int(time.time() * 1000) + self._time_offset

    # ---------- signing ----------
    def _sign(self, params: dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        sig = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    # ---------- low-level REST ----------
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = True,
        max_retries: int = 4,
    ) -> Any:
        if self._session is None:
            await self.start()
        assert self._session is not None

        params = dict(params or {})
        if signed:
            if not self.api_key:
                raise RuntimeError("API key required for signed request")
            params["timestamp"] = self._ts()
            params["recvWindow"] = self.recv_window

        url = f"{self.rest_base}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                if signed:
                    body = self._sign(params)
                    async with self._session.request(
                        method, url, data=body,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    ) as resp:
                        return await self._handle_response(resp)
                else:
                    async with self._session.request(method, url, params=params) as resp:
                        return await self._handle_response(resp)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt >= max_retries:
                    raise
                backoff = min(2 ** attempt, 30)
                log.warning("REST %s %s failed (%s), retry %d/%d in %ds", method, path, e, attempt, max_retries, backoff)
                await asyncio.sleep(backoff)

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        self._track_headers(resp.headers)
        status = resp.status
        text = await resp.text()
        if status == 200:
            return json.loads(text) if text else {}
        if status in (429, 418):
            retry_after = float(resp.headers.get("Retry-After", "5"))
            log.error("rate limited HTTP %d, backing off %.1fs", status, retry_after)
            await asyncio.sleep(retry_after)
            raise BinanceError(-1003, "rate limited", status)
        try:
            data = json.loads(text)
            raise BinanceError(int(data.get("code", -1)), data.get("msg", text), status)
        except json.JSONDecodeError:
            raise BinanceError(-1, text, status)

    def _track_headers(self, headers: Any) -> None:
        try:
            self._weight_used = int(headers.get("x-mbx-used-weight-1m", self._weight_used))
            self._order_count_10s = int(headers.get("X-MBX-ORDER-COUNT-10S", self._order_count_10s))
            self._order_count_1m = int(headers.get("X-MBX-ORDER-COUNT-1M", self._order_count_1m))
        except (TypeError, ValueError):
            pass

    async def _ensure_capacity(self) -> None:
        async with self._lock:
            if self._weight_used > 1100:
                log.warning("weight %d near limit, sleeping 10s", self._weight_used)
                await asyncio.sleep(10)

    # ---------- public market data ----------
    async def exchange_info(self) -> dict:
        return await self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    async def ticker_24hr(self) -> list[dict]:
        return await self._request("GET", "/fapi/v1/ticker/24hr", signed=False)

    async def mark_price(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol}, signed=False)

    async def klines(self, symbol: str, interval: str, limit: int = 500, end_time: int | None = None) -> list[list]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        return await self._request("GET", "/fapi/v1/klines", params, signed=False)

    async def depth(self, symbol: str, limit: int = 20) -> dict:
        return await self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit}, signed=False)

    async def funding_rate(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1}, signed=False)

    async def open_interest(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol}, signed=False)

    # ---------- account ----------
    async def account(self) -> dict:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def balance(self) -> list[dict]:
        return await self._request("GET", "/fapi/v2/balance", signed=True)

    async def position_risk(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/fapi/v2/positionRisk", params, signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        return await self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)

    async def position_mode(self) -> dict:
        return await self._request("GET", "/fapi/v1/positionSide/dual", signed=True)

    # ---------- orders ----------
    async def new_order(self, **kwargs: Any) -> dict:
        await self._ensure_capacity()
        return await self._request("POST", "/fapi/v1/order", kwargs, signed=True)

    async def cancel_order(self, symbol: str, order_id: int | None = None, client_order_id: str | None = None) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return await self._request("DELETE", "/fapi/v1/order", params, signed=True)

    async def get_order(self, symbol: str, order_id: int | None = None, client_order_id: str | None = None) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return await self._request("GET", "/fapi/v1/order", params, signed=True)

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    # ---------- ALGO ORDERS (SL / TP / trailing) ----------
    async def new_algo_order(self, **kwargs: Any) -> dict:
        """POST /fapi/v1/algoOrder with algoType=CONDITIONAL."""
        kwargs.setdefault("algoType", "CONDITIONAL")
        await self._ensure_capacity()
        return await self._request("POST", "/fapi/v1/algoOrder", kwargs, signed=True)

    async def cancel_algo_order(self, algo_id: int, symbol: str) -> dict:
        return await self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id, "symbol": symbol}, signed=True)

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/fapi/v1/openAlgoOrders", params, signed=True)

    # ---------- user data stream ----------
    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", signed=False)
        return data["listenKey"]

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key}, signed=False)

    async def close_listen_key(self, listen_key: str) -> None:
        await self._request("DELETE", "/fapi/v1/listenKey", {"listenKey": listen_key}, signed=False)

    # ---------- WebSocket streams ----------
    async def stream_forever(
        self,
        category: str,
        streams: list[str],
        on_message: Callable[[dict], Any],
        on_reconnect: Callable[[], Any] | None = None,
    ) -> None:
        """
        category: "public" | "market"
        Uses NEW split endpoints: /public/ws or /market/ws
        Auto-reconnect with backoff + re-subscribe.
        """
        assert category in ("public", "market")
        attempt = 0
        while True:
            url = f"{self.ws_base}/{category}/ws"
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.ws_connect(url, heartbeat=30) as ws:
                        sub = {"method": "SUBSCRIBE", "params": streams, "id": int(time.time())}
                        await ws.send_json(sub)
                        log.info("WS %s connected, subscribed %d streams", category, len(streams))
                        attempt = 0
                        if on_reconnect:
                            await on_reconnect()
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if "result" in data and data.get("id"):
                                    continue
                                await on_message(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                attempt += 1
                backoff = min(2 ** attempt, 60)
                log.warning("WS %s error %s — reconnect in %ds", category, e, backoff)
                await asyncio.sleep(backoff)

    async def user_stream_forever(
        self,
        on_message: Callable[[dict], Any],
        on_reconnect: Callable[[], Any] | None = None,
    ) -> None:
        """Private stream via /private/ws?listenKey=... with keepalive + 24h reconnect."""
        listen_key = await self.create_listen_key()
        last_keepalive = time.time()
        while True:
            url = f"{self.ws_base}/private/ws?listenKey={listen_key}"
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.ws_connect(url, heartbeat=30) as ws:
                        log.info("private WS connected")
                        if on_reconnect:
                            await on_reconnect()
                        async for msg in ws:
                            if msg.type == aiohttp.WebSocketMsgType.TEXT if hasattr(aiohttp, 'WebSocketMsgType') else (aiohttp.WSMsgType.TEXT,):
                                pass
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await on_message(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            if time.time() - last_keepalive > 1800:
                                await self.keepalive_listen_key(listen_key)
                                last_keepalive = time.time()
            except Exception as e:
                log.warning("private WS error %s — recreating listenKey", e)
                try:
                    listen_key = await self.create_listen_key()
                except Exception:
                    await asyncio.sleep(5)
            await asyncio.sleep(3)
