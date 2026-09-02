import asyncio
from nepseman_api import NepseClient


class Nepse:
    def __init__(self, *args, **kwargs):
        self.security_map = None

    def _run(self, coro):
        return asyncio.run(coro)

    def get_security_list(self):
        async def fetch():
            async with NepseClient() as client:
                return await client.security_list()

        data = self._run(fetch())

        self.security_map = {}

        if isinstance(data, dict):
            items = data.get("data", data.get("content", []))
        else:
            items = data

        result = []

        for item in items:
            if not isinstance(item, dict):
                continue

            sid = (
                item.get("id")
                or item.get("securityId")
                or item.get("security_id")
            )

            symbol = (
                item.get("symbol")
                or item.get("stockSymbol")
                or item.get("ticker")
            )

            if sid is not None and symbol:
                self.security_map[str(sid)] = symbol
                result.append({
                    "symbol": symbol,
                    "id": sid
                })

        return result

    def get_historical_chart(
        self,
        security_id,
        start_date=None,
        end_date=None,
        **kwargs
    ):
        if self.security_map is None:
            self.get_security_list()

        symbol = self.security_map.get(str(security_id))

        if not symbol:
            return []

        async def fetch():
            async with NepseClient() as client:
                return await client.price_history(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    size=500
                )

        return self._run(fetch())

    def get_nepse_index(self):
        async def fetch():
            async with NepseClient() as client:
                return await client.nepse_index()

        try:
            return self._run(fetch())
        except Exception:
            return {}


        
