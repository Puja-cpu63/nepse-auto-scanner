
import requests

BASE = "https://nepseman-api-production.up.railway.app/api/v1"


class Nepse:
    def __init__(self, *args, **kwargs):
        self.session = requests.Session()
        self.security_map = None

    def get_security_list(self):
        r = self.session.get(
            f"{BASE}/securities/list",
            timeout=30
        )
        r.raise_for_status()
        data = r.json()

        items = data.get("data", data) if isinstance(data, dict) else data

        self.security_map = {}
        result = []

        for x in items:
            if not isinstance(x, dict):
                continue

            symbol = x.get("symbol") or x.get("stockSymbol") or x.get("ticker")
            sid = x.get("id") or x.get("securityId") or x.get("security_id")

            if symbol and sid is not None:
                self.security_map[str(sid)] = symbol
                result.append({
                    "symbol": symbol,
                    "id": sid
                })

        return result

    def get_historical_chart(self, security_id, start_date=None, end_date=None, **kwargs):
        if self.security_map is None:
            self.get_security_list()

        symbol = self.security_map.get(str(security_id))

        if not symbol:
            return []

        params = {"size": 500}

        if start_date:
            params["start_date"] = start_date

        if end_date:
            params["end_date"] = end_date

        r = self.session.get(
            f"{BASE}/securities/{symbol}/history",
            params=params,
            timeout=30
        )
        r.raise_for_status()

        data = r.json()
        return data.get("data", data) if isinstance(data, dict) else data

    def get_nepse_index(self):
        return {}
