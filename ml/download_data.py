from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://health.data.ny.gov/resource/gnzp-ekau.json"


def download_records(output_path: Path, api_key: str | None, limit: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    offset = 0
    page_size = 50_000

    while len(records) < limit:
        params: dict[str, object] = {
            "$limit": min(page_size, limit - len(records)),
            "$offset": offset,
            "$where": "health_service_area='New York City'",
        }
        if api_key:
            params["$$app_token"] = api_key

        response = requests.get(API_URL, params=params, timeout=120)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        records.extend(page)
        offset += len(page)
        if len(page) < page_size:
            break

    if not records:
        raise RuntimeError("The data API returned no New York City records.")
    pd.DataFrame(records[:limit]).to_csv(output_path, index=False)
    print(f"Wrote {min(len(records), limit)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LOS source records.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    download_records(args.output, os.getenv("NY_DATA_API_KEY"), args.limit)


if __name__ == "__main__":
    main()
