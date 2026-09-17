"""Download the Census cartographic boundary shapefile used by the map visualisation.

No API key required. Run once; the extracted files are cached in data/geo/.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
DEST = Path(__file__).resolve().parent / "geo"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "cb_2023_us_state_20m.shp").exists():
        print(f"Boundaries already present in {DEST}")
        return

    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DEST)
    print(f"  extracted to {DEST}")


if __name__ == "__main__":
    main()
