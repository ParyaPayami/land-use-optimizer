"""
Download the public context data used by the outcome model.

Every file is written to ``out_dir/<name>/`` and listed in
``out_dir/manifest.json`` with its source URL, query, retrieval time and
SHA-256, so a run can be traced to the exact inputs.

Sources (all public, no key required)
-------------------------------------
facilities      NYC DCP Facilities Database (NYC Open Data ji82-xba5)
food_stores     NYS Agriculture & Markets retail food stores (data.ny.gov 9a8c-vfzj)
subway          MTA subway stations (data.ny.gov 39hk-dx4f)
parks           NYC Parks properties (NYC Open Data enfh-gkve)
streets         NYC Street Centerline, CSCL (NYC Open Data inkn-q76z)
lodes_wac/rac   US Census LEHD LODES 8, jobs by workplace / residence block
lodes_xwalk     LODES 8 geography crosswalk (block internal points)
census_pl       2020 Census P.L. 94-171 redistricting file, New York
clf_wblca       CLF whole-building LCA benchmark dataset v2 (figshare)
ll84            NYC energy and water benchmarking (Local Law 84), report year 2024,
                Manhattan buildings built since 2010 (NYC Open Data 5zyy-y8am)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SOCRATA_NYC = "https://data.cityofnewyork.us/resource"
SOCRATA_NYS = "https://data.ny.gov/resource"
LODES = "https://lehd.ces.census.gov/data/lodes/LODES8/ny"

SOURCES: Dict[str, Dict] = {
    "facilities": {"url": f"{SOCRATA_NYC}/ji82-xba5.csv",
                   "query": {"borocode": "1", "$limit": "100000"}, "file": "facilities.csv",
                   "cite": "NYC Department of City Planning, Facilities Database (NYC Open Data ji82-xba5)"},
    "food_stores": {"url": f"{SOCRATA_NYS}/9a8c-vfzj.csv",
                    "query": {"county": "NEW YORK", "$limit": "100000"}, "file": "food_stores.csv",
                    "cite": "NYS Department of Agriculture and Markets, Retail Food Stores (data.ny.gov 9a8c-vfzj)"},
    "subway": {"url": f"{SOCRATA_NYS}/39hk-dx4f.csv", "query": {"borough": "M", "$limit": "10000"},
               "file": "subway_stations.csv", "cite": "MTA, Subway Stations (data.ny.gov 39hk-dx4f)"},
    "parks": {"url": f"{SOCRATA_NYC}/enfh-gkve.geojson", "query": {"borough": "M", "$limit": "100000"},
              "file": "parks.geojson", "cite": "NYC Department of Parks and Recreation, Parks Properties (enfh-gkve)"},
    "streets": {"url": f"{SOCRATA_NYC}/inkn-q76z.geojson", "query": {"boroughcode": "1", "$limit": "200000"},
                "file": "streets.geojson",
                "cite": "NYC Office of Technology and Innovation, NYC Street Centerline (CSCL) (inkn-q76z)"},
    "lodes_wac": {"url": f"{LODES}/wac/ny_wac_S000_JT00_2023.csv.gz", "file": "ny_wac_S000_JT00_2023.csv.gz",
                  "cite": "U.S. Census Bureau, LEHD Origin-Destination Employment Statistics (LODES 8), 2023"},
    "lodes_rac": {"url": f"{LODES}/rac/ny_rac_S000_JT00_2023.csv.gz", "file": "ny_rac_S000_JT00_2023.csv.gz",
                  "cite": "U.S. Census Bureau, LODES 8 residence area characteristics, 2023"},
    "lodes_xwalk": {"url": f"{LODES}/ny_xwalk.csv.gz", "file": "ny_xwalk.csv.gz",
                    "cite": "U.S. Census Bureau, LODES 8 geography crosswalk"},
    "census_pl": {"url": "https://www2.census.gov/programs-surveys/decennial/2020/data/"
                         "01-Redistricting_File--PL_94-171/New_York/ny2020.pl.zip",
                  "file": "ny2020.pl.zip",
                  "cite": "U.S. Census Bureau, 2020 Census P.L. 94-171 Redistricting Data, New York"},
    "ll84": {"url": f"{SOCRATA_NYC}/5zyy-y8am.csv",
             "query": {"$where": "report_year='2024' AND starts_with(nyc_borough_block_and_lot,'1') "
                                 "AND year_built>='2010'",
                       "$limit": "100000"},
             "file": "ll84_2024_manhattan_built2010plus.csv",
             "cite": "NYC Mayor's Office of Climate and Environmental Justice, Energy and Water Data Disclosure "
                     "for Local Law 84 (2022-present), NYC Open Data 5zyy-y8am"},
    "clf_wblca": {"url": "https://ndownloader.figshare.com/files/52575179", "file": "buildings_metadata.xlsx",
                  "cite": "Carbon Leadership Forum WBLCA Benchmark Study v2 dataset, "
                          "doi:10.6084/m9.figshare.28462145.v2"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 4) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "pimaluos/0.3 (research; open data)"})
    for k in range(retries):
        try:
            tmp = dest.with_name(dest.name + ".part")
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - network errors are retried
            if k == retries - 1:
                raise
            logger.warning("download failed (%s); retrying", e)
            time.sleep(2 ** (k + 1))


def fetch_all(out_dir: str | Path, names: Optional[list] = None, force: bool = False) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name, src in SOURCES.items():
        if names and name not in names:
            continue
        d = out / name
        d.mkdir(exist_ok=True)
        dest = d / src["file"]
        url = src["url"] + ("?" + urllib.parse.urlencode(src["query"]) if src.get("query") else "")
        if dest.exists() and not force and name in manifest:
            logger.info("%s: present", name)
            continue
        logger.info("%s: downloading %s", name, url)
        _download(url, dest)
        manifest[name] = {"url": url, "file": str(dest.relative_to(out)), "sha256": _sha256(dest),
                          "bytes": dest.stat().st_size, "retrieved_utc": dt.datetime.utcnow().isoformat() + "Z",
                          "cite": src["cite"]}
        man_path.write_text(json.dumps(manifest, indent=1))
    return manifest
