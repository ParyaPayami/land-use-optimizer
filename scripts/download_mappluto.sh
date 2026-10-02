#!/usr/bin/env bash
# Download the MapPLUTO release used in the paper and verify its checksum.
set -euo pipefail
mkdir -p data/raw
URL=https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/mappluto/nyc_mappluto_26v2_arc_shp.zip
OUT=data/raw/nyc_mappluto_26v2_arc_shp.zip
[ -f "$OUT" ] || curl -fL --retry 3 -o "$OUT" "$URL"
echo "38b518ff2f6ccc7e2824cac30cadaf15edc9d35933eb5eae0e10dab432774ecc  $OUT" | sha256sum -c -
