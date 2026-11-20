#!/bin/bash
# download_GTEx_v11_junctions.sh
#
# Downloads the GTEx v11 junction read counts GCT file (all tissues, all samples).
# This is the v11 equivalent of GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz
# and is required for compute_GTEx_v11_allTissue_PSI.py.

set -euo pipefail

GTEX_DIR="/ESL/Data/GTEx"
FILE="GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct.gz"
URL="https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/${FILE}"
DEST="${GTEX_DIR}/${FILE}"

mkdir -p "${GTEX_DIR}"

echo "=== GTEx v11 Junction Counts GCT Download ==="
echo "URL:  ${URL}"
echo "Dest: ${DEST}"
echo ""

if [ -f "${DEST}" ]; then
    echo "File already exists: ${DEST}"
    echo "Size: $(du -sh "${DEST}" | cut -f1)"
else
    echo "Downloading (this file is large — may take a while) ..."
    wget -c -O "${DEST}" "${URL}"
    echo "Download complete."
    echo "Size: $(du -sh "${DEST}" | cut -f1)"
fi

echo ""
echo "Verifying file (reading header) ..."
zcat "${DEST}" | head -3
echo ""
echo "Done."
