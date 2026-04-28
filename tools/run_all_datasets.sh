#!/usr/bin/env bash
# Run fusion tracking on all valid datasets under a data root path.
# Usage: ./tools/run_all_datasets.sh <data_root> [parallel_workers]
#
# A valid dataset must have:
#   <dataset>/CollectionSystem/IE/<dataset>.txt   (IE GT file)
#   <dataset>/CollectionSystem/*/rosbag/          (rosbag directory with *.bag files)
#
# Example:
#   ./tools/run_all_datasets.sh /root/data 5

set -euo pipefail

DATA_ROOT="${1:?Usage: $0 <data_root> [parallel_workers]}"
PARALLEL="${2:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve absolute path
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"

echo "============================================"
echo "Data root:  ${DATA_ROOT}"
echo "Parallel:   ${PARALLEL}"
echo "Project:    ${PROJECT_ROOT}"
echo "============================================"

# Discover valid datasets
DATASETS=()
SKIPPED=()

for dataset_dir in "${DATA_ROOT}"/*/; do
    [ -d "${dataset_dir}" ] || continue
    dataset_name="$(basename "${dataset_dir}")"

    # Check for IE GT file
    ie_file="${dataset_dir}CollectionSystem/IE/${dataset_name}.txt"
    if [ ! -f "${ie_file}" ]; then
        SKIPPED+=("${dataset_name} (no IE GT file)")
        continue
    fi

    # Find rosbag directory
    rosbag_dir="$(find "${dataset_dir}CollectionSystem" -maxdepth 2 -name "rosbag" -type d 2>/dev/null | head -1)"
    if [ -z "${rosbag_dir}" ]; then
        SKIPPED+=("${dataset_name} (no rosbag directory)")
        continue
    fi

    # Check for .bag files
    bag_count="$(ls "${rosbag_dir}"/*.bag 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${bag_count}" -eq 0 ]; then
        SKIPPED+=("${dataset_name} (no .bag files)")
        continue
    fi

    DATASETS+=("${dataset_name}|${rosbag_dir}|${ie_file}|${bag_count}")
done

echo ""
echo "Found ${#DATASETS[@]} valid dataset(s):"
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name rosbag ie bags <<< "${entry}"
    echo "  ${name}: ${bags} bags"
done

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped ${#SKIPPED[@]} dataset(s):"
    for reason in "${SKIPPED[@]}"; do
        echo "  ${reason}"
    done
fi

echo ""
echo "============================================"

# Process each dataset sequentially
TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0
FAILED_NAMES=()

for i in "${!DATASETS[@]}"; do
    IFS='|' read -r dataset_name rosbag_dir ie_file bag_count <<< "${DATASETS[$i]}"
    idx=$((i + 1))
    output_dir="${DATA_ROOT}/${dataset_name}/outputs"

    echo ""
    echo "[${idx}/${TOTAL}] Processing ${dataset_name} (${bag_count} bags)..."
    echo "  rosbag:  ${rosbag_dir}"
    echo "  IE GT:   ${ie_file}"
    echo "  output:  ${output_dir}"
    echo ""

    START_TIME=$(date +%s)

    if python3 "${PROJECT_ROOT}/tools/run_fusion_bag_folder.py" \
        --bag-dir "${rosbag_dir}" \
        --output-dir "${output_dir}" \
        --ie-path "${ie_file}" \
        --parallel "${PARALLEL}" \
        --export-kml \
        --continue-on-error; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "[${idx}/${TOTAL}] ${dataset_name} completed in ${ELAPSED}s"
        SUCCESS=$((SUCCESS + 1))
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "[${idx}/${TOTAL}] ${dataset_name} FAILED after ${ELAPSED}s"
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("${dataset_name}")
    fi
done

echo ""
echo "============================================"
echo "All done: ${SUCCESS} succeeded, ${FAILED} failed out of ${TOTAL}"
if [ ${#FAILED_NAMES[@]} -gt 0 ]; then
    echo "Failed datasets:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  ${name}"
    done
fi
echo "============================================"

[ "${FAILED}" -eq 0 ] || exit 1
