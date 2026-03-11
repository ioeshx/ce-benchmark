#!/usr/bin/env bash
set -euo pipefail

# Base paths: each immediate child directory will be treated as one group.
real_base="/home/shx/code/SPEED/data/pretrain/instance"
gen_base="/home/shx/code/ce/result/robust_PCA_Tom_and_Jerry/Snoopy_Mickey_Spongebob"

# Optional settings.
device="cuda"
out_root="results/fid"

mkdir -p "${out_root}"

echo "[Info] real_base=${real_base}"
echo "[Info] gen_base=${gen_base}"
echo "[Info] out_root=${out_root}"

if [[ ! -d "${real_base}" ]]; then
    echo "[Error] real_base not found: ${real_base}" >&2
    exit 1
fi

if [[ ! -d "${gen_base}" ]]; then
    echo "[Error] gen_base not found: ${gen_base}" >&2
    exit 1
fi

matched=0

for real_dir in "${real_base}"/*; do
    [[ -d "${real_dir}" ]] || continue

    name="$(basename "${real_dir}")"
    gen_dir="${gen_base}/${name}"

    if [[ ! -d "${gen_dir}" ]]; then
        echo "[Skip] missing generated folder for ${name}: ${gen_dir}"
        continue
    fi

    out_json="${out_root}/${name}-fid.json"
    echo "[Run] ${name}"
    echo "  real_dir: ${real_dir}"
    echo "  gen_dir: ${gen_dir}"

    python metric/eval_fid.py \
        --real_dir "${real_dir}/original" \
        --gen_dir "${gen_dir}/edit" \
        --device "${device}" \
        --out_json "${out_json}"

    matched=$((matched + 1))
done

echo "[Done] matched groups: ${matched}"
