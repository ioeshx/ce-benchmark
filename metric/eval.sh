image_root="/home/shx/code/ce/result/orth_max_valid/Snoopy_to_dog_proj_t2a_proj_only"
# image_root="/home/shx/code/ce/result/orth_max_valid/Snoopy_to_dog_proj_t2a_proj_only"

echo image_root: ${image_root}

python metric/eval_fid.py \
    --real_dir "${image_root}/target_original" \
    --gen_dir "${image_root}/projected" \
    --device "cuda" \
    --out_json "result/orth_max_valid/Snoopy_to_dog_proj_t2a_proj_only/fid_score.json"

python metric/eval_clip_score.py \
    --images-root "${image_root}/projected" \
    --prompt_from_filename \
    --out_csv "result/orth_max_valid/Snoopy_to_dog_proj_t2a_proj_only/clip_scores.csv"

python metric/eval_lpips.py \
    --original_dir "${image_root}/target_original" \
    --edited_dir "${image_root}/projected" \
    --device "cuda" \
    --out_json "result/orth_max_valid/Snoopy_to_dog_proj_t2a_proj_only/lpips_scores.json"