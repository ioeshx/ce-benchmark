import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import csv
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

@torch.no_grad()
def get_clip_score(model, processor, image: Image.Image, prompt: str, device: str):
    inputs = processor(
        text=[prompt], 
        images=[image], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    
    img_feat = outputs.image_embeds
    txt_feat = outputs.text_embeds
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    
    return torch.sum(img_feat * txt_feat, dim=-1).item()

def get_prompt_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    prompt = stem.rsplit("_", 1)[0]
    prompt = prompt.replace("_", " ")
    return prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default=None)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--prompt_from_filename", action="store_true")
    args = ap.parse_args()

    if not args.in_csv and not args.images_root:
        raise ValueError("Must provide either --in_csv or --images_root")

    if args.in_csv and not os.path.isfile(args.in_csv):
        raise FileNotFoundError(f"missing: {args.in_csv}")
    
    if not args.in_csv:
        args.prompt_from_filename = True

    out_csv = args.out_csv
    if not out_csv:
        if args.in_csv:
            out_csv = os.path.splitext(args.in_csv)[0] + "_clip.csv"
        else:
            out_csv = os.path.normpath(args.images_root).rstrip(os.sep) + "_clip.csv"

    base_dir = args.images_root if args.images_root else (os.path.dirname(os.path.abspath(args.in_csv)) if args.in_csv else "")

    model = CLIPModel.from_pretrained(args.model_id).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_id)

    scores_sum = 0.0
    scores_count = 0
    scores = []

    with open(out_csv, "w", encoding="utf-8", newline="") as wf:
        if args.in_csv:
            with open(args.in_csv, "r", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                fieldnames = list(reader.fieldnames)
                if "clip_score" not in fieldnames: 
                    fieldnames.append("clip_score")
                if args.prompt_from_filename and "prompt" not in fieldnames:
                    fieldnames.append("prompt")
                    
                writer = csv.DictWriter(wf, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    p = row.get("image_path")
                    
                    if args.prompt_from_filename and p:
                        prompt = get_prompt_from_filename(p)
                        row["prompt"] = prompt
                    else:
                        prompt = row.get("prompt", "")
                    
                    if not p:
                        writer.writerow(row)
                        continue
                        
                    full_path = p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
                    
                    try:
                        img = Image.open(full_path).convert("RGB")
                        score = get_clip_score(model, processor, img, prompt, args.device)
                        scores.append(score)
                        row["clip_score"] = float(score)
                        scores_sum += score
                        scores_count += 1
                    except Exception as e:
                        row["clip_score"] = None
                        
                    writer.writerow(row)
        else:
            fieldnames = ["image_path", "prompt", "clip_score"]
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            for root, _, files in os.walk(args.images_root):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                        p = os.path.join(root, file)
                        prompt = get_prompt_from_filename(p)
                        
                        try:
                            img = Image.open(p).convert("RGB")
                            score = get_clip_score(model, processor, img, prompt, args.device)
                            scores.append(score)
                            row = {"image_path": p, "prompt": prompt, "clip_score": float(score)}
                            scores_sum += score
                            scores_count += 1
                        except Exception as e:
                            row = {"image_path": p, "prompt": prompt, "clip_score": None}
                            
                        writer.writerow(row)

    mean_score = (scores_sum / scores_count) if scores_count > 0 else float('nan')
    std_dev = np.std(scores) if scores else float('nan')
    print(f"CLIP Score - mean: {mean_score:.6f}, std_dev: {std_dev:.6f}, count: {scores_count}")

if __name__ == "__main__":
    main()