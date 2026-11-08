import os
import glob
import torch
import numpy as np
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
from DISTS_pytorch import DISTS

from cleanfid.fid import frechet_distance, kernel_distance, get_batch_features
from cleanfid.resize import build_resizer
from cleanfid.features import build_feature_extractor


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)

def to_tensor(img_np):
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t

def to_tensor_neg1_1(img_np):
    t = to_tensor(img_np) * 2 - 1
    return t

def compute_fid_kid_from_paths(fake_paths, real_paths, mode="legacy_pytorch", batch_size=64, device="cuda"):
    fn_resize = build_resizer(mode)
    feat_model = build_feature_extractor(mode, device)

    def gen_features(paths):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch_imgs = []
            for p in paths[i:i + batch_size]:
                img_np = load_image(p)
                img_resized = fn_resize(img_np)
                img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
                batch_imgs.append(img_t)
            batch = torch.cat(batch_imgs, dim=0)
            feats.append(get_batch_features(batch, feat_model, device))
        return np.concatenate(feats, axis=0)

    feats_fake = gen_features(fake_paths)
    feats_real = gen_features(real_paths)

    mu_fake, sigma_fake = np.mean(feats_fake, axis=0), np.cov(feats_fake, rowvar=False)
    mu_real, sigma_real = np.mean(feats_real, axis=0), np.cov(feats_real, rowvar=False)

    fid = frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    kid = kernel_distance(feats_fake, feats_real)
    return fid, kid


def compute_ssim_psnr(real_path, fake_path):
    real_np = load_image(real_path)
    fake_np = load_image(fake_path, size=real_np.shape[:2][::-1])
    s = ssim(real_np, fake_np, channel_axis=2, data_range=255)
    p = psnr(real_np, fake_np, data_range=255)
    return s, p


def compute_metrics(model_dir, batch_size=8, num_workers=8):
    print(f"\nModel: {model_dir}")

    real_paths = sorted(glob.glob(os.path.join(model_dir, 'real_B', "*.png")))
    fake_paths = sorted(glob.glob(os.path.join(model_dir, 'fake_B', "*.png")))
    
    real_dict = {os.path.basename(p).replace(".png", ""): p for p in real_paths}
    fake_dict = {os.path.basename(p).replace(".png", ""): p for p in fake_paths}

    names = sorted(set(real_dict.keys()) & set(fake_dict.keys()))
    if not names:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssim_scores, psnr_scores = [], []
    real_paths_list = [real_dict[name] for name in names]
    fake_paths_list = [fake_dict[name] for name in names]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_ssim_psnr, r, f) for r, f in zip(real_paths_list, fake_paths_list)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SSIM/PSNR"):
            s, p = future.result()
            ssim_scores.append(s)
            psnr_scores.append(p)

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, names, real_dict, fake_dict):
            self.names = names
            self.real_dict = real_dict
            self.fake_dict = fake_dict
        def __len__(self):
            return len(self.names)
        def __getitem__(self, idx):
            name = self.names[idx]
            real_np = load_image(self.real_dict[name])
            fake_np = load_image(self.fake_dict[name], size=real_np.shape[:2][::-1])
            r01 = to_tensor(real_np).squeeze(0)
            f01 = to_tensor(fake_np).squeeze(0)
            r11 = to_tensor_neg1_1(real_np).squeeze(0)
            f11 = to_tensor_neg1_1(fake_np).squeeze(0)
            return r01, f01, r11, f11

    dists_metric = DISTS().to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_metric.eval()

    dataset = PairDataset(names, real_dict, fake_dict)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    dists_scores, lpips_scores = [], []
    with torch.no_grad():
        for r01, f01, r11, f11 in tqdm(loader, desc="Processing LPIPS/DISTS"):
            r01, f01 = r01.to(device), f01.to(device)
            r11, f11 = r11.to(device), f11.to(device)

            d = dists_metric(r01, f01)
            l = lpips_metric(r11, f11)

            # если батч=1, d/l могут быть float
            if isinstance(d, torch.Tensor):
                dists_scores.extend(d.cpu().numpy().tolist()) if d.dim() > 0 else dists_scores.append(d.item())
            else:
                dists_scores.append(float(d))

            if isinstance(l, torch.Tensor):
                lpips_scores.extend(l.cpu().numpy().tolist()) if l.dim() > 0 else lpips_scores.append(l.item())
            else:
                lpips_scores.append(float(l))

            del r01, f01, r11, f11, d, l
            if device == "cuda":
                torch.cuda.empty_cache()

    fid_value, kid_value = compute_fid_kid_from_paths(
        [fake_dict[name] for name in names],
        [real_dict[name] for name in names],
        device=device
    )

    return {
        "model_dir": model_dir,
        "FID": fid_value,
        "KID": kid_value,
        "SSIM": np.mean(ssim_scores),
        "PSNR": np.mean(psnr_scores),
        "LPIPS": np.mean(lpips_scores),
        "DISTS": np.mean(dists_scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output_csv", type=str, help="Path to save metrics CSV")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for LPIPS/DISTS")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2, help="Number of workers for CPU/GPU tasks")
    args = parser.parse_args()

    metrics = compute_metrics(args.results_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    if metrics is not None:
        df = pd.DataFrame([metrics])
        df.to_csv(args.output_csv, index=False)
        print("Saved to", args.output_csv)
