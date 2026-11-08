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
from skimage.color import rgb2lab
import lpips
from DISTS_pytorch import DISTS

from cleanfid.fid import frechet_distance, kernel_distance, get_batch_features
from cleanfid.resize import build_resizer
from cleanfid.features import build_feature_extractor
from pytorch_msssim import ms_ssim


# -------------------------------
# Утилиты загрузки
# -------------------------------

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)


def to_tensor(img_np):
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def to_tensor_neg1_1(img_np):
    return to_tensor(img_np) * 2 - 1


# -------------------------------
# Парсинг мета-информации из пути
# -------------------------------

def parse_path_meta(results_dir: str):
    """
    Извлекает model и experiment из пути вида:
      .../benchmarking/<MODEL>/results/<EXPERIMENT>/...
      .../benchmarking/<MODEL>/results/<EXPERIMENT>/images   (подпапка images/good/norm/comb — игнорируется)

    Примеры:
      benchmarking/PSPStain/results/random_25/images  -> model=PSPStain, experiment=random_25, epoch=unknown
      benchmarking/PSPStain/results/MIST/val_pretrained/images -> model=PSPStain, experiment=MIST, epoch=unknown
    """
    parts = os.path.normpath(os.path.abspath(results_dir)).split(os.sep)

    model      = "unknown"
    experiment = "unknown"
    epoch      = "unknown"

    # benchmarking -> model
    try:
        bi = parts.index("benchmarking")
        if bi + 1 < len(parts):
            model = parts[bi + 1]
    except ValueError:
        pass

    # results -> experiment
    try:
        ri = parts.index("results")
        if ri + 1 < len(parts):
            experiment = parts[ri + 1]

        # Ищем test_<epoch> если есть
        for part in parts[ri + 2:]:
            if part.startswith("test_") or part.startswith("val_"):
                epoch = part.split("_", 1)[1]
                break
    except ValueError:
        experiment = parts[-1]

    return model, experiment, epoch


# -------------------------------
# Дополнительные метрики
# -------------------------------

def compute_l_ssim(real_np, fake_np):
    real_lab = rgb2lab(real_np)[..., 0]
    fake_lab = rgb2lab(fake_np)[..., 0]
    return ssim(real_lab, fake_lab, data_range=100)


def compute_ms_ssim_torch(real_np, fake_np):
    r = to_tensor(real_np)
    f = to_tensor(fake_np)
    return ms_ssim(r, f, data_range=1.0).item()


def compute_ncc(real_np, fake_np):
    r = real_np.astype(np.float32).mean(axis=2)
    f = fake_np.astype(np.float32).mean(axis=2)
    r -= r.mean()
    f -= f.mean()
    denom = np.sqrt((r * r).sum()) * np.sqrt((f * f).sum()) + 1e-8
    return (r * f).sum() / denom


def compute_pearson(real_np, fake_np):
    r = real_np.astype(np.float32).mean(axis=2).flatten()
    f = fake_np.astype(np.float32).mean(axis=2).flatten()
    return np.corrcoef(r, f)[0, 1]


# -------------------------------
# SSIM/PSNR + доп. метрики (CPU, параллельно)
# -------------------------------

def compute_full_metrics(real_path, fake_path):
    real_np = load_image(real_path)
    fake_np = load_image(fake_path, size=real_np.shape[:2][::-1])

    return {
        "SSIM":   ssim(real_np, fake_np, channel_axis=2, data_range=255),
        "PSNR":   psnr(real_np, fake_np, data_range=255),
        "LSSIM":  compute_l_ssim(real_np, fake_np),
        "MSSSIM": compute_ms_ssim_torch(real_np, fake_np),
        "NCC":    compute_ncc(real_np, fake_np),
        "PCC":    compute_pearson(real_np, fake_np),
    }


# -------------------------------
# FID / KID
# -------------------------------

def compute_fid_kid_from_paths(fake_paths, real_paths,
                                mode="legacy_pytorch", batch_size=64, device="cuda"):
    fn_resize  = build_resizer(mode)
    feat_model = build_feature_extractor(mode, device)

    def gen_features(paths):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch_imgs = []
            for p in paths[i:i + batch_size]:
                img_np      = load_image(p)
                img_resized = fn_resize(img_np)
                img_t       = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
                batch_imgs.append(img_t)
            batch = torch.cat(batch_imgs, dim=0)
            feats.append(get_batch_features(batch, feat_model, device))
        return np.concatenate(feats, axis=0)

    feats_fake = gen_features(fake_paths)
    feats_real = gen_features(real_paths)

    mu_fake,  sigma_fake = np.mean(feats_fake, axis=0), np.cov(feats_fake, rowvar=False)
    mu_real,  sigma_real = np.mean(feats_real, axis=0), np.cov(feats_real, rowvar=False)

    fid = frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    kid = kernel_distance(feats_fake, feats_real)
    return fid, kid


# -------------------------------
# Dataset для LPIPS / DISTS
# -------------------------------

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, names, real_dict, fake_dict):
        self.names     = names
        self.real_dict = real_dict
        self.fake_dict = fake_dict

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name    = self.names[idx]
        real_np = load_image(self.real_dict[name])
        fake_np = load_image(self.fake_dict[name], size=real_np.shape[:2][::-1])
        r01 = to_tensor(real_np).squeeze(0)
        f01 = to_tensor(fake_np).squeeze(0)
        r11 = to_tensor_neg1_1(real_np).squeeze(0)
        f11 = to_tensor_neg1_1(fake_np).squeeze(0)
        return r01, f01, r11, f11


# -------------------------------
# Фильтрация пар по txt-файлу
# -------------------------------

def filter_by_txt(names, txt_path):
    with open(txt_path, "r") as f:
        allowed = {
            line.strip().rstrip("\r\n")
            for line in f
            if line.strip()
        }
    return [n for n in names if n in allowed]


# -------------------------------
# Метрики для одного subset-а
# -------------------------------

def compute_metrics_for_subset(names, real_dict, fake_dict,
                                batch_size, num_workers, device):
    if not names:
        return None

    ssim_scores, psnr_scores    = [], []
    lssim_scores, msssim_scores = [], []
    ncc_scores, pcc_scores      = [], []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compute_full_metrics, real_dict[n], fake_dict[n])
            for n in names
        ]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  SSIM/PSNR/extras"):
            m = future.result()
            ssim_scores.append(m["SSIM"])
            psnr_scores.append(m["PSNR"])
            lssim_scores.append(m["LSSIM"])
            msssim_scores.append(m["MSSSIM"])
            ncc_scores.append(m["NCC"])
            pcc_scores.append(m["PCC"])

    # LPIPS / DISTS
    dists_metric = DISTS().to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_metric.eval()

    dataset = PairDataset(names, real_dict, fake_dict)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    dists_scores, lpips_scores = [], []
    with torch.no_grad():
        for r01, f01, r11, f11 in tqdm(loader, desc="  LPIPS/DISTS"):
            r01, f01 = r01.to(device), f01.to(device)
            r11, f11 = r11.to(device), f11.to(device)
            dists_scores.extend(dists_metric(r01, f01).cpu().numpy().flatten().tolist())
            lpips_scores.extend(lpips_metric(r11, f11).cpu().numpy().flatten().tolist())

    fid_value, kid_value = compute_fid_kid_from_paths(
        [fake_dict[n] for n in names],
        [real_dict[n] for n in names],
        device=device
    )

    return {
        "PSNR":    np.mean(psnr_scores),
        "SSIM":    np.mean(ssim_scores),
        "FID":     fid_value,
        "KID":     kid_value,
        "LPIPS":   np.mean(lpips_scores),
        "DISTS":   np.mean(dists_scores),
        "L-SSIM":  np.mean(lssim_scores),
        "MS-SSIM": np.mean(msssim_scores),
        "NCC":     np.mean(ncc_scores),
        "PCC":     np.mean(pcc_scores),
    }


# -------------------------------
# Главная функция
# -------------------------------

def compute_metrics(results_dir, output_csv,
                    batch_size=8, num_workers=8,
                    good_txt=None, norm_txt=None, comb_txt=None):

    print(f"\nResults dir : {results_dir}")

    model, experiment, epoch = parse_path_meta(results_dir)
    print(f"Model: {model}  |  Experiment: {experiment}  |  Epoch: {epoch}")

    # Создаём директорию для CSV
    csv_dir = os.path.dirname(output_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # --- Ключевое отличие от BCI: подпапки real_B/ и fake_B/ ---
    real_paths = sorted(glob.glob(os.path.join(results_dir, "real_B", "*.png")))
    fake_paths = sorted(glob.glob(os.path.join(results_dir, "fake_B", "*.png")))

    # Поддержка jpg на случай если встретится
    if not real_paths:
        real_paths = sorted(glob.glob(os.path.join(results_dir, "real_B", "*.jpg")))
    if not fake_paths:
        fake_paths = sorted(glob.glob(os.path.join(results_dir, "fake_B", "*.jpg")))

    real_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in real_paths}
    fake_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in fake_paths}

    all_names = sorted(set(real_dict.keys()) & set(fake_dict.keys()))
    if not all_names:
        print("❌ Не найдено ни одной пары изображений!")
        print(f"   Проверь наличие подпапок real_B/ и fake_B/ в {results_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Total pairs: {len(all_names)}")

    # Subset-ы
    subsets = [("full", all_names)]

    for subset_name, txt_path in [("good", good_txt), ("norm", norm_txt), ("combined", comb_txt)]:
        if txt_path is not None:
            filtered = filter_by_txt(all_names, txt_path)
            print(f"  Subset '{subset_name}': {len(filtered)} pairs (from {txt_path})")
            if filtered:
                subsets.append((subset_name, filtered))
            else:
                print(f"  ⚠ Subset '{subset_name}' пуст — пропускаем")

    # Считаем метрики
    rows = []
    for subset_name, names in subsets:
        print(f"\n{'='*50}")
        print(f"Processing subset: {subset_name}  ({len(names)} pairs)")
        print(f"{'='*50}")

        metrics = compute_metrics_for_subset(
            names, real_dict, fake_dict,
            batch_size, num_workers, device
        )
        if metrics is None:
            continue

        row = {
            "model":      model,
            "experiment": experiment,
            "epoch":      epoch,
            "subset":     subset_name,
            **metrics,
        }
        rows.append(row)
        print(f"✅ Done: {subset_name}")

    # Сохраняем
    if rows:
        df = pd.DataFrame(rows)
        col_order = [
            "model", "experiment", "epoch", "subset",
            "PSNR", "SSIM", "FID", "KID",
            "LPIPS", "DISTS", "L-SSIM", "MS-SSIM", "NCC", "PCC",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved to {output_csv}")
        print(df.to_string(index=False))
    else:
        print("❌ Нет результатов для сохранения")


# -------------------------------
# Entry point
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Папка с подпапками real_B/ и fake_B/. "
                             "Ожидаемый формат пути: "
                             ".../benchmarking/<MODEL>/results/<EXPERIMENT>/...")
    parser.add_argument("--output_csv",  type=str, default=None,
                        help="Путь для сохранения CSV. "
                             "По умолчанию: ~/benchmarking/metrics/<model>/<experiment>/results.csv")
    parser.add_argument("--subsets_dir", type=str,
                        default="~/benchmarking/artemii_selected_images",
                        help="Папка с good_clean.txt, norm_clean.txt, combined_clean.txt")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)

    args = parser.parse_args()

    results_dir = os.path.expanduser(args.results_dir)
    subsets_dir = os.path.expanduser(args.subsets_dir)

    good_txt = os.path.join(subsets_dir, "good_clean.txt")
    norm_txt = os.path.join(subsets_dir, "norm_clean.txt")
    comb_txt = os.path.join(subsets_dir, "combined_clean.txt")

    for label, path in [("good", good_txt), ("norm", norm_txt), ("combined", comb_txt)]:
        if not os.path.isfile(path):
            print(f"⚠ Файл для subset '{label}' не найден: {path} — subset будет пропущен")

    if args.output_csv is not None:
        output_csv = os.path.expanduser(args.output_csv)
    else:
        model, experiment, epoch = parse_path_meta(results_dir)
        output_csv = os.path.expanduser(
            f"~/benchmarking/metrics/{model}/{experiment}/results.csv"
        )

    print(f"Output CSV  : {output_csv}")

    compute_metrics(
        results_dir=results_dir,
        output_csv=output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        good_txt=good_txt if os.path.isfile(good_txt) else None,
        norm_txt=norm_txt if os.path.isfile(norm_txt) else None,
        comb_txt=comb_txt if os.path.isfile(comb_txt) else None,
    )
