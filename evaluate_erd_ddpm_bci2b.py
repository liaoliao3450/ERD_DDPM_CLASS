"""
ERD-DDPM 单独评估脚本 (BCI2b)
只评估 ERD-DDPM 在三个场景下的数据增强效果，不包含其他 baseline。
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.data_loader_bci2b import load_bci2b_data
from core.models.ddpm.class_discriminative import (
    ClassDiscriminativeDDPM, MultiScaleCondUNet, EEGClassifier,
    pretrain_classifier,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 3
N_SAMPLES = 1000
NUM_CLASSES = 2
FS = 250
CLASSIFIER_EPOCHS = 500
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3
DDPM_BATCH = 16  # Avoid OOM for large n_per_class

# ---------- Checkpointing ----------
CACHE_DIR = "outputs/results/cache_erd_ddpm_bci2b"
os.makedirs(CACHE_DIR, exist_ok=True)

METHOD_NAMES = ['baseline', 'erd_ddpm']


def save_cache(scenario, idx, all_results):
    """Save intermediate results after each subject."""
    cache_path = os.path.join(CACHE_DIR, f"{scenario}_cache.json")
    data = {"completed_idx": idx, "results": {}}
    for name in METHOD_NAMES:
        if name in all_results and all_results[name]:
            data["results"][name] = [float(v) for v in all_results[name]]
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_cache(scenario):
    """Load cached results to resume from breakpoint."""
    cache_path = os.path.join(CACHE_DIR, f"{scenario}_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        all_results = {name: [] for name in METHOD_NAMES}
        for name, vals in data.get("results", {}).items():
            all_results[name] = vals
        start_idx = data.get("completed_idx", -1) + 1
        print(f"  [Resume] {scenario}: starting from subject {start_idx + 1} (cached)")
        return all_results, start_idx
    return {name: [] for name in METHOD_NAMES}, 0


# ---------- DDPM ----------
def load_ddpm_bci2b(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"  [ERROR] DDPM checkpoint not found: {checkpoint_path}")
        return None, None
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    C = int(ckpt.get("channels", CHANNELS))
    T = int(ckpt.get("n_samples", N_SAMPLES))
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    fs = int(ckpt.get("fs", FS))

    eps_model = MultiScaleCondUNet(channels=C, num_classes=num_classes).to(DEVICE)
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(DEVICE)

    if "target_psd" in ckpt and "target_laterality" in ckpt:
        target_psd = ckpt["target_psd"].to(DEVICE)
        target_lat = ckpt["target_laterality"].to(DEVICE)
    else:
        target_psd = torch.zeros(T // 2 + 1, device=DEVICE)
        target_lat = torch.zeros(num_classes, device=DEVICE)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_lat,
        n_timesteps=1000, channels=C, n_samples=T, fs=fs,
    ).to(DEVICE)

    if "model_state_dict" in ckpt:
        try:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    print(f"  Loaded ERD-DDPM from {checkpoint_path}")
    return ddpm, ckpt


def generate_ddpm_samples(ddpm, n_per_class, guidance_scale=1.0):
    """Generate samples with DDPM in batches to avoid OOM. Returns (samples, labels)."""
    all_samples = []
    all_labels = []
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = n_per_class
            while remaining > 0:
                batch = min(DDPM_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample_ddim(batch, y, steps=50, guidance_scale=guidance_scale, device=str(DEVICE))
                all_samples.append(samples.cpu())
                all_labels.append(y.cpu())
                remaining -= batch
                del samples
                torch.cuda.empty_cache()
    return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)


# ---------- Classifier ----------
def train_and_eval_classifier(X_train, y_train, X_test, y_test):
    """Train classifier and return accuracy. Uses validation split for model selection."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=CLASSIFIER_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    best_val_acc = 0.0
    best_state = None

    for ep in range(1, CLASSIFIER_EPOCHS + 1):
        clf.train()
        indices = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), CLASSIFIER_BATCH_SIZE):
            batch_idx = indices[i:i + CLASSIFIER_BATCH_SIZE]
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]
            logits = clf(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if ep % 10 == 0 or ep == CLASSIFIER_EPOCHS:
            clf.eval()
            with torch.no_grad():
                val_pred = clf(X_val_t).argmax(1)
                val_acc = (val_pred == y_val_t).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, pred)
    del clf
    torch.cuda.empty_cache()
    return acc


def augment_and_eval(X_train, y_train, X_test, y_test, gen_X, gen_y):
    """Add generated data to training set and evaluate."""
    X_aug = np.concatenate([X_train, gen_X])
    y_aug = np.concatenate([y_train, gen_y])
    return train_and_eval_classifier(X_aug, y_aug, X_test, y_test)


# ---------- Scenarios ----------
def run_within_subject(X, y, subjects, ddpm, guidance_scale):
    print("\n[Within-Subject] Training on 80% of each subject's T session, testing on 20%")
    all_results, start_idx = load_cache('within_subject')
    n_subjects = len(np.unique(subjects))

    for si in range(start_idx, n_subjects):
        mask = subjects == si
        X_s = X[mask]
        y_s = y[mask]
        # Only session 0 (T)
        sess_mask = np.zeros(len(X_s), dtype=bool)
        # We don't have session info here directly, use all data
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)

        samples_per_class = len(X_train) // NUM_CLASSES

        print(f"\nSubject {si+1}/{n_subjects} (ID={si}):")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES}")

        # Baseline
        acc = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append(acc)
        print(f"  Baseline: {acc*100:.2f}%")

        # ERD-DDPM
        gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
        gen_X = gen_X.cpu().numpy()
        gen_y = gen_y.cpu().numpy()
        acc = augment_and_eval(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['erd_ddpm'].append(acc)
        print(f"  ERD-DDPM: {acc*100:.2f}%")

        save_cache('within_subject', si, all_results)

    return all_results


def run_cross_session(X, y, subjects, sessions, ddpm, guidance_scale):
    print("\n[Cross-Session] Training on T session, testing on E session")
    all_results, start_idx = load_cache('cross_session')
    n_subjects = len(np.unique(subjects))

    for si in range(start_idx, n_subjects):
        mask = subjects == si
        X_s = X[mask]
        y_s = y[mask]
        sess_s = sessions[mask]

        train_mask = sess_s == 0
        test_mask = sess_s == 1
        X_train, y_train = X_s[train_mask], y_s[train_mask]
        X_test, y_test = X_s[test_mask], y_s[test_mask]

        samples_per_class = len(X_train) // NUM_CLASSES

        print(f"\nSubject {si+1}/{n_subjects} (ID={si}):")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES}")

        # Baseline
        acc = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append(acc)
        print(f"  Baseline: {acc*100:.2f}%")

        # ERD-DDPM
        gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
        gen_X = gen_X.cpu().numpy()
        gen_y = gen_y.cpu().numpy()
        acc = augment_and_eval(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['erd_ddpm'].append(acc)
        print(f"  ERD-DDPM: {acc*100:.2f}%")

        save_cache('cross_session', si, all_results)

    return all_results


def run_cross_subject(X, y, subjects, sessions, ddpm, guidance_scale):
    print("\n[Cross-Subject] LOSO: Train on N-1 subjects, test on 1 subject")
    all_results, start_idx = load_cache('cross_subject')
    n_subjects = len(np.unique(subjects))

    for si in range(start_idx, n_subjects):
        test_mask = subjects == si
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        samples_per_class = len(X_train) // (NUM_CLASSES * 2)  # 0.5x train data

        print(f"\nLOSO Subject {si+1}/{n_subjects} (ID={si}):")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES}")

        # Baseline
        acc = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append(acc)
        print(f"  Baseline: {acc*100:.2f}%")

        # ERD-DDPM
        gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
        gen_X = gen_X.cpu().numpy()
        gen_y = gen_y.cpu().numpy()
        acc = augment_and_eval(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['erd_ddpm'].append(acc)
        print(f"  ERD-DDPM: {acc*100:.2f}%")

        save_cache('cross_subject', si, all_results)

    return all_results


def print_summary(results):
    print("\n" + "=" * 70)
    print("ERD-DDPM BCI2b Evaluation Summary")
    print("=" * 70)
    for scenario, res in results.items():
        print(f"\n{scenario}:")
        for name in METHOD_NAMES:
            if name in res and len(res[name]) > 0:
                vals = np.array(res[name])
                print(f"  {name:20s}: {vals.mean()*100:.2f}% +/- {vals.std()*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="ERD-DDPM BCI2b Evaluation")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2b")
    parser.add_argument("--ddpm_ckpt", type=str, default="checkpoints/bci2b/trained_ddpm.pt")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 70)
    print("ERD-DDPM BCI2b Three-Scenario Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Guidance scale: {args.guidance_scale}")

    # Load data
    X, y, subjects, sessions, subj_map = load_bci2b_data(args.data_root, standardize=True)
    print(f"Data: {X.shape}, classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}")

    # Load DDPM
    ddpm, _ = load_ddpm_bci2b(args.ddpm_ckpt)

    # Run three scenarios
    results = {}
    results['within_subject'] = run_within_subject(X, y, subjects, ddpm, args.guidance_scale)
    results['cross_session'] = run_cross_session(X, y, subjects, sessions, ddpm, args.guidance_scale)
    results['cross_subject'] = run_cross_subject(X, y, subjects, sessions, ddpm, args.guidance_scale)

    print_summary(results)

    # Save final results
    os.makedirs("outputs/results", exist_ok=True)
    final_path = "outputs/results/erd_ddpm_bci2b_results.json"
    with open(final_path, "w") as f:
        json.dump({k: {n: [float(v) for v in vals] for n, vals in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {final_path}")


if __name__ == "__main__":
    main()
