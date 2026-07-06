import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import DTTDEnhanced
from models.baselines import CVAE, Generator, EEGDiff, BrainDiff
from models.classifier import EEGNet as EEGNetClassifier


INPUT_CH = [7, 9, 11, 1, 3, 5, 13, 15, 17]
DATA_SCALE = 1e5
N_PER_CLASS = 150
NUM_CLASSES = 4

METHODS_CONFIG = {
    'DTTD': {
        'checkpoint': 'checkpoints/bci2a_enhanced/best_model.pth',
    },
    'CVAE': {
        'checkpoint': 'checkpoints/bci2a/baseline_cvae/best_model.pth',
    },
    'cGAN': {
        'checkpoint': 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth',
    },
    'EEGDiff': {
        'checkpoint': 'checkpoints/bci2a/baseline_eegdiff/best_model.pth',
    },
    'BrainDiff': {
        'checkpoint': 'checkpoints/bci2a/baseline_braindiff/best_model.pth',
    },
    'Spline': {
        'type': 'interpolation',
    },
    'Kriging': {
        'type': 'interpolation',
    },
}


class CVAE_Large(nn.Module):
    def __init__(self, input_channels, output_channels, time_steps, num_classes, latent_dim=128, pool_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_size)
        )
        self.condition_embed = nn.Embedding(num_classes, 64)
        enc_out_dim = 256 * pool_size
        self.fc_mu = nn.Linear(enc_out_dim + 64, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim + 64, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim + 64, 256 * 64)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, condition):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        c = self.condition_embed(condition)
        h = torch.cat([h, c], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_cond = torch.cat([z, c], dim=1)
        h_dec = self.decoder_fc(z_cond).view(-1, 256, 64)
        out = self.decoder_conv(h_dec)
        if out.size(2) > self.time_steps:
            out = out[:, :, :self.time_steps]
        elif out.size(2) < self.time_steps:
            out = F.pad(out, (0, self.time_steps - out.size(2)))
        return out


def load_real_data():
    data_path = 'results/generated_samples_test.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path)
    generated = data['generated']
    targets = data['targets']
    labels = data['labels']
    print(f"Loaded data: generated={generated.shape}, targets={targets.shape}, labels={labels.shape}")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")
    return generated, targets, labels


def extract_features(clf, data, ch_m, ch_s, batch_size=128):
    feats = []
    for i in range(0, len(data), batch_size):
        batch = ((data[i:i+batch_size] - ch_m) / ch_s).astype(np.float32)
        data_t = torch.FloatTensor(batch).to(device)
        with torch.no_grad():
            x = data_t.unsqueeze(1)
            x = clf.conv1(x)
            x = clf.batchnorm1(x)
            x = clf.depthwise(x)
            x = clf.batchnorm2(x)
            x = clf.activation1(x)
            x = clf.pooling1(x)
            x = clf.dropout1(x)
            x = clf.separable(x)
            x = clf.batchnorm3(x)
            x = clf.activation2(x)
            x = clf.pooling2(x)
            x = clf.dropout2(x)
            x = x.view(x.size(0), -1)
            feats.append(x.cpu().numpy())
    return np.concatenate(feats)


def compute_cross_eegnet(clf, gen_data, gen_labels, ch_m, ch_s, batch_size=128):
    all_pred = []
    for i in range(0, len(gen_data), batch_size):
        batch = ((gen_data[i:i+batch_size] - ch_m) / ch_s).astype(np.float32)
        data_t = torch.FloatTensor(batch).to(device)
        with torch.no_grad():
            pred = clf(data_t).argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
    return (np.concatenate(all_pred) == gen_labels).mean()


def compute_separation_ratio(features, labels):
    unique_labels = np.unique(labels)
    class_centers = []
    for lbl in unique_labels:
        mask = labels == lbl
        class_centers.append(features[mask].mean(axis=0))
    class_centers = np.array(class_centers)

    n_classes = len(unique_labels)
    inter_dists = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            inter_dists.append(np.linalg.norm(class_centers[i] - class_centers[j]))
    inter_class_distance = np.mean(inter_dists)

    intra_dists = []
    for lbl in unique_labels:
        mask = labels == lbl
        cls_feats = features[mask]
        center = cls_feats.mean(axis=0)
        dists = np.linalg.norm(cls_feats - center, axis=1)
        intra_dists.append(dists.mean())
    intra_class_distance = np.mean(intra_dists)

    separation_ratio = inter_class_distance / (intra_class_distance + 1e-10)
    return separation_ratio, inter_class_distance, intra_class_distance


def compute_mmd(x, y, sigma=1.0):
    from scipy.spatial.distance import cdist
    xx = cdist(x, x, 'sqeuclidean')
    yy = cdist(y, y, 'sqeuclidean')
    xy = cdist(x, y, 'sqeuclidean')

    c1 = 1.0 / (2 * sigma ** 2)
    c2 = -1.0 / (2 * sigma ** 2)

    k_xx = np.exp(c2 * xx)
    k_yy = np.exp(c2 * yy)
    k_xy = np.exp(c2 * xy)

    m = x.shape[0]
    n = y.shape[0]

    mmd = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1) + 1e-10) + \
          (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1) + 1e-10) - \
          2 * k_xy.sum() / (m * n + 1e-10)
    return max(mmd, 0.0)


def compute_fid(real_feat, gen_feat):
    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    sigma_r = np.cov(real_feat, rowvar=False)
    sigma_g = np.cov(gen_feat, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = _matrix_sqrt(sigma_r @ sigma_g)

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return max(fid, 0.0)


def _matrix_sqrt(mat):
    from scipy.linalg import sqrtm
    result = sqrtm(mat)
    return result.real, result


def generate_spline(bci_targets, bci_labels, n_per_class):
    from scipy.interpolate import CubicSpline
    n_total_ch = bci_targets.shape[1]
    all_indices = []
    all_lbl = []
    for cls in range(NUM_CLASSES):
        mask = bci_labels == cls
        indices = np.where(mask)[0][:n_per_class]
        all_indices.extend(indices)
        all_lbl.extend([cls] * len(indices))
    sel_data = bci_targets[all_indices]
    n_samples, n_total_ch, n_time = sel_data.shape
    input_data = sel_data[:, INPUT_CH, :]
    sorted_order = np.argsort(INPUT_CH)
    sorted_indices = np.array(INPUT_CH)[sorted_order]
    all_positions = np.arange(n_total_ch, dtype=np.float64)
    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
    for i in range(n_samples):
        for t in range(n_time):
            values = input_data[i, sorted_order, t]
            cs = CubicSpline(sorted_indices.astype(np.float64), values, bc_type='natural')
            result[i, :, t] = cs(all_positions)
    return result, np.array(all_lbl)


def generate_kriging(bci_targets, bci_labels, n_per_class):
    BCI2A_CHANNELS = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    electrode_2d = {
        'Fz': (0.0, 0.3), 'FC3': (-0.3, 0.2), 'FC1': (-0.1, 0.2),
        'FCz': (0.0, 0.2), 'FC2': (0.1, 0.2), 'FC4': (0.3, 0.2),
        'C5': (-0.4, 0.0), 'C3': (-0.3, 0.0), 'C1': (-0.1, 0.0),
        'Cz': (0.0, 0.0), 'C2': (0.1, 0.0), 'C4': (0.3, 0.0),
        'C6': (0.4, 0.0), 'CP3': (-0.3, -0.2), 'CP1': (-0.1, -0.2),
        'CPz': (0.0, -0.2), 'CP2': (0.1, -0.2), 'CP4': (0.3, -0.2),
        'P1': (-0.1, -0.3), 'Pz': (0.0, -0.3), 'P2': (0.1, -0.3),
        'POz': (0.0, -0.4),
    }
    n_total_ch = bci_targets.shape[1]
    all_indices = []
    all_lbl = []
    for cls in range(NUM_CLASSES):
        mask = bci_labels == cls
        indices = np.where(mask)[0][:n_per_class]
        all_indices.extend(indices)
        all_lbl.extend([cls] * len(indices))
    sel_data = bci_targets[all_indices]
    n_samples, n_total_ch, n_time = sel_data.shape
    input_data = sel_data[:, INPUT_CH, :]
    all_coords = np.array([electrode_2d[BCI2A_CHANNELS[i]] for i in range(n_total_ch)])
    input_coords = np.array([electrode_2d[BCI2A_CHANNELS[i]] for i in INPUT_CH])
    n_input = len(INPUT_CH)
    sigma = 0.3
    dists = np.zeros((n_total_ch, n_input))
    for i in range(n_total_ch):
        for j in range(n_input):
            dists[i, j] = np.sum((all_coords[i] - input_coords[j]) ** 2)
    weights = np.exp(-dists / (2 * sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
    for i in range(n_samples):
        for t in range(0, n_time, 50):
            t_end = min(t + 50, n_time)
            vals = input_data[i, :, t:t_end]
            result[i, :, t:t_end] = weights @ vals
    return result, np.array(all_lbl)


def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading BCI2a data...")
    bci_generated, bci_targets, bci_labels = load_real_data()

    print("\nTraining EEGNet classifier on BCI2a...")
    bci_clf = EEGNetClassifier(num_channels=22, num_classes=4, time_steps=1000).to(device)
    ch_mean = bci_targets.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    ch_std = bci_targets.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    normed = ((bci_targets - ch_mean) / ch_std).astype(np.float32)
    opt = torch.optim.Adam(bci_clf.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = nn.CrossEntropyLoss()
    n = len(bci_targets)
    bci_clf.train()
    for ep in range(100):
        perm = np.random.permutation(n)
        for start in range(0, n, 64):
            idx = perm[start:start+64]
            data_t = torch.FloatTensor(normed[idx]).to(device)
            labels_t = torch.LongTensor(bci_labels[idx]).to(device)
            opt.zero_grad()
            loss = criterion(bci_clf(data_t), labels_t)
            loss.backward()
            opt.step()
        scheduler.step()
    bci_clf.eval()
    with torch.no_grad():
        all_pred = []
        for start in range(0, n, 256):
            data_t = torch.FloatTensor(normed[start:start+256]).to(device)
            all_pred.append(bci_clf(data_t).argmax(dim=1).cpu().numpy())
    acc = (np.concatenate(all_pred) == bci_labels).mean()
    print(f"  EEGNet accuracy on real data: {acc:.4f}")

    print("\nExtracting real data features...")
    real_list, real_label_list = [], []
    for cls in range(NUM_CLASSES):
        mask = bci_labels == cls
        indices = np.where(mask)[0]
        if len(indices) > N_PER_CLASS:
            indices = np.random.choice(indices, N_PER_CLASS, replace=False)
        real_feat = extract_features(bci_clf, bci_targets[indices], ch_mean, ch_std)
        real_list.append(real_feat)
        real_label_list.extend([cls] * len(indices))
    real_feat = np.concatenate(real_list)
    real_labels = np.array(real_label_list)

    real_sep, real_inter, real_intra = compute_separation_ratio(real_feat, real_labels)
    print(f"  Real Data - Sep. Ratio: {real_sep:.4f} (inter={real_inter:.4f}, intra={real_intra:.4f})")

    results = {
        'Real': {
            'separation_ratio': float(real_sep),
            'inter_class_distance': float(real_inter),
            'intra_class_distance': float(real_intra),
            'cross_eegnet_acc': float(acc),
            'mmd': None,
            'fid': None,
        }
    }

    gen_raw_dict = {}
    gen_labels_dict = {}

    for method_name, cfg in METHODS_CONFIG.items():
        method_type = cfg.get('type', 'model')
        ckpt_path = cfg.get('checkpoint', None)

        if method_type == 'model' and ckpt_path and not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] {method_name}: checkpoint not found ({ckpt_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {method_name}")
        print(f"{'='*60}")

        try:
            if method_name == 'Spline':
                gen_np, gen_lbl = generate_spline(bci_targets, bci_labels, N_PER_CLASS)
            elif method_name == 'Kriging':
                gen_np, gen_lbl = generate_kriging(bci_targets, bci_labels, N_PER_CLASS)
            elif method_name == 'DTTD':
                gen_list = []
                gen_lbl = []
                for cls in range(NUM_CLASSES):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        gen = bci_generated[idx].copy()
                        gen[INPUT_CH, :] = bci_targets[idx, INPUT_CH, :]
                        gen_list.append(gen)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)
            elif method_name == 'CVAE':
                checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=False)
                model = CVAE(input_channels=9, output_channels=22, time_steps=1000,
                             num_classes=4, latent_dim=128, pool_size=256).to(device)
                model.decoder_fc = nn.Sequential(
                    nn.Linear(128 + 64, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 22000),
                ).to(device)
                model.decoder_conv = None
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()
                gen_list = []
                gen_lbl = []
                for cls in range(NUM_CLASSES):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        label_t = torch.tensor([cls], dtype=torch.long).to(device)
                        with torch.no_grad():
                            gen_output, _, _ = model(batch_input, label_t)
                            gen_np_sample = gen_output.cpu().numpy()[0]
                        gen_np_sample[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np_sample)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)
            elif method_name == 'cGAN':
                checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=False)
                model = Generator(input_channels=9, output_channels=22, time_steps=1000,
                                   num_classes=4, latent_dim=128).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()
                gen_list = []
                gen_lbl = []
                for cls in range(NUM_CLASSES):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        label_t = torch.tensor([cls], dtype=torch.long).to(device)
                        z = torch.randn(1, 128).to(device)
                        with torch.no_grad():
                            gen_output = model(z, label_t, batch_input)
                            gen_np_sample = gen_output.cpu().numpy()[0]
                        gen_np_sample[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np_sample)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)
            elif method_name == 'EEGDiff':
                checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=False)
                model = EEGDiff(input_channels=9, output_channels=22, time_steps=1000,
                                 num_classes=4).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()
                gen_list = []
                gen_lbl = []
                for cls in range(NUM_CLASSES):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            gen_output = model.sample(batch_input, num_steps=1, noise_scale=0.02)
                            gen_np_sample = gen_output.cpu().numpy()[0]
                        gen_np_sample[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np_sample)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)
            elif method_name == 'BrainDiff':
                checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=False)
                model = BrainDiff(input_channels=9, output_channels=22, time_steps=1000,
                                   num_classes=4).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()
                gen_list = []
                gen_lbl = []
                for cls in range(NUM_CLASSES):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            gen_output = model.sample(batch_input, num_steps=1, noise_scale=0.02)
                            gen_np_sample = gen_output.cpu().numpy()[0]
                        gen_np_sample[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np_sample)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            gen_raw_dict[method_name] = gen_np
            gen_labels_dict[method_name] = np.array(gen_lbl)
            print(f"  Generated: {gen_np.shape}")

            print(f"  Extracting features...")
            gen_feat = extract_features(bci_clf, gen_np, ch_mean, ch_std)

            sep_ratio, inter_dist, intra_dist = compute_separation_ratio(gen_feat, gen_lbl)
            print(f"  Sep. Ratio: {sep_ratio:.4f} (inter={inter_dist:.4f}, intra={intra_dist:.4f})")

            cross_acc = compute_cross_eegnet(bci_clf, gen_np, gen_lbl, ch_mean, ch_std)
            print(f"  Cross EEGNet Acc: {cross_acc:.4f}")

            print(f"  Computing MMD...")
            mmd_val = compute_mmd(real_feat, gen_feat)
            print(f"  MMD: {mmd_val:.6f}")

            print(f"  Computing FID...")
            fid_val = compute_fid(real_feat, gen_feat)
            print(f"  FID: {fid_val:.6e}")

            results[method_name] = {
                'separation_ratio': float(sep_ratio),
                'inter_class_distance': float(inter_dist),
                'intra_class_distance': float(intra_dist),
                'cross_eegnet_acc': float(cross_acc),
                'mmd': float(mmd_val),
                'fid': float(fid_val),
            }

        except Exception as e:
            import traceback
            print(f"  [ERROR] {method_name}: {e}")
            traceback.print_exc()

    output_path = 'paper_results/figures/distribution_metrics.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Sep.Ratio':>10} {'MMD':>10} {'FID':>12} {'CrossAcc':>10}")
    print("-" * 60)
    for method, data in results.items():
        sep = f"{data['separation_ratio']:.2f}" if data['separation_ratio'] is not None else '-'
        mmd = f"{data['mmd']:.4f}" if data['mmd'] is not None else '-'
        fid = f"{data['fid']:.2e}" if data['fid'] is not None else '-'
        acc = f"{data['cross_eegnet_acc']:.4f}" if data['cross_eegnet_acc'] is not None else '-'
        print(f"{method:<15} {sep:>10} {mmd:>10} {fid:>12} {acc:>10}")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    main()
