"""Modelos de redes neurais (fora do escopo final do projeto, que é só visão clássica).

Mantidos como experimento separado. Mesmo protocolo dos modelos clássicos.

Dois detectores One-Class, ambos ajustados SOMENTE nas 256 boas de
desenvolvimento (`fit_ids` do manifesto externo) e com limiar calibrado nas 64
boas reservadas (`calibration_ids`, FPR-alvo 5%), exatamente como os modelos
clássicos. O teste não é lido aqui.

- `patchcore`: características de patches de uma Wide ResNet-50 pré-treinada
  no ImageNet (camadas 2 e 3), banco de memória reduzido por coreset guloso e
  score = maior distância de um patch ao vizinho mais próximo do banco
  (Roth et al., 2022, "Towards Total Recall in Industrial Anomaly Detection").
- `autoencoder`: autoencoder convolucional treinado do zero; score = média dos
  1% maiores erros quadráticos de reconstrução (mapa suavizado).

Hiperparâmetros fixados a priori (padrões da literatura); não há busca, pois
não há defeitos de desenvolvimento independentes do teste.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anomaly import CalibratedAnomalyMixin
from src.feature_experiments import file_hash, load_study, reserve_directory, save_json

TARGET_FPR = 0.05
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def carregar_rgb(path, size):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Imagem ilegível: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


class NeuralCalibrated(CalibratedAnomalyMixin):
    """Mesma regra do mixin: score <= limiar -> +1 (normal); acima -> -1 (defeito)."""

    def decision_function(self, paths):
        if not hasattr(self, "threshold_"):
            raise ValueError("Modelo ainda não calibrado.")
        return self.threshold_ - self.anomaly_score(paths)


# ----------------------------------------------------------------------------
# PatchCore
# ----------------------------------------------------------------------------

class PatchCoreDetector(NeuralCalibrated):
    def __init__(self, resize=256, crop=224, coreset_ratio=0.1, projection_dim=128,
                 feature_dim=1024, seed=42, target_fpr=TARGET_FPR):
        self.resize, self.crop = resize, crop
        self.coreset_ratio, self.projection_dim = coreset_ratio, projection_dim
        self.feature_dim, self.seed, self.target_fpr = feature_dim, seed, target_fpr

    def _backbone(self):
        if getattr(self, "_net", None) is None:
            import torchvision
            net = torchvision.models.wide_resnet50_2(
                weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self._net = net.eval()
        return self._net

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_net", None)  # pesos ImageNet recarregados do torchvision
        return state

    def _tensor(self, paths):
        off = (self.resize - self.crop) // 2
        batch = []
        for path in paths:
            image = carregar_rgb(path, self.resize)[off:off + self.crop, off:off + self.crop]
            batch.append(((image.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch))

    @torch.no_grad()
    def patch_features(self, paths, batch_size=16):
        """Retorna tensor (n_imagens, n_patches, feature_dim)."""
        net, out = self._backbone(), []
        for i in range(0, len(paths), batch_size):
            x = self._tensor(paths[i:i + batch_size])
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            l2 = net.layer2(net.layer1(x))
            l3 = net.layer3(l2)
            # Agregação de vizinhança local 3x3 e alinhamento de resolução.
            l2 = F.avg_pool2d(l2, 3, 1, 1)
            l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear",
                               align_corners=False)
            feats = torch.cat([l2, l3], 1)  # (b, 1536, h, w)
            b, c, h, w = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
            feats = F.adaptive_avg_pool1d(feats, self.feature_dim)
            out.append(feats)
            self.patch_grid_ = (h, w)
        return torch.cat(out)

    def _coreset(self, bank):
        n_select = max(1, int(len(bank) * self.coreset_ratio))
        generator = torch.Generator().manual_seed(self.seed)
        projection = torch.randn(bank.shape[1], self.projection_dim, generator=generator) / np.sqrt(self.projection_dim)
        reduced = bank @ projection
        selected = [int(torch.randint(len(bank), (1,), generator=generator))]
        min_dist = torch.cdist(reduced, reduced[selected]).squeeze(1)
        for _ in range(n_select - 1):
            idx = int(torch.argmax(min_dist))
            selected.append(idx)
            min_dist = torch.minimum(min_dist, torch.cdist(reduced, reduced[idx:idx + 1]).squeeze(1))
        return bank[selected]

    def fit(self, paths, sample_ids=None):
        self._reset_calibration()
        start = time.perf_counter()
        feats = self.patch_features(paths)
        bank = feats.reshape(-1, feats.shape[-1])
        self.n_patches_total_ = len(bank)
        self.memory_bank_ = self._coreset(bank)
        self.fit_ids_ = tuple(sample_ids) if sample_ids is not None else None
        self.fit_seconds_ = time.perf_counter() - start
        return self

    @torch.no_grad()
    def anomaly_maps(self, paths):
        feats = self.patch_features(paths)
        maps = []
        for f in feats:
            maps.append(torch.cdist(f, self.memory_bank_).min(1).values.reshape(self.patch_grid_))
        return torch.stack(maps).numpy()

    def anomaly_score(self, paths):
        return self.anomaly_maps(paths).reshape(len(paths), -1).max(1).astype(np.float64)


# ----------------------------------------------------------------------------
# Autoencoder convolucional
# ----------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.encoder = nn.Sequential(down(1, 32), down(32, 64), down(64, 128), down(128, 256),
                                     nn.Conv2d(256, latent, 3, 1, 1))  # 128 -> 8x8xlatent
        self.decoder = nn.Sequential(nn.Conv2d(latent, 256, 3, 1, 1), nn.ReLU(True),
                                     up(256, 128), up(128, 64), up(64, 32),
                                     nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(NeuralCalibrated):
    def __init__(self, size=128, epochs=200, batch_size=16, lr=1e-3, latent=64, top_fraction=0.01,
                 seed=42, target_fpr=TARGET_FPR):
        self.size, self.epochs, self.batch_size, self.lr = size, epochs, batch_size, lr
        self.latent, self.top_fraction, self.seed, self.target_fpr = latent, top_fraction, seed, target_fpr

    def _load(self, paths):
        imgs = []
        for path in paths:
            gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Imagem ilegível: {path}")
            imgs.append(cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA))
        return torch.from_numpy(np.stack(imgs).astype(np.float32)[:, None] / 255)

    def _augment(self, x, generator):
        # Parafusos aparecem em orientações arbitrárias: rotação aleatória e espelhamento.
        angles = (torch.rand(len(x), generator=generator) * 2 - 1) * np.pi
        cos, sin = torch.cos(angles), torch.sin(angles)
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                             torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
        flip = torch.rand(len(x), generator=generator) < 0.5
        x[flip] = x[flip].flip(-1)
        return x

    def fit(self, paths, sample_ids=None):
        self._reset_calibration()
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        data = self._load(paths)
        model = ConvAutoencoder(self.latent)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        self.loss_history_ = []
        start = time.perf_counter()
        for epoch in range(self.epochs):
            model.train()
            order = torch.randperm(len(data), generator=generator)
            total = 0.0
            for i in range(0, len(data), self.batch_size):
                batch = self._augment(data[order[i:i + self.batch_size]].clone(), generator)
                loss = F.mse_loss(model(batch), batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            scheduler.step()
            self.loss_history_.append(total / len(data))
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                print(f"  autoencoder época {epoch + 1}/{self.epochs}: MSE {self.loss_history_[-1]:.5f}", flush=True)
        self.model_ = model.eval()
        self.fit_ids_ = tuple(sample_ids) if sample_ids is not None else None
        self.fit_seconds_ = time.perf_counter() - start
        return self

    @torch.no_grad()
    def anomaly_maps(self, paths):
        x = self._load(paths)
        error = (self.model_(x) - x) ** 2
        return F.avg_pool2d(error, 5, 1, 2).squeeze(1).numpy()

    def anomaly_score(self, paths):
        maps = self.anomaly_maps(paths).reshape(len(paths), -1)
        k = max(1, int(maps.shape[1] * self.top_fraction))
        return np.sort(maps, 1)[:, -k:].mean(1).astype(np.float64)


# ----------------------------------------------------------------------------
# Ajuste + calibração final
# ----------------------------------------------------------------------------

def make_neural(name):
    if name == "patchcore":
        return PatchCoreDetector()
    if name == "autoencoder":
        return AutoencoderDetector()
    raise ValueError("Modelo neural desconhecido.")


def finalize(study, name, out, *, confirm=False):
    if not confirm:
        raise ValueError("Calibração final exige --confirm-final-calibration.")
    if Path(out).exists():
        raise FileExistsError(f"Destino existente: {out}")
    inventory, outer, _ = load_study(study)
    hashes = inventory.set_index("caminho")["sha256"]
    for path in outer["fit_ids"] + outer["calibration_ids"]:
        if file_hash(path) != hashes[path]:
            raise ValueError(f"Imagem alterada: {path}")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dest = reserve_directory(out)
    detector = make_neural(name)
    print(f"{name}: ajuste em {len(outer['fit_ids'])} boas de desenvolvimento", flush=True)
    detector.fit(outer["fit_ids"], sample_ids=outer["fit_ids"])
    print(f"{name}: calibração do limiar em {len(outer['calibration_ids'])} boas reservadas", flush=True)
    detector.calibrate(outer["calibration_ids"], sample_ids=outer["calibration_ids"])
    torch.save(detector, dest / "model.pt")
    params = {k: v for k, v in vars(detector).items() if isinstance(v, (int, float, str)) and not k.endswith("_")}
    save_json(dest / "final_calibration.json", {
        "status": "complete", "model": name, "parameters": params,
        "outer_manifest_hash": outer["manifest_hash"], "n_fit": len(outer["fit_ids"]),
        "n_calibration": len(outer["calibration_ids"]), "fit_ids": outer["fit_ids"],
        "calibration_ids": outer["calibration_ids"], "target_fpr": TARGET_FPR,
        "threshold": detector.threshold_, "calibration_fpr": detector.calibration_fpr_,
        "calibration_scores": detector.calibration_scores_.tolist(),
        "fit_seconds": detector.fit_seconds_,
        "loss_history": getattr(detector, "loss_history_", None),
        "memory_bank_size": (len(detector.memory_bank_) if name == "patchcore" else None),
        "torch_version": torch.__version__, "source_sha256": file_hash(__file__),
        "model_sha256": file_hash(dest / "model.pt"), "test_accessed": False})
    print(f"{name}: limiar {detector.threshold_:.6g}; salvo em {dest}", flush=True)
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("patchcore", "autoencoder"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--confirm-final-calibration", action="store_true")
    args = parser.parse_args(argv)
    try:
        finalize(args.study, args.model, args.out, confirm=args.confirm_final_calibration)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Etapa 7 interrompida: {exc}\n")


if __name__ == "__main__":
    # Executa pela importação do pacote para que o pickle registre src.neural.<Classe>.
    from neural_networks.neural import main as _main
    _main()
