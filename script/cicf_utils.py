import datetime
import math
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _infer_pin_memory(device):
    return torch.device(device).type == "cuda"


def _restore_model_mode(model, was_training):
    if was_training:
        model.train()
    else:
        model.eval()


def build_initial_clusters(
    model,
    dataset,
    num_clusters_per_class=3,
    batch_size=256,
    device="cpu",
    num_workers=0,
):
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for CICF clustering. "
            "Please install it before running training."
        ) from exc

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_infer_pin_memory(device),
    )

    was_training = model.training
    model.eval()

    all_features = []
    all_labels = []

    print(f"[{datetime.datetime.now().isoformat()}] Extracting features from {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model.extract_features(images)
            all_features.append(features.cpu())
            all_labels.append(labels.clone())
            if (batch_idx + 1) % 5 == 0:
                print(f"[{datetime.datetime.now().isoformat()}] Extracted {(batch_idx + 1) * batch_size} / {len(dataset)} features")

    _restore_model_mode(model, was_training)

    print(f"[{datetime.datetime.now().isoformat()}] Concatenating features...")
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    features_np = features_tensor.numpy()
    labels_np = labels_tensor.numpy()

    print(f"[{datetime.datetime.now().isoformat()}] Starting KMeans clustering...")
    cluster_index_map = []
    class_to_clusters = {}

    unique_classes = sorted(set(labels_np.tolist()))
    print(f"[{datetime.datetime.now().isoformat()}] Found {len(unique_classes)} classes")

    for class_idx in unique_classes:
        class_indices = np.where(labels_np == class_idx)[0]
        class_features = features_np[class_indices]
        actual_clusters = min(num_clusters_per_class, len(class_indices))
        print(f"[{datetime.datetime.now().isoformat()}] KMeans for class {class_idx}: {len(class_indices)} samples, {actual_clusters} clusters...")
        if actual_clusters <= 0:
            continue

        kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init=2)
        assignments = kmeans.fit_predict(class_features)
        print(f"[{datetime.datetime.now().isoformat()}] KMeans for class {class_idx} complete")

        class_to_clusters[class_idx] = []
        for local_cluster_id in range(actual_clusters):
            member_mask = assignments == local_cluster_id
            member_indices = class_indices[member_mask].tolist()
            if not member_indices:
                continue

            cluster_entry = {
                "class_idx": int(class_idx),
                "cluster_id": len(cluster_index_map),
                "indices": member_indices,
                "size": len(member_indices),
            }
            cluster_index_map.append(cluster_entry)
            class_to_clusters[class_idx].append(cluster_entry["cluster_id"])

    total_size = sum(cluster["size"] for cluster in cluster_index_map)
    for cluster in cluster_index_map:
        cluster["weight"] = cluster["size"] / total_size

    return {
        "clusters": cluster_index_map,
        "class_to_clusters": class_to_clusters,
        "num_clusters": len(cluster_index_map),
        "num_samples": total_size,
    }


def _allocate_samples_by_weight(weights: List[float], total_batch_size: int):
    raw = [weight * total_batch_size for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total_batch_size - sum(counts)

    ranked = sorted(
        enumerate(raw),
        key=lambda item: item[1] - math.floor(item[1]),
        reverse=True,
    )
    for index, _ in ranked[:remainder]:
        counts[index] += 1

    return counts


def sample_g_plus_batch(cluster_state, total_batch_size=256, rng=None):
    rng = rng or random
    clusters = cluster_state["clusters"]
    weights = [cluster["weight"] for cluster in clusters]
    sample_counts = _allocate_samples_by_weight(weights, total_batch_size)

    sampled_indices = []
    for cluster, sample_count in zip(clusters, sample_counts):
        if sample_count <= 0:
            continue

        cluster_indices = cluster["indices"]
        replace = sample_count > len(cluster_indices)
        if replace:
            samples = [rng.choice(cluster_indices) for _ in range(sample_count)]
        else:
            samples = rng.sample(cluster_indices, sample_count)
        sampled_indices.extend(samples)

    rng.shuffle(sampled_indices)
    return sampled_indices


def build_g_plus_batch(dataset, indices, device):
    images = []
    labels = []

    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(label)

    batch_images = torch.stack(images, dim=0).to(device, non_blocking=True)
    batch_labels = torch.tensor(labels, dtype=torch.long, device=device)
    return batch_images, batch_labels


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def run_single_step_checks(
    model,
    theta_plus,
    first_f_param_before,
):
    first_f_name, first_f_virtual_param = next(iter(theta_plus.items()))
    first_f_param_after = next(model.f.parameters()).detach().clone()

    if torch.allclose(first_f_param_before, first_f_param_after):
        raise RuntimeError("Real f parameters did not update after optimizer.step().")

    if torch.allclose(first_f_virtual_param.detach(), first_f_param_after):
        raise RuntimeError(
            f"Virtual parameter '{first_f_name}' unexpectedly matched the real parameter."
        )

    h_has_grad = any(
        parameter.grad is not None for parameter in model.h.parameters() if parameter.requires_grad
    )
    if not h_has_grad:
        raise RuntimeError("No gradient reached h; the higher-order graph is broken.")
