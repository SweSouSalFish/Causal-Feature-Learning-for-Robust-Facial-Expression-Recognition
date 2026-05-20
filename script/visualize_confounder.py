import argparse
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .dataset import create_dataloaders
    from .model import CICFModel
except ImportError:
    from dataset import create_dataloaders
    from model import CICFModel


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize counterfactual Grad-CAM for ERM and CICF using Original and Gray-CF."
    )
    parser.add_argument("--data_dir", type=str, default="dataset_classified_biased")
    parser.add_argument("--erm_checkpoint", type=str, default="checkpoints/erm_best_bs84_e50.pt")
    parser.add_argument("--cicf_checkpoint", type=str, default="checkpoints/50_best.pt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/feature_visualizations/test_set_original_gray",
        help="Directory where all generated figures will be saved.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        choices=["all", "class"],
        help="Use all samples from the split or only a single target class.",
    )
    parser.add_argument("--target_class", type=str, default="anger")
    parser.add_argument("--max_examples", type=int, default=24)
    parser.add_argument(
        "--filter_mode",
        type=str,
        default="cicf_correct_erm_one_error",
        choices=["cicf_correct_erm_error", "erm_both_error", "erm_one_error", "cicf_correct_erm_one_error", "cicf_correct_erm_both_error"],
        help="Filter mode: cicf_correct_erm_error (CICF both right, ERM at least one wrong), "
             "erm_both_error (ERM both wrong), erm_one_error (ERM one right one wrong), "
             "cicf_correct_erm_one_error (CICF both right, ERM one right one wrong), "
             "cicf_correct_erm_both_error (CICF both right, ERM both wrong)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model(checkpoint_path, num_classes, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        msg = str(exc)
        if "PytorchStreamReader failed reading zip archive" in msg:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}'. The file appears corrupted or incomplete. "
                "Please use a valid checkpoint, e.g. '--cicf_checkpoint checkpoints/cicf_best.pt'."
            ) from exc
        raise

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = CICFModel(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_dataset(data_dir, split, batch_size, num_workers, image_size):
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    dataset_key = f"{split}_dataset"
    if split not in loaders or dataset_key not in loaders:
        raise ValueError(f"Split '{split}' is not available in the dataset loaders.")
    return loaders[dataset_key]


def unnormalize(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def normalize(image_np):
    image = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image.astype(np.float32))


def to_grayscale(image_np):
    gray = np.dot(image_np[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    out = np.stack([gray, gray, gray], axis=-1)
    return np.clip(out, 0.0, 1.0)


def collect_target_examples(dataset, target_idx, max_examples):
    selected = []
    for index in range(len(dataset)):
        _, label = dataset[index]
        if int(label) == int(target_idx):
            selected.append(index)
        if len(selected) >= max_examples:
            break
    return selected


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        self.activations.retain_grad()

    def close(self):
        self.handle.remove()

    def __call__(self, images, target_indices):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)

        if target_indices.ndim == 0:
            target_indices = target_indices.unsqueeze(0)

        scores = logits.gather(1, target_indices.view(-1, 1)).sum()
        scores.backward()

        activations = self.activations
        gradients = activations.grad

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)

        activations.grad = None
        return logits.detach(), cam.detach()


def build_variants(image_tensor):
    image_np = unnormalize(image_tensor)
    original = image_np
    gray = to_grayscale(image_np)

    variants_np = {
        "Original": original,
        "Gray-CF": gray,
    }
    variants_tensor = {name: normalize(img) for name, img in variants_np.items()}
    return variants_np, variants_tensor


def forward_with_cam(model, image_batch, target_indices):
    cam_extractor = GradCAM(model, model.f.layer4)
    with torch.enable_grad():
        logits, cams = cam_extractor(image_batch, target_indices)
    cam_extractor.close()

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    target_probs = probs.gather(1, target_indices.view(-1, 1)).squeeze(1)
    return preds, target_probs, cams


def sanitize_for_path(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_counterfactual(
    output_path,
    class_names,
    target_names,
    variant_images,
    erm_results,
    cicf_results,
    page_idx=0,
    total_pages=1,
):
    variants = ["Original", "Gray-CF"]
    num_examples = 1

    fig, axes = plt.subplots(3, 2, figsize=(6.8, 8.8), constrained_layout=False)

    for col, variant in enumerate(variants):
        target_name = target_names[0]

        ax_input = axes[0, col]
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        ax_input.imshow(variant_images[variant][0])
        ax_input.set_title(variant, fontsize=10)

        ax_erm = axes[1, col]
        ax_erm.set_xticks([])
        ax_erm.set_yticks([])
        ax_erm.imshow(variant_images[variant][0])
        ax_erm.imshow(
            erm_results[variant]["cam"][0],
            cmap="jet",
            alpha=0.48,
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        erm_pred_idx = int(erm_results[variant]["pred"][0])
        erm_p_target = float(erm_results[variant]["p_target"][0])
        ax_erm.set_xlabel(
            f"{class_names[erm_pred_idx]} | P({target_name})={erm_p_target:.2f}",
            fontsize=7,
            labelpad=2,
        )

        ax_cicf = axes[2, col]
        ax_cicf.set_xticks([])
        ax_cicf.set_yticks([])
        ax_cicf.imshow(variant_images[variant][0])
        ax_cicf.imshow(
            cicf_results[variant]["cam"][0],
            cmap="jet",
            alpha=0.48,
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        cicf_pred_idx = int(cicf_results[variant]["pred"][0])
        cicf_p_target = float(cicf_results[variant]["p_target"][0])
        ax_cicf.set_xlabel(
            f"{class_names[cicf_pred_idx]} | P({target_name})={cicf_p_target:.2f}",
            fontsize=7,
            labelpad=2,
        )

    axes[0, 0].set_ylabel("Input", fontsize=10)
    axes[1, 0].set_ylabel("ERM CAM", fontsize=10)
    axes[2, 0].set_ylabel("CICF CAM", fontsize=10)

    fig.suptitle(
        "Color-Confounder Counterfactual: does model rely on tint or face?",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    subtitle = (
        "Each example is expanded into Original / Gray-CF. "
        "If ERM is color-driven, its prediction and target-class probability change sharply under grayscale edits."
    )
    fig.subplots_adjust(left=0.03, right=0.995, top=0.87, bottom=0.16, wspace=0.06, hspace=0.18)
    fig.text(0.5, 0.04, subtitle, ha="center", va="bottom", fontsize=10)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_file


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = build_dataset(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    classes = dataset.classes
    if args.target_class not in classes:
        raise ValueError(f"target_class '{args.target_class}' not found in classes: {classes}")

    if args.scope == "class":
        target_idx = classes.index(args.target_class)
        selected_indices = collect_target_examples(dataset, target_idx, args.max_examples)
        if not selected_indices:
            raise RuntimeError(f"No samples found for class '{args.target_class}' in split '{args.split}'.")
    else:
        target_idx = None
        selected_indices = list(range(len(dataset)))

    erm_model = load_model(args.erm_checkpoint, args.num_classes, device)
    cicf_model = load_model(args.cicf_checkpoint, args.num_classes, device)

    saved_paths = []
    for index in selected_indices:
        image_tensor, label = dataset[index]
        label_name = classes[int(label)]
        variants_np, variants_tensor = build_variants(image_tensor)

        variant_images = {
            "Original": [variants_np["Original"]],
            "Gray-CF": [variants_np["Gray-CF"]],
        }
        variant_tensors = {
            "Original": variants_tensor["Original"].unsqueeze(0).to(device),
            "Gray-CF": variants_tensor["Gray-CF"].unsqueeze(0).to(device),
        }

        if args.scope == "class":
            target_tensor = torch.tensor([int(target_idx)], dtype=torch.long, device=device)
            target_names = [args.target_class]
            sample_scope = f"class_{args.target_class}"
        else:
            target_tensor = torch.tensor([int(label)], dtype=torch.long, device=device)
            target_names = [label_name]
            sample_scope = "testset"

        erm_results = {}
        cicf_results = {}
        for key in ["Original", "Gray-CF"]:
            erm_pred, erm_p_target, erm_cam = forward_with_cam(erm_model, variant_tensors[key], target_tensor)
            cicf_pred, cicf_p_target, cicf_cam = forward_with_cam(cicf_model, variant_tensors[key], target_tensor)

            erm_results[key] = {
                "pred": erm_pred.detach().cpu().numpy(),
                "p_target": erm_p_target.detach().cpu().numpy(),
                "cam": erm_cam.detach().cpu().numpy(),
            }
            cicf_results[key] = {
                "pred": cicf_pred.detach().cpu().numpy(),
                "p_target": cicf_p_target.detach().cpu().numpy(),
                "cam": cicf_cam.detach().cpu().numpy(),
            }

        target_label = int(label)
        cicf_orig_correct = int(cicf_results["Original"]["pred"][0]) == target_label
        cicf_gray_correct = int(cicf_results["Gray-CF"]["pred"][0]) == target_label
        cicf_both_correct = cicf_orig_correct and cicf_gray_correct

        erm_orig_correct = int(erm_results["Original"]["pred"][0]) == target_label
        erm_gray_correct = int(erm_results["Gray-CF"]["pred"][0]) == target_label
        erm_both_correct = erm_orig_correct and erm_gray_correct
        erm_both_error = not erm_orig_correct and not erm_gray_correct
        erm_one_error = (erm_orig_correct and not erm_gray_correct) or (not erm_orig_correct and erm_gray_correct)

        if args.filter_mode == "cicf_correct_erm_error":
            should_save = cicf_both_correct and (not erm_both_correct)
        elif args.filter_mode == "erm_both_error":
            should_save = erm_both_error
        elif args.filter_mode == "erm_one_error":
            should_save = erm_one_error
        elif args.filter_mode == "cicf_correct_erm_one_error":
            should_save = cicf_both_correct and erm_one_error
        elif args.filter_mode == "cicf_correct_erm_both_error":
            should_save = cicf_both_correct and erm_both_error
        else:
            should_save = False

        if not should_save:
            continue

        class_dir = sanitize_for_path(label_name if args.scope == "all" else args.target_class)
        output_name = f"{args.split}_{sample_scope}_{index:05d}_{sanitize_for_path(label_name)}.png"
        output_path = Path(args.output_dir) / class_dir / output_name
        saved_path = plot_counterfactual(
            output_path=output_path,
            class_names=classes,
            target_names=target_names,
            variant_images=variant_images,
            erm_results=erm_results,
            cicf_results=cicf_results,
        )
        saved_paths.append(saved_path)

    print(f"Saved {len(saved_paths)} counterfactual figures into: {Path(args.output_dir).resolve()}")
    if args.scope == "class":
        print(f"Target class: {args.target_class} (index={target_idx})")
    else:
        print(f"Scope: all samples in split '{args.split}'")
    print(f"Examples used: {len(selected_indices)}")


if __name__ == "__main__":
    main()