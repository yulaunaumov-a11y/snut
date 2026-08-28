from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device | str | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict[str, float]:

    device = torch.device(device)

    model.eval()
    correct = 0
    total = 0
    true_positives: torch.Tensor | None = None
    predicted_totals: torch.Tensor | None = None
    target_totals: torch.Tensor | None = None

    for clips, targets in tqdm(data_loader, total=len(data_loader), desc="Validate metrics", position=1, leave=False):
        logits = model(clips.to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True, dtype=torch.long)

        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
        num_classes = logits.size(1)
        if true_positives is None:
            true_positives = torch.zeros(num_classes, dtype=torch.long, device=device)
            predicted_totals = torch.zeros_like(true_positives)
            target_totals = torch.zeros_like(true_positives)

        predicted_totals += torch.bincount(predictions, minlength=num_classes)
        target_totals += torch.bincount(targets, minlength=num_classes)
        true_positives += torch.bincount(
            targets[predictions == targets], minlength=num_classes
        )

    precision = torch.where(
        predicted_totals > 0,
        true_positives / predicted_totals,
        torch.zeros_like(true_positives, dtype=torch.float),
    )
    recall = torch.where(
        target_totals > 0,
        true_positives / target_totals,
        torch.zeros_like(true_positives, dtype=torch.float),
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    active_classes = (predicted_totals + target_totals) > 0

    return {
        "accuracy": correct / total,
        "precision": precision[active_classes].mean().item(),
        "recall": recall[active_classes].mean().item(),
        "f1": f1[active_classes].mean().item(),
    }