"""
    Пример запуска скрипта:
        $ python3 main.py /PATH/TO/ANNOTATION/FILE.csv
"""

import argparse

import torch
from video_model import vit_base_patch16_224

def get_args():
    parser = argparse.ArgumentParser(
        "Run Video Model training.",
        add_help=True)
    parser.add_argument(
        "--path_to_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the CSV file containing annotations."
        )
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        help="Number of action classes. Defaults to 10."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for training model. Defaults to \"cpu\"."
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help="Batch size. Defaults to 8."
    )
    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs. Defaults to 10."
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Initial learning rate. Defaults to 1e-3."
    )
    parser.add_argument(
        "--sampling_strategy",
        default="fixed",
        type=str,
        help="Frame sampling strategy from one video clip. Defaults to \"fixed\"."
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        help="Path to save trained model state_dict in .pth format."
    )

    return parser.parse_known_args()


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    device: torch.device
                    ):
    pass


if __name__ == "__main__":
    key_args, args = get_args()
    device = torch.device(key_args.device)

    model = vit_base_patch16_224(num_classes=key_args.num_classes).to(device)

    for epoch in range(key_args.num_epochs):
        ...

    if key_args.path_to_save:
        torch.save(model.state_dict(), key_args.path_to_save)
