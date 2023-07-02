from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from fraud_detection import GAT, EllipticDataset, Trainer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/elliptic_gat.yaml",
        required=True,
        help="Path to training config",
    )
    parser.add_argument(
        "--step",
        required=True,
        help="The timestamp step to visualize predictions",
    )
    parser.add_argument(
        "--weights_file",
        default=None,
        help="Path to PyTorch weights file. Grabs from config if not provided.",
    )

    args = parser.parse_args()
    config_path = args.config
    time_step = args.step
    weights_file = args.weights_file

    config = OmegaConf.load(config_path)

    dataset = EllipticDataset(config.dataset)
    config.model.input_dim = dataset.pyg_dataset().num_node_features

    model = GAT(config.model)
    if weights_file is None:
        weights_file = f"weights/{config.name}.pt"
    model.load_state_dict(torch.load(weights_file))
    trainer = Trainer(config)
    trainer.model = model.double().to(config.train.device)

    trainer.visualize(dataset, time_step=time_step, save_to=f"visualizations/{config.name}/{time_step}.png")
