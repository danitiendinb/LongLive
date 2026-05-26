#!/usr/bin/env python3
"""Main inference script for LongLive video generation.

This script handles inference using pre-trained LongLive models,
supporting both standard and sparse (SP) inference modes.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="LongLive: Long Video Generation Inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for video generation (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated videos",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to generate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--fp4",
        action="store_true",
        help="Use NV FP4 quantization for inference",
    )
    return parser.parse_args()


def load_config(config_path: str, fp4: bool = False) -> OmegaConf:
    """Load and merge inference configuration.

    Args:
        config_path: Path to the YAML config file.
        fp4: Whether to load the FP4 quantization config.

    Returns:
        Merged OmegaConf configuration object.
    """
    cfg = OmegaConf.load(config_path)
    if fp4:
        fp4_cfg = OmegaConf.load("configs/nvfp4/inference_nvfp4.yaml")
        cfg = OmegaConf.merge(cfg, fp4_cfg)
    return cfg


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(cfg: OmegaConf, args: argparse.Namespace):
    """Execute the inference pipeline.

    Args:
        cfg: Merged configuration object.
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Override config values with CLI arguments if provided
    if args.checkpoint:
        cfg.model.checkpoint = args.checkpoint
    if args.prompt:
        cfg.inference.prompt = args.prompt
    if args.num_frames:
        cfg.inference.num_frames = args.num_frames

    output_dir = setup_output_dir(args.output_dir)
    set_seed(args.seed)

    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    print(f"Output directory: {output_dir}")
    print("Inference pipeline initialized. Model loading coming in next update.")


def main():
    """Entry point for LongLive inference."""
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)

    cfg = load_config(args.config, fp4=args.fp4)
    run_inference(cfg, args)


if __name__ == "__main__":
    main()
