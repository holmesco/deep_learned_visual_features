import json
import pickle
import sys
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sdprlayer.stereo_tuner as st
import torch

from src.dataset import Dataset
from src.model.loc_block import LocBlock
from src.model.pipeline import Pipeline
from src.model.svd_block import SVDBlock
from src.model.unet import UNet, UNetVGG16
from src.utils.keypoint_tools import get_inv_cov_weights
from src.utils.lie_algebra import se3_inv, se3_log
from src.utils.stereo_camera_model import StereoCameraModel
from visualization.plots import plot_ellipsoid


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestPipeline(unittest.TestCase):
    def __init__(t, config, *args, **kwargs):
        super(TestPipeline, t).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        # Set seed
        set_seed(0)
        # Load JSON config
        t.config = json.load(open(config))
        t.config["training"]["start_pose_estimation"] = 0
        # Load dataset
        print("Loading data...")
        t.dataset = t.load_data(t.config)
        print("...Done!")
        # Load  network
        print("Loading network...")
        t.net = t.load_net(t.config)
        print("...Done!")

    def test_compare_pipelines(t, idx_data=1):
        """Test that the SVD and SDPR pipelines produce similar results"""
        # Run pipeline
        with torch.no_grad():
            pose_sdpr_mw, _ = t.run_pipeline("sdpr", True, idx_data=idx_data)
            pose_svd, pose_gt = t.run_pipeline("svd", False, idx_data=idx_data)
            pose_sdpr, _ = t.run_pipeline("sdpr", False, idx_data=idx_data)

        # Print poses
        torch.set_printoptions(sci_mode=True)
        print("POSES:")
        print("Ground truth pose:"), print(pose_gt)
        print("SVD pose:"), print(pose_svd)
        print("SDPR (scalar) pose:"), print(pose_sdpr)
        print("SDPR (matrix) pose:"), print(pose_sdpr_mw)

        pose_gt = pose_gt.to(pose_svd)
        diff_svd = se3_log(pose_svd.bmm(torch.inverse(pose_gt))).unsqueeze(2)
        diff_sdpr = se3_log(pose_sdpr.bmm(torch.inverse(pose_gt))).unsqueeze(2)
        diff_sdpr_mw = se3_log(pose_sdpr_mw.bmm(torch.inverse(pose_gt))).unsqueeze(2)
        # Print log diffs
        print("LOG DIFF POSE:")
        print(
            f"SVD: norm trans: {torch.norm(diff_svd[0,:3,0])}, norm rot: {torch.norm(diff_svd[0,3:,0])}"
        )
        print(diff_svd)
        print(
            f"SDPR (scalar): norm trans: {torch.norm(diff_sdpr[0,:3,0])}, norm rot: {torch.norm(diff_sdpr[0,3:,0])}"
        )
        print(diff_sdpr)
        print(
            f"SDPR (matrix): norm trans: {torch.norm(diff_sdpr_mw[0,:3,0])}, norm rot: {torch.norm(diff_sdpr_mw[0,3:,0])}"
        )
        print(diff_sdpr_mw)

    def run_pipeline(t, localization, mat_weights, idx_data=24):
        """Test the pipeline on inputs from the training set"""
        # Alter config
        t.config["pipeline"]["localization"] = localization
        t.config["pipeline"]["use_inv_cov_weights"] = mat_weights
        # Instantiate pipeline
        pipeline = Pipeline(t.config).cuda()
        # Load image data
        images, disparities, ids, pose_se3, pose_log = t.dataset[idx_data]
        images = images[None, :, :, :]  # Add batch dimension
        disparities = disparities[None, :, :, :]
        pose_se3 = pose_se3[None, :, :]
        pose_log = pose_log[None, :]
        # Run Pipeline
        losses, output_se3 = pipeline.forward(
            t.net, images, disparities, pose_se3, pose_log, 0
        )
        # Check that output is close to ground truth
        output_se3 = output_se3.cpu()

        return output_se3, pose_se3

    @staticmethod
    def load_data(config):
        data_path = f"{config['home_path']}/data"
        datasets_path = f"{config['home_path']}/datasets"
        dataset_name = config["dataset_name"]
        dataset_path = f"{datasets_path}/{dataset_name}.pickle"

        # Load the data.
        dataset_params = config["dataset"]
        dataset_params["data_dir"] = data_path

        localization_data = None
        with open(dataset_path, "rb") as handle:
            localization_data = pickle.load(handle)

        # Training data generator (randomly sample a subset of the full dataset for each epoch).
        train_set = Dataset(**dataset_params)
        train_set.load_mel_data(localization_data, "training")

        return train_set

    @staticmethod
    def load_net(config):
        # Instantiate network
        if config["network"]["type"] == "unet":
            net = UNet(
                config["network"]["num_channels"],
                config["network"]["num_classes"],
                config["network"]["layer_size"],
            )
        elif config["network"]["type"] == "unet_vgg16":
            net = UNetVGG16(
                config["network"]["num_channels"],
                config["network"]["num_classes"],
                config["network"]["layer_size"],
            )
        # Load network state from dictionary (checkpoint)
        checkpoints_path = f"{config['home_path']}/networks"
        checkpoint_name = config["checkpoint_name"]
        checkpoint_path = f"{checkpoints_path}/{checkpoint_name}"
        checkpoint = torch.load(f"{checkpoint_path}.pth")
        net.load_state_dict(checkpoint["model_state_dict"])
        net.cuda()

        return net


if __name__ == "__main__":
    # New Network
    # t = TestPipeline("./_test/config_vgg16.json")
    # Mona's Network
    t = TestPipeline("./_test/config.json")
    # Lens flare:
    t.test_compare_pipelines(4)
