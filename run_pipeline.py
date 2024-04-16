import json
import os
import pickle
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import Dataset
from src.model.pipeline import Pipeline
from src.model.unet import UNet, UNetVGG16


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestPipeline:
    def __init__(t, config, sample_ids=None):

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
        if sample_ids is not None:
            data_dir = f"{t.config['home_path']}/data"
            t.dataset = Dataset(data_dir=data_dir)
            t.dataset.get_data_from_ids(sample_ids=sample_ids)
        else:
            t.dataset = t.load_data(t.config)
        print("...Done!")
        # Load  network
        print("Loading network...")
        t.net = t.load_net(t.config)
        print("...Done!")

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
        output_se3, saved_data = pipeline.forward(
            t.net, images, disparities, pose_se3, pose_log, 0, test=True
        )
        # Check that output is close to ground truth
        diff = output_se3.inverse().bmm(pose_se3)
        diff.cpu()

        return diff

    @staticmethod
    def get_imgs(id="inthedark-27-2182-8-1886"):

        dataset, run1, frame1, run2, frame2 = id.split("-")
        im_1_fn = f"/home/cho/projects/deep_learned_visual_features/data/{dataset}/run_{run1.zfill(6)}/images/left/{frame1.zfill(6)}.png"
        im_2_fn = f"/home/cho/projects/deep_learned_visual_features/data/{dataset}/run_{run2.zfill(6)}/images/left/{frame2.zfill(6)}.png"
        # Import to open cv
        img_1 = np.uint8(cv.imread(im_1_fn, 1))
        img_1 = cv.cvtColor(img_1, cv.COLOR_RGB2BGR)
        img_2 = np.uint8(cv.imread(im_2_fn, 1))
        img_2 = cv.cvtColor(img_2, cv.COLOR_RGB2BGR)

        return img_1, img_2

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

    # get sample ids
    sample_ids = [
        "inthedark-21-2057-27-1830",
        "inthedark-1-15-19-16",
        "inthedark-27-2182-8-1886",
    ]
    # Instantiate
    t = TestPipeline("./_test/config.json", sample_ids=sample_ids)
    t.run_pipeline(
        localization="svd", mat_weights=True, idx_data="inthedark-21-2057-27-1830"
    )
