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
from src.utils.lie_algebra import se3_log


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
        # Set up device
        device = torch.device(
            "cuda:{}".format(t.config["cuda_device"])
            if torch.cuda.device_count() > 0
            else "cpu"
        )
        torch.cuda.set_device(device)
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

    def run_pipeline(t, id=0):
        """Test the pipeline on inputs from the training set"""
        localization = t.config["pipeline"]["localization"]
        print(f"Running {localization} pipeline on id {id}")
        if t.config["pipeline"]["use_inv_cov_weights"]:
            print("Using inverse covariance weights")
        # Instantiate pipeline
        pipeline = Pipeline(t.config).cuda()
        # Load image data
        images, disparities, ids, pose_se3, pose_log = t.dataset[id]
        images = images[None, :, :, :]  # Add batch dimension
        disparities = disparities[None, :, :, :]
        pose_se3 = pose_se3[None, :, :]
        pose_log = pose_log[None, :]
        # Run Pipeline
        with torch.no_grad():
            output_se3, saved_data = pipeline.forward(
                t.net,
                images,
                disparities,
                pose_se3,
                pose_log,
                0,
                test=True,
                save_data=True,
            )
            # Check that output is close to ground truth
            output_se3 = output_se3.cpu().float()
            diff = se3_log(output_se3.bmm(pose_se3.inverse()))

        return diff, saved_data

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


def compare_keypoints(id, num_matches=50):
    sample_ids = [id]
    with torch.no_grad():
        # Get images
        img_1, img_2 = get_imgs(id=id)
        # Run new pipeline
        p_new = TestPipeline(config="./_test/config_sdpr.json", sample_ids=sample_ids)
        diff_new, data_new = p_new.run_pipeline(id=id)
        # Draw matches
        img_match_new = draw_keypoint_matches(
            data_new["kpt_2D_src"],
            data_new["kpt_2D_pseudo"],
            data_new["weights"],
            img_1,
            img_2,
            num_matches=num_matches,
        )

        # run old pipeline
        p_old = TestPipeline(config="./_test/config_svd.json", sample_ids=sample_ids)
        diff_old, data_old = p_old.run_pipeline(id=id)
        # Draw matches
        img_match_old = draw_keypoint_matches(
            data_old["kpt_2D_src"],
            data_old["kpt_2D_pseudo"],
            data_old["weights"],
            img_1,
            img_2,
            num_matches=num_matches,
        )

        # plot
        dpi = 200
        width = 512 * 3 / dpi
        height = 384 * 2 / dpi
        fig, axs = plt.subplots(2, 3, figsize=(width, height))
        axs[0, 0].imshow(img_1)
        axs[0, 0].imshow(data_new["scores_src"][0, 0, :, :], cmap="jet", alpha=0.6)
        axs[0, 1].imshow(img_match_new[:, :512, :])
        axs[0, 2].imshow(img_match_new[:, 512:, :])
        axs[1, 0].imshow(img_1)
        axs[1, 0].imshow(data_old["scores_src"][0, 0, :, :], cmap="jet", alpha=0.6)
        axs[1, 1].imshow(img_match_old[:, :512, :])
        axs[1, 2].imshow(img_match_old[:, 512:, :])
        for a in axs.flatten():
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


def draw_keypoint_matches(kp1, kp2_pseudo, weights, im1_cv, im2_cv, num_matches=20):
    # Generate opencv match array
    matches = [
        cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=1 - weights[0, 0, i])
        for i in range(kp1.shape[2])
    ]
    matches = sorted(matches, key=lambda x: x.distance)
    # convert to numpy remove batch dim
    kp1 = kp1[0, :, :]
    kp2_pseudo = kp2_pseudo[0, :, :]
    # convert to opencv keypoints
    kp1_cv = [cv.KeyPoint(kp1[0, i], kp1[1, i], 1) for i in range(kp1.shape[1])]
    kp2_cv = [
        cv.KeyPoint(kp2_pseudo[0, i], kp2_pseudo[1, i], 1)
        for i in range(kp2_pseudo.shape[1])
    ]

    # Draw matches
    img3 = cv.drawMatches(
        im1_cv,
        kp1_cv,
        im2_cv,
        kp2_cv,
        matches[:num_matches],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return img3


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


if __name__ == "__main__":

    # # get sample ids
    # sample_ids = [
    #     "inthedark-21-2057-27-1830",
    #     "inthedark-1-15-19-16",
    #     "inthedark-27-2182-8-1886",
    # ]
    # # Instantiate
    # t = TestPipeline("./_test/config_vgg16.json", sample_ids=sample_ids)
    # t.run_pipeline(id="inthedark-1-15-19-16")
    compare_keypoints("inthedark-1-15-19-16")
