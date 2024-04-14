"""A script that shows how to pass an image to the network to get keypoints, descriptors and scrores. """

import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model.keypoint_block import KeypointBlock
from src.model.matcher_block import MatcherBlock
from src.model.unet import UNet, UNetVGG16
from src.model.weight_block import WeightBlock
from src.utils.keypoint_tools import get_norm_descriptors, get_scores, normalize_coords


def get_keypoint_info(kpt_2D, scores_map, descriptors_map):
    """
    Gather information we need associated with each detected keypoint. Compute the normalized
    descriptor and the score for each keypoint.

    Args:
        kpt_2D (torch.tensor): keypoint 2D image coordinates, (Bx2xN).
        scores_map (torch.tensor): scores for each pixel, (Bx1xHxW).
        descriptors_map (torch.tensor): descriptors for each pixel, (BxCxHxW).

    Returns:
        kpt_desc_norm (torch.tensor): Normalized descriptor for each keypoint, (BxCxN).
        kpt_scores (torch.tensor): score for each keypoint, (Bx1xN).

    """
    batch_size, _, height, width = scores_map.size()

    kpt_2D_norm = normalize_coords(kpt_2D, batch_size, height, width).unsqueeze(
        1
    )  # Bx1xNx2

    kpt_desc_norm = get_norm_descriptors(descriptors_map, True, kpt_2D_norm)

    kpt_scores = get_scores(scores_map, kpt_2D_norm)

    return kpt_desc_norm, kpt_scores


class LearnedFeatureBlocks(nn.Module):
    """
    Class to detect learned features.
    """

    def __init__(
        self,
        n_channels,
        layer_size,
        window_height,
        window_width,
        image_height,
        image_width,
        checkpoint_path,
        cuda,
        vgg16=False,
    ):
        """
        Set the variables needed to initialize the network.

        Args:
            num_channels (int): number of channels in the input image (we use 3 for one RGB image).
            layer_size (int): size of the first layer if the encoder. The size of the following layers are
                              determined from this.
            window_height (int): height of window, inside which we detect one keypoint.
            window_width (int): width of window, inside which we detect one keypoint.
            image_height (int): height of the image.
            image_width (int): width of the image.
            checkpoint_path (string): path to where the network weights are stored.
            cuda (bool): true if using the GPU.
        """
        super(LearnedFeatureBlocks, self).__init__()

        self.cuda = cuda
        self.n_classes = 1
        self.n_channels = n_channels
        self.layer_size = layer_size
        self.window_h = window_height
        self.window_w = window_width
        self.height = image_height
        self.width = image_width

        # Load the network weights from a checkpoint.
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise RuntimeError(
                f"The specified checkpoint path does not exists: {checkpoint_path}"
            )
        if vgg16:
            self.net = UNetVGG16(self.n_channels, self.n_classes, False)
        else:
            self.net = UNet(
                self.n_channels,
                self.n_classes,
                self.layer_size,
            )
        # self.net = UNet(self.n_channels, self.n_classes, self.layer_size, self.height, self.width, checkpoint)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.keypoint_block = KeypointBlock(
            self.window_h, self.window_w, self.height, self.width
        )
        self.matcher_block = MatcherBlock()
        self.weight_block = WeightBlock()
        self.sigmoid = nn.Sigmoid()

        if cuda:
            self.net.cuda()
            self.keypoint_block.cuda()

        self.net.eval()

        self.image_transforms = transforms.Compose([transforms.ToTensor()])

    def process_img(self, image_tensor):
        """
        Forward pass of network to get keypoint detector values, descriptors and, scores

        Args:
            image_tensor (torch.tensor, Bx3xHxW): RGB images to input to the network.

        Returns:
            keypoints (torch.tensor, Bx2xN): the detected keypoints, N=number of keypoints.
            descriptors (torch.tensor, BxCxN): descriptors for each keypoint, C=496 is length of descriptor.
            scores (torch.tensor, Bx1xN): an importance score for each keypoint.

        """
        if self.cuda:
            image_tensor = image_tensor.cuda()

        detector_scores, scores, descriptors = self.net(image_tensor)
        scores = self.sigmoid(scores)

        # Get 2D keypoint coordinates from detector scores, Bx2xN
        keypoints = self.keypoint_block(detector_scores)

        # Get one descriptor and scrore per keypoint, BxCxN, Bx1xN, C=496.
        point_descriptors_norm, point_scores = get_keypoint_info(
            keypoints, scores, descriptors
        )

        return (
            keypoints.detach().cpu(),
            point_descriptors_norm.detach().cpu(),
            point_scores.detach().cpu(),
            scores.detach().cpu(),
        )

    def import_img(self, im_fn):
        """Apply the required transforms to process image with net"""
        # Convert to tensor
        img = Image.open(im_fn)
        img_tensor = self.image_transforms(img)[None, :, :, :]
        # Import to open cv
        img_cv = np.uint8(cv.imread(im_fn, 1))
        img_cv = cv.cvtColor(img_cv, cv.COLOR_RGB2BGR)

        return img_tensor, img_cv


def plot_matches(blocks: LearnedFeatureBlocks, im_1_fn, im_2_fn):
    """Run pipeline up to matcher block and plot matches according to weights.

    Args:
        blocks (LearnedFeatureBlocks): _description_
        im_1_fn (_type_): _description_
        im_2_fn (_type_): _description_
    """
    # NOTE dimensions for opencv are (H,W,C) but feature detector expects (B, C, H, W)
    # Load Images
    im1, im1_cv = blocks.import_img(im_1_fn)
    im2, im2_cv = blocks.import_img(im_2_fn)

    # Get keypoints and descriptors -
    kp1, des1, scores1, score_map1 = blocks.process_img(im1)
    kp2, des2, scores2, score_map2 = blocks.process_img(im2)
    # Plot score heatmaps
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im1_cv)
    axs[0].imshow(score_map1[0, 0, :, :], cmap="jet", alpha=0.4)
    axs[0].set_xticks([]), axs[0].set_yticks([])
    axs[1].imshow(im2_cv)
    axs[1].imshow(score_map2[0, 0, :, :], cmap="jet", alpha=0.4)
    axs[1].set_xticks([]), axs[1].set_yticks([])
    plt.tight_layout()
    # Run matcher to get pseudo-points
    kp2_pseudo = blocks.matcher_block(kp1, kp2, des1, des2)
    # Get descriptors for pseudopoints
    des2_pseudo, scores2_pseudo = get_keypoint_info(kp2_pseudo, score_map2, des2)
    # Generate weights
    weights = blocks.weight_block(des1, des2_pseudo, scores1, scores2_pseudo)
    weights = weights.numpy()
    # Generate opencv match array
    matches = [
        cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=1 - weights[0, 0, i])
        for i in range(kp1.shape[2])
    ]
    matches = sorted(matches, key=lambda x: x.distance)
    # convert to numpy remove batch dim
    kp1 = kp1.numpy()[0, :, :]
    kp2_pseudo = kp2_pseudo.numpy()[0, :, :]
    # convert to opencv keypoints
    kp1_cv = [cv.KeyPoint(kp1[0, i], kp1[1, i], 1) for i in range(kp1.shape[1])]
    kp2_cv = [
        cv.KeyPoint(kp2_pseudo[0, i], kp2_pseudo[1, i], 1)
        for i in range(kp2_pseudo.shape[1])
    ]

    # Draw first 10 matches.
    img3 = cv.drawMatches(
        im1_cv,
        kp1_cv,
        im2_cv,
        kp2_cv,
        matches[:200],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure()
    plt.imshow(img3)
    plt.show()


if __name__ == "__main__":

    cuda = False
    vgg16 = True
    if vgg16 == True:
        checkpoint = "/home/cho/projects/deep_learned_visual_features/networks/network_vgg16_sdpr_4_debug_mw_nograd.pth"
    else:
        checkpoint = "/home/cho/projects/deep_learned_visual_features/networks/network_multiseason_inthedark_layer16.pth"
    learned_feature_detector = LearnedFeatureBlocks(
        n_channels=3,
        layer_size=16,
        window_height=16,
        window_width=16,
        image_height=384,
        image_width=512,
        checkpoint_path=checkpoint,
        cuda=cuda,
        vgg16=vgg16,
    )

    # Image paths - ID inthedark-27-2182-8-1886
    im_1_fn = "/home/cho/projects/deep_learned_visual_features/data/inthedark/run_000027/images/left/002182.png"
    im_2_fn = "/home/cho/projects/deep_learned_visual_features/data/inthedark/run_000008/images/left/001886.png"

    plot_matches(learned_feature_detector, im_1_fn, im_2_fn)
