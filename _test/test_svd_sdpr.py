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
from src.model.unet import UNet
from src.utils.keypoint_tools import get_inv_cov_weights
from src.utils.lie_algebra import se3_inv, se3_log
from src.utils.stereo_camera_model import StereoCameraModel
from visualization.plots import plot_ellipsoid

matplotlib.use("TkAgg")


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestLocalize(unittest.TestCase):
    def __init__(t, *args, **kwargs):
        super(TestLocalize, t).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        # set device
        torch.cuda.set_device(1)
        # Set seed
        set_seed(0)
        # Define camera
        camera = st.Camera(
            f_u=484.5,
            f_v=484.5,
            c_u=0.0,
            c_v=0.0,
            b=0.24,
            sigma_u=0.0,
            sigma_v=0.0,
        )
        batch_size = 1
        # Set up test problem
        r_p0s, C_p0s, r_ls = st.get_gt_setup(
            N_map=50, N_batch=batch_size, traj_type="circle", n_turns=0.25
        )
        r_p0s = torch.tensor(r_p0s)
        C_p0s = torch.tensor(C_p0s)
        r_ls = torch.tensor(r_ls)[None, :, :].expand(batch_size, -1, -1)
        # Define Stereo Camera
        stereo_cam = StereoCameraModel(0.0, 0.0, 484.5, 0.24).cuda()

        # Generate image coordinates
        cam_coords = torch.bmm(C_p0s, r_ls - r_p0s)
        cam_coords = torch.concat(
            [cam_coords, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        ).cuda()
        src_coords = torch.concat(
            [r_ls, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        )
        # img_coords = stereo_cam.camera_model(cam_coords)
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(r_p0s)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(r_p0s)  # Bx1x1
        r_0p_p = -C_p0s.bmm(r_p0s)
        trans_cols = torch.cat([r_0p_p, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([C_p0s, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4
        # Store values
        t.keypoints_3D_src = src_coords.cuda()
        t.keypoints_3D_trg = cam_coords.cuda()
        t.T_trg_src = T_trg_src
        t.stereo_cam = stereo_cam
        # Generate Scalar Weights
        t.weights = torch.ones(
            t.keypoints_3D_src.size(0), 1, t.keypoints_3D_src.size(2)
        ).cuda()
        t.stereo_cam = stereo_cam
        # Frame tranform
        t.T_s_v = torch.eye(4).cuda()

    def test_svd_forward(t):
        """Test that the SVD Block properly estimates the target transformation"""

        # Instantiate
        svd_block = SVDBlock(t.T_s_v)
        # Run forward with data

        T_trg_src = svd_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-12)

    def test_sdpr_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Instantiate
        loc_block = LocBlock(t.T_s_v)
        T_trg_src_est = loc_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        diff = se3_log(se3_inv(T_trg_src_est.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=2e-6)

    def test_sdpr_mat_weight_cost(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )
        # Instantiate
        loc_block = LocBlock(t.T_s_v)
        Q, scales, offsets = loc_block.get_obj_matrix(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        Q_vec, scales_vec, offsets_vec = loc_block.get_obj_matrix_vec(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        # Check that
        np.testing.assert_allclose(Q.cpu().numpy(), Q_vec.cpu().numpy(), atol=1e-20)

    def test_sdpr_mat_weight_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation. Use matrix weights."""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights > 0
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )

        # Instantiate
        loc_block = LocBlock(t.T_s_v)
        T_trg_src = loc_block(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )

        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-12)

    def test_inv_cov_numerical(self):
        N_pts = 1000
        # get random point
        pt = torch.tensor([3.0, 3.0, 3.0, 1.0])[None, :, None].cuda()
        # Convert to pixel space
        pixel_gt = t.stereo_cam.camera_model(pt)
        # Generate noise in pixel space
        noise_pxl = torch.randn(1, 4, N_pts) * t.stereo_cam.sigma
        pixel_noisy = pixel_gt.expand(1, -1, N_pts) + noise_pxl.cuda()
        # Convert back to 3D
        disparity = pixel_noisy[0, 2, :] - pixel_noisy[0, 0, :]
        pt_noisy = self.get_cam_points(pixel_noisy, disparity, t.stereo_cam.Q)
        noise_pt = (pt_noisy - torch.mean(pt_noisy, dim=2, keepdim=True))[:, :3, :]
        cov = torch.matmul(noise_pt, noise_pt.transpose(1, 2)) / (N_pts - 1)
        # Compute inverse covariance
        valid = (torch.ones(1, 1, 1) > 0).cuda()
        W, cov_cam = get_inv_cov_weights(pt, valid, t.stereo_cam)
        # Move back to cpu
        cov_cam = cov_cam[0, 0].cpu().detach().numpy()
        cov = cov[0].cpu().detach().numpy()
        pt = pt.cpu().detach().numpy()
        noise_pt = noise_pt.cpu().detach().numpy()
        # Plot covariance
        plt.figure()
        ax = plt.axes(projection="3d")
        # plot measurements in camera frame
        plot_ellipsoid(np.zeros((3, 1)), cov_cam, ax=ax, color="r")
        plot_ellipsoid(np.zeros((3, 1)), cov, ax=ax, color="b")
        ax.scatter3D(
            noise_pt[0, 0, :],
            noise_pt[0, 1, :],
            noise_pt[0, 2, :],
            marker=".",
            color="black",
            alpha=0.5,
        )
        plt.show()

        # Compare covariances
        np.testing.assert_allclose(cov_cam, cov, atol=1e-4)

    @staticmethod
    def get_cam_points(img_coords, point_disparities, Q):

        batch_size, _, num_points = img_coords.size()
        point_disparities = point_disparities.reshape(
            batch_size, 1, num_points
        )  # Bx1xN

        # Create the [ul, vl, d, 1] vector
        ones = torch.ones(batch_size, num_points).type_as(point_disparities)
        uvd1_pixel_coords = torch.stack(
            (
                img_coords[:, 0, :],
                img_coords[:, 1, :],
                point_disparities[:, 0, :],
                ones,
            ),
            dim=1,
        )  # Bx4xN

        # [X, Y, Z, d]^T = Q * [ul, vl, d, 1]^T
        Q_b = Q.expand(batch_size, 4, 4).cuda()
        cam_coords = Q_b.bmm(uvd1_pixel_coords)  # Bx4xN

        # [x, y, z, 1]^T = (1/d) * [X, Y, Z, d]^T
        inv_disparity = 1.0 / point_disparities  # Elementwise division
        cam_coords = cam_coords * inv_disparity  # Elementwise multiplication

        return cam_coords

    def test_inv_cov_weights(self):
        import matplotlib

        matplotlib.use("TkAgg")

        valid = t.weights > 0
        # test "valid" masking
        valid[0, 0, 0] = torch.tensor(0)
        W, cov_cam = get_inv_cov_weights(t.keypoints_3D_trg, valid, t.stereo_cam)
        id = torch.eye(3).cuda()
        assert torch.all(
            W[0, 0, :, :] == torch.eye(3).cuda()
        ), "Invalid mask not working"
        assert W.size() == (1, 30, 3, 3), "Size Incorrect"

        # Check plot
        valid[0, 0, 0] = torch.tensor(1)
        W, cov_cam = get_inv_cov_weights(t.keypoints_3D_trg, valid, t.stereo_cam)
        cov_cam = cov_cam.cpu().detach().numpy()
        targ = t.keypoints_3D_trg.cpu().detach().numpy()
        plt.figure()
        ax = plt.axes(projection="3d")
        # plot measurements in camera frame
        for i in range(t.keypoints_3D_trg.size(2)):
            plot_ellipsoid(targ[0, :3, [i]].T, cov_cam[0, i, :, :], ax=ax)
        ax.scatter3D(
            targ[0, 0, :],
            targ[0, 1, :],
            targ[0, 2, :],
            marker="*",
            color="g",
        )
        ax.scatter3D(
            0.0,
            0.0,
            0.0,
            marker="*",
            color="r",
        )

        plt.show()


if __name__ == "__main__":
    t = TestLocalize()
    # t.test_sdpr_mat_weight_forward()
    t.test_sdpr_forward()
    # t.test_svd_forward()
    # t.test_inv_cov_weights()
