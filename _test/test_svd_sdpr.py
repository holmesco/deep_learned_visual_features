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
            N_map=30, N_batch=batch_size, traj_type="clusters"
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

    def test_svd_forward(t):
        """Test that the SVD Block properly estimates the target transformation"""

        # Instantiate
        svd_block = SVDBlock(torch.eye(4))
        # Run forward with data

        T_trg_src = svd_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        np.testing.assert_allclose(T_trg_src.numpy(), t.T_trg_src.numpy(), atol=1e-14)

    def test_sdpr_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Instantiate
        loc_block = LocBlock(torch.eye(4))
        T_trg_src = loc_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        np.testing.assert_allclose(
            T_trg_src.cpu().numpy(), t.T_trg_src.numpy(), atol=1e-7
        )

    def test_sdpr_mat_weight_cost(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )
        # Instantiate
        loc_block = LocBlock(torch.eye(4))
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
        valid = t.weights
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )
        # Instantiate
        loc_block = LocBlock(torch.eye(4))
        T_trg_src = loc_block(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        # Check that
        np.testing.assert_allclose(
            T_trg_src.cpu().numpy(), t.T_trg_src.numpy(), atol=1e-7
        )

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

        valid = t.weights
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

    def test_compare_pipelines(t):
        """Test that the SVD and SDPR pipelines produce similar results"""
        # Run pipeline
        idx_data = 1
        pose_svd, pose_gt = t.run_pipeline("svd", False, idx_data=idx_data)
        pose_sdpr, _ = t.run_pipeline("sdpr", False, idx_data=idx_data)
        pose_sdpr_mw, _ = t.run_pipeline("sdpr", True, idx_data=idx_data)

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
        print("SVD pose:"), print(diff_svd)
        print("SDPR (scalar) pose:"), print(diff_sdpr)
        print("SDPR (matrix) pose:"), print(diff_sdpr_mw)

    def run_pipeline(t, localization, mat_weights, idx_data=24):
        """Test the pipeline on inputs from the training set"""
        # Load JSON config
        config = json.load(open("./_test/config.json"))
        config["pipeline"]["localization"] = localization
        config["pipeline"]["use_inv_cov_weights"] = mat_weights
        config["training"]["start_pose_estimation"] = 0
        # Instantiate pipeline
        pipeline = Pipeline(config).cuda()
        # Load input data and network
        train_set, net = load_data_net(config)
        images, disparities, ids, pose_se3, pose_log = train_set[idx_data]
        images = images[None, :, :, :]  # Add batch dimension
        disparities = disparities[None, :, :, :]
        pose_se3 = pose_se3[None, :, :]
        pose_log = pose_log[None, :]
        # Run Pipeline
        losses, output_se3 = pipeline.forward(
            net, images, disparities, pose_se3, pose_log, 0
        )
        # Check that output is close to ground truth
        output_se3 = output_se3.cpu()

        return output_se3, pose_se3


def load_data_net(config):
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
    # Load network
    net = UNet(
        config["network"]["num_channels"],
        config["network"]["num_classes"],
        config["network"]["layer_size"],
    )
    checkpoints_path = f"{config['home_path']}/networks"
    checkpoint_name = config["checkpoint_name"]
    checkpoint_path = f"{checkpoints_path}/{checkpoint_name}"
    checkpoint = torch.load(f"{checkpoint_path}.pth")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.cuda()

    return train_set, net


if __name__ == "__main__":
    t = TestLocalize()
    # t.test_sdpr_mat_weight_forward()
    # t.test_inv_cov_weights()
    t.test_compare_pipelines()
