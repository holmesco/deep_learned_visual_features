import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import sdprlayer.stereo_tuner as st
import torch

from src.model.loc_block import LocBlock
from src.model.svd_block import SVDBlock
from src.utils.keypoint_tools import get_inv_cov_weights
from src.utils.stereo_camera_model import StereoCameraModel
from visualization.plots import plot_ellipsoid


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
        r_p0s, C_p0s, r_ls = st.get_gt_setup(N_map=30, N_batch=batch_size)
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

    def test_svd_forward(t):
        """Test that the SVD Block properly estimates the target transformation"""

        # Instantiate
        svd_block = SVDBlock(torch.eye(4))
        # Run forward with data
        weights = torch.ones(
            t.keypoints_3D_src.size(0), 1, t.keypoints_3D_src.size(2)
        ).cuda()
        T_trg_src = svd_block(t.keypoints_3D_src, t.keypoints_3D_trg, weights)
        # Check that
        np.testing.assert_allclose(T_trg_src.numpy(), t.T_trg_src.numpy(), atol=1e-14)

    def test_loc_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Instantiate
        loc_block = LocBlock(torch.eye(4))
        # Run forward with data
        weights = torch.ones(
            t.keypoints_3D_src.size(0), 1, t.keypoints_3D_src.size(2)
        ).cuda()
        T_trg_src = loc_block(t.keypoints_3D_src, t.keypoints_3D_trg, weights)
        # Check that
        np.testing.assert_allclose(
            T_trg_src.cpu().numpy(), t.T_trg_src.numpy(), atol=1e-7
        )

    def test_inv_cov_weights(self):
        import matplotlib

        matplotlib.use("TkAgg")
        W, cov_cam = get_inv_cov_weights(t.keypoints_3D_trg, t.stereo_cam)
        assert W.size() == (1, 30, 3, 3)

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
        plt.show()


if __name__ == "__main__":
    t = TestLocalize()
    t.test_inv_cov_weights()
