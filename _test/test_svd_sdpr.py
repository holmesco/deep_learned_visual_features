import unittest

import numpy as np
import torch
from src.model.svd_block import SVDBlock

import sdprlayer.stereo_tuner as st
from sdprlayer import SDPRLayer


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
        r_p0s, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=30, batch_size=1
        )
        # Generate Euclidean measurements
        meas, weights = camera.inverse(pixel_meass)
        # Convert to scalar weights (minimum eigval)
        weights_eigs = torch.linalg.eigvalsh(weights)
        weights = torch.min(weights_eigs, 1)
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(r_p0s)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(r_p0s)  # Bx1x1
        r_0p_p = -C_p0s.bmm(r_p0s)
        trans_cols = torch.cat([r_0p_p, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([C_p0s, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4
        # Store values
        t.keypoints_3D_src = r_ls
        t.keypoints_3D_trg = meas
        t.weights = weights
        t.T_trg_src = T_trg_src

    def test_svd_forward(t):
        """Test that the SVD Block properly estimates the target transformation"""

        # Instantiate
        svd_block = SVDBlock(torch.eye(4))
        # Run forward with data
        T_trg_src = svd_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        np.testing.assert_allclose(T_trg_src.numpy(), t.T_trg_src.numpy())


if __name__ == "__main__":
    t = TestLocalize()
    t.test_svd_forward()
