import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.lie_algebra import se3_log, se3_inv
from sdprlayer import SDPRLayer


class LocBlock(nn.Module):
    """
    Compute the relative pose between the source and target frames.
    """

    def __init__(self, T_s_v):
        """
        Initialize the SVD class.

        Args:
            T_s_v (torch.tensor): 4x4 transformation matrix providing the transform from the vehicle frame to the
                                  sensor frame.
        """
        super(LocBlock, self).__init__()

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(n_vars=13)

        self.register_buffer("T_s_v", T_s_v)

    def forward(self, keypoints_3D_src, keypoints_3D_trg, weights):
        """
        Compute the pose, T_trg_src, from the source to the target frame.

        Args:
            keypoints_3D_src (torch,tensor, Bx4xN): 3D point coordinates of keypoints from source frame.
            keypoints_3D_trg (torch,tensor, Bx4xN): 3D point coordinates of keypoints from target frame.
            weights (torch.tensor, Bx1xN): weights in range (0, 1) associated with the matched source and target
                                           points.

        Returns:
            T_trg_src (torch.tensor, Bx4x4): relative transform from the source to the target frame.
        """
        batch_size, _, n_points = keypoints_3D_src.size()

        # Construct objective function matrix
        Qs = self.get_obj_matrix(keypoints_3D_src, keypoints_3D_trg, weights)
        # Evaluate
        solver_args = {"solve_method": "SCS", "eps": 1e-9}
        x = self.sdprlayer(Qs, solver_args=solver_args)[1]
        # Extract solution
        t_trg_src_intrg = x[:, 10:]
        R_trg_src = torch.reshape(x[:, 1:10], (-1, 3, 3), order="F")
        t_src_trg_intrg = -t_trg_src_intrg
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(keypoints_3D_src)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(keypoints_3D_src)  # Bx1x1
        trans_cols = torch.cat([t_src_trg_intrg, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([R_trg_src, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4

        # Convert from sensor to vehicle frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4)
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src

    @staticmethod
    def get_obj_matrix(keypoints_3D_src, keypoints_3D_trg, weights):
        """Compute QCQP objective matrix (homogenized) based on the given points

        Args:
            keypoints_3D_src (_type_): _description_
            keypoints_3D_trg (_type_): _description_
            weights (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get batch dimension
        N_batch = keypoints_3D_src.shape[0]
        # Indices
        h = [0]
        c = slice(1, 10)
        t = slice(10, 13)
        Q_batch = []
        scales = torch.zeros(N_batch, dtype=torch.double)
        offsets = torch.zeros(N_batch, dtype=torch.double)
        for b in range(N_batch):
            Q_es = []
            for i in range(keypoints_3D_trg.shape[-1]):
                W_ij = weights[b, i] * torch.eye(3)
                m_j0_0 = keypoints_3D_src[b, :, [i]]
                if m_j0_0.shape == (1, 3):
                    m_j0_0 = m_j0_0.T
                y_ji_i = keypoints_3D_trg[b, :, [i]]
                # Define matrix
                Q_e = torch.zeros(13, 13, dtype=torch.double)
                # Diagonals
                Q_e[c, c] = kron(m_j0_0 @ m_j0_0.T, W_ij)
                Q_e[t, t] = W_ij
                Q_e[h, h] = y_ji_i.T @ W_ij @ y_ji_i
                # Off Diagonals
                Q_e[c, t] = -kron(m_j0_0, W_ij)
                Q_e[t, c] = Q_e[c, t].T
                Q_e[c, h] = -kron(m_j0_0, W_ij @ y_ji_i)
                Q_e[h, c] = Q_e[c, h].T
                Q_e[t, h] = W_ij @ y_ji_i
                Q_e[h, t] = Q_e[t, h].T

                # Add to running list of measurements
                Q_es += [Q_e]
            # Combine objective
            Q = torch.stack(Q_es).sum(dim=0)
            # remove constant offset
            offsets[b] = Q[0, 0].clone()
            Q[0, 0] = 0.0
            # Rescale
            scales[b] = torch.norm(Q, p="fro")
            Q = Q / torch.norm(Q, p="fro")
            # Add to running list of batched data matrices
            Q_batch += [Q]

        return torch.stack(Q_batch), scales, offsets


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]
    )
