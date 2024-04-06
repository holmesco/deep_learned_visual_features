import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mwcerts.mat_weight_problem import Constraint
from poly_matrix import PolyMatrix
from sdprlayer import SDPRLayer

from src.utils.lie_algebra import se3_inv, se3_log


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

        # Generate constraints
        constraints = (
            self.gen_orthoganal_constraints()
            + self.gen_handedness_constraints()
            + self.gen_row_col_constraints()
        )

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(n_vars=13, constraints=constraints)

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

        # Construct objective function
        Qs, _, _ = self.get_obj_matrix_vec(keypoints_3D_src, keypoints_3D_trg, weights)
        # Evaluate
        solver_args = {"solve_method": "SCS", "eps": 1e-9, "verbose": False}
        x = self.sdprlayer(Qs, solver_args=solver_args)[1]
        # Extract solution
        t_trg_src_intrg = x[:, 9:]
        R_trg_src = torch.reshape(x[:, 0:9], (-1, 3, 3)).transpose(-1, -2)
        t_src_trg_intrg = -t_trg_src_intrg
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(keypoints_3D_src)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(keypoints_3D_src)  # Bx1x1
        trans_cols = torch.cat([t_src_trg_intrg, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([R_trg_src, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4

        # Convert from sensor to vehicle frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4).cuda()
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src

    @staticmethod
    def get_obj_matrix(keypoints_3D_src, keypoints_3D_trg, weights):
        """
        Compute the QCQP (Quadratically Constrained Quadratic Program) objective matrix
        based on the given 3D keypoints from source and target frames, and their corresponding weights.

        This function is currently not vectorized and iterates over each batch and each keypoint.

        Args:
            keypoints_3D_src (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the source frame.
                                             N_batch is the batch size and N is the number of keypoints.
            keypoints_3D_trg (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the target frame.
                                             N_batch is the batch size and N is the number of keypoints.
            weights (torch.Tensor): A tensor of shape (N_batch, 1, N) representing the weights
                                    corresponding to each keypoint.

        Returns:
            list: A list of tensors representing the QCQP objective matrices for each batch.
        """
        # TODO should vectorize this function to improve runtime
        # Get batch dimension
        N_batch = keypoints_3D_src.shape[0]
        # Indices
        h = [0]
        c = slice(1, 10)
        t = slice(10, 13)
        Q_batch = []
        scales = torch.zeros(N_batch).cuda()
        offsets = torch.zeros(N_batch).cuda()
        for b in range(N_batch):
            Q_es = []
            for i in range(keypoints_3D_trg.shape[-1]):
                W_ij = weights[b, 0, i] * torch.eye(3).cuda()
                m_j0_0 = keypoints_3D_src[b, :3, [i]]
                y_ji_i = keypoints_3D_trg[b, :3, [i]]
                # Define matrix
                Q_e = torch.zeros(13, 13).cuda()
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

    @staticmethod
    def get_obj_matrix_vec(keypoints_3D_src, keypoints_3D_trg, weights):
        """Compute the QCQP (Quadratically Constrained Quadratic Program) objective matrix
        based on the given 3D keypoints from source and target frames, and their corresponding weights.
        This is a vectorized version of the function above

         Args:
            keypoints_3D_src (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the source frame.
                                             N_batch is the batch size and N is the number of keypoints.
            keypoints_3D_trg (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the target frame.
                                             N_batch is the batch size and N is the number of keypoints.
            weights (torch.Tensor): A tensor of shape (N_batch, 1, N) representing the weights
                                    corresponding to each keypoint.
        Returns:
            _type_: _description_
        """
        B = keypoints_3D_src.shape[0]  # Batch dimension
        N = keypoints_3D_src.shape[2]  # Number of points
        # Indices
        h = 0
        c = slice(1, 10)
        t = slice(10, 13)
        # relabel and dehomogenize
        m = keypoints_3D_src[:, :3, :]
        y = keypoints_3D_trg[:, :3, :]
        Q_e = torch.zeros(B, N, 13, 13).cuda()
        # diagonal elements
        M = torch.einsum("bin,bjn->bnij", m, m)  # outer product
        I = torch.eye(3, 3).cuda().expand(B, N, -1, -1)  # identity
        Q_e[:, :, c, c] = kron(M, I)
        Q_e[:, :, t, t] = I
        Q_e[:, :, h, h] = torch.einsum("bin,bin->bn", y, y)  # inner product
        # Off Diagonals
        m_ = m.transpose(-1, -2).unsqueeze(3)
        y_ = y.transpose(-1, -2).unsqueeze(3)
        Q_e[:, :, c, t] = -kron(m_, I)
        Q_e[:, :, t, c] = Q_e[:, :, c, t].transpose(-1, -2)
        Q_e[:, :, c, h] = -kron(m_, y_).squeeze(-1)
        Q_e[:, :, h, c] = Q_e[:, :, c, h]
        Q_e[:, :, t, h] = y.transpose(-1, -2)
        Q_e[:, :, h, t] = Q_e[:, :, t, h]
        # Scale by weights
        weights = weights.squeeze(1)
        Q = torch.einsum("bnij,bn->bij", Q_e, weights)
        # operations below are to improve optimization conditioning
        # remove constant offset
        offsets = Q[:, 0, 0].clone()
        Q[:, 0, 0] = torch.zeros(B).cuda()
        # rescale
        scales = torch.norm(Q, p="fro")
        Q = Q / torch.norm(Q, p="fro")

        return Q, scales, offsets

    @staticmethod
    def gen_orthoganal_constraints():
        """Generate 6 orthongonality constraints for rotation matrices"""
        # labels
        h = "h"
        C = "C"
        t = "t"
        variables = {h: 1, C: 9, t: 3}
        constraints = []
        for i in range(3):
            for j in range(i, 3):
                A = PolyMatrix()
                E = np.zeros((3, 3))
                E[i, j] = 1.0 / 2.0
                A[C, C] = np.kron(E + E.T, np.eye(3))
                if i == j:
                    A[h, h] = -1.0
                else:
                    A[h, h] = 0.0
                constraints += [A.get_matrix(variables)]
        return constraints

    @staticmethod
    def gen_row_col_constraints():
        """Generate constraint that every row vector length equal every column vector length"""
        # labels
        h = "h"
        C = "C"
        t = "t"
        variables = {h: 1, C: 9, t: 3}
        # define constraints
        constraints = []
        for i in range(3):
            for j in range(3):
                A = PolyMatrix()
                c_col = np.zeros(9)
                ind = 3 * j + np.array([0, 1, 2])
                c_col[ind] = np.ones(3)
                c_row = np.zeros(9)
                ind = np.array([0, 3, 6]) + i
                c_row[ind] = np.ones(3)
                A[C, C] = np.diag(c_col - c_row)
                constraints += [A.get_matrix(variables)]
        return constraints

    @staticmethod
    def gen_handedness_constraints():
        """Generate Handedness Constraints - Equivalent to the determinant =1
        constraint for rotation matrices. See Tron,R et al:
        On the Inclusion of Determinant Constraints in Lagrangian Duality for 3D SLAM"""
        # labels
        h = "h"
        C = "C"
        t = "t"
        variables = {h: 1, C: 9, t: 3}
        # define constraints
        constraints = []
        i, j, k = 0, 1, 2
        for col_ind in range(3):
            l, m, n = 0, 1, 2
            for row_ind in range(3):
                # Define handedness matrix and vector
                mat = np.zeros((9, 9))
                mat[3 * j + m, 3 * k + n] = 1 / 2
                mat[3 * j + n, 3 * k + m] = -1 / 2
                mat = mat + mat.T
                vec = np.zeros((9, 1))
                vec[i * 3 + l] = -1 / 2
                # Create constraint
                A = PolyMatrix()
                A[C, C] = mat
                A[C, h] = vec
                constraints += [A.get_matrix(variables)]
                # cycle row indices
                l, m, n = m, n, l
            # Cycle column indicies
            i, j, k = j, k, i
        return constraints


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    prod = A[..., :, None, :, None] * B[..., None, :, None, :]
    other_dims = tuple(A.shape[:-2])
    return prod.reshape(
        *other_dims, A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]
    )


def bkron(a, b):
    """
    Compute the Kronecker product between two matrices a and b.

    Args:
        a (torch.Tensor): A tensor of shape (..., M, N).
        b (torch.Tensor): A tensor of shape (..., P, Q).

    Returns:
        torch.Tensor: The Kronecker product of a and b, a tensor of shape (..., M*P, N*Q).
    """
    return torch.einsum("...ij,...kl->...ikjl", a, b).reshape(
        *a.shape[:-2], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]
    )
