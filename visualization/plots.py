import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Plotting:
    """
    Class for plotting results.
    """

    def __init__(self, results_dir):
        """
        Initialize plotting.

        Args:
            results_dir (string): the directory in which to store the plots.
        """
        self.results_dir = results_dir

    def plot_epoch_losses(self, epoch_losses_train, epoch_losses_valid):
        """
        Plot the average training and validation loss for each epoch. Plot each individual type of loss and also
        the weighted sum of the losses.

        Args:
            epoch_losses_train (dict): the average training losses for each epoch.
            epoch_losses_valid (dict): the average training losses for each epoch.
        """
        for loss_type in epoch_losses_train.keys():

            plt.figure()
            p1 = plt.plot(epoch_losses_train[loss_type])
            p2 = plt.plot(epoch_losses_valid[loss_type])

            plt.legend((p1[0], p2[0]), ("training", "validation"))
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.title(f"Loss for each epoch, {loss_type}")

            plt.savefig(f"{self.results_dir}loss_epoch_{loss_type}.png", format="png")
            plt.savefig(f"{self.results_dir}loss_epoch_{loss_type}.pdf", format="pdf")
            plt.close()

            plt.figure()
            p1 = plt.plot(np.log(epoch_losses_train[loss_type]))
            p2 = plt.plot(np.log(epoch_losses_valid[loss_type]))

            plt.legend((p1[0], p2[0]), ("training", "validation"))
            plt.ylabel("Log of loss")
            plt.xlabel("Epoch")
            plt.title("Log of loss for each epoch, {}".format(loss_type))

            plt.savefig(
                f"{self.results_dir}log_loss_epoch_{loss_type}.png", format="png"
            )
            plt.savefig(
                f"{self.results_dir}log_loss_epoch_{loss_type}.pdf", format="pdf"
            )
            plt.close()

    def plot_epoch_errors(self, epoch_error_train, epoch_error_valid, dof):
        """
        Plot the average error for each specified pose DOF for each epoch for training and validation.

        Args:
            epoch_error_train (dict): the average pose errors for each DOF for each epoch.
            epoch_error_valid (dict): the average pose errors for each DOF for each epoch.
            dof (list[int]): indices of the DOF to plot.
        """
        dof_str = ["x", "y", "z", "roll", "pitch", "yaw"]

        for i in range(len(dof)):
            dof_index = dof[i]

            plt.figure()
            p1 = plt.plot(epoch_error_train[:, dof_index])
            p2 = plt.plot(epoch_error_valid[:, dof_index])

            plt.legend((p1[0], p2[0]), ("training", "validation"))
            plt.ylabel("RMSE")
            plt.xlabel("Epoch")
            plt.title(f"Error for each epoch, {dof_str[dof_index]}")

            plt.savefig(
                f"{self.results_dir}error_epoch_{dof_str[dof_index]}.png", format="png"
            )
            plt.savefig(
                f"{self.results_dir}error_epoch_{dof_str[dof_index]}.pdf", format="pdf"
            )
            plt.close()

    def plot_outputs(self, outputs_log, targets_log, path_name, map_run_id, dof):
        """
        Plot estimated and target poses. Plot each of the estimated DOF separately.

        Args:
            outputs_log (dict): a map from the live run id to the estimated poses for all localized vertices on
                                that run provided as length 6 vectors stacked in a numpy array.
            targets_log (dict): a map from the live run id to the ground truth target poses for all localized
                                vertices on that run provided as length 6 vectors stacked in a numpy array.
            path_name (string): name of the path.
            map_run_id (int): the id of the run used as the map, i.e. which all the other runs are localized to.
            dof (list[int]): indices of the DOF to plot.
        """
        # Store the plots of all runs localized to the map run in the same directory.
        directory = f"{self.results_dir}/{path_name}/map_run_{map_run_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        dof_str = ["x", "y", "z", "roll", "pitch", "yaw"]

        for live_run_id in outputs_log.keys():

            for dof_ind in dof:

                # Convert to degrees if rotation.
                targets_plot = (
                    np.rad2deg(targets_log[live_run_id][:, dof_ind])
                    if dof_ind > 2
                    else targets_log[live_run_id][:, dof_ind]
                )
                outputs_plot = (
                    np.rad2deg(outputs_log[live_run_id][:, dof_ind])
                    if dof_ind > 2
                    else outputs_log[live_run_id][:, dof_ind]
                )

                f = plt.figure(figsize=(18, 6))
                f.tight_layout(rect=[0, 0.03, 1, 0.95])
                p1 = plt.plot(targets_plot, "C1")
                p2 = plt.plot(outputs_plot, "C0")
                plt.legend((p1[0], p2[0]), ("ground truth", "estimated"))

                ylabel = "Degrees" if dof_ind > 2 else "Metres"
                plt.ylabel(ylabel)
                plt.xlabel("Vertex")
                plt.title(f"Error - {dof_str[dof_ind]}")

                plt.savefig(
                    f"{directory}/pose_{dof_str[dof_ind]}_live_run_{live_run_id}.png",
                    format="png",
                )
                plt.savefig(
                    f"{directory}/pose_{dof_str[dof_ind]}_live_run_{live_run_id}.pdf",
                    format="pdf",
                )
                plt.close()


def plot_ellipsoid(bias, cov, ax=None, color="b", stds=3, label: str = None):
    """Plot a 3D ellipsoid

    Args:
        bias (_type_): offset to center of ellipsoid
        cov (_type_): covariance matrix associated with ellipsoid
        ax (_type_, optional): axis handle. Defaults to None.
        color (str, optional): color of ellipsoid. Defaults to 'b'.
        stds (int, optional): Number of standard deviations for ellipsoid. Defaults to 1.
        label (str): Label string for plotting. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:
        ax = plt.axes(projection="3d")
    # Parameterize in terms of angles
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    L = np.linalg.cholesky(cov)
    ellipsoid = (stds * L @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(
        3, *x.shape
    )
    surf = ax.plot_surface(
        *ellipsoid, rstride=4, cstride=4, color=color, alpha=0.25, label=label
    )
    # These lines required for legend
    surf._edgecolors = surf._edgecolors3d
    surf._facecolors = surf._facecolors3d
    return surf
