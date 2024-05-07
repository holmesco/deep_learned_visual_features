import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import use
from pandas import DataFrame as df

from data.build_datasets import build_sub_graph
from src.utils.lie_algebra import se3_log
from src.utils.statistics import Statistics

use("tkagg")


class CompareStats:
    def __init__(self, home_dir, results_paths, labels, map_ids):
        self.home_dir = home_dir
        self.results_paths = [f"{home_dir}/{path}/" for path in results_paths]
        self.plot_dir = f"{home_dir}/plots/"
        self.map_ids = map_ids
        self.labels = labels
        self.data_dir = f"{self.home_dir}/data/inthedark"
        self.pose_graph = build_sub_graph(map_ids, self.data_dir)

    def load_stats(self, results_dir, map_run_id):
        # Load the stats
        results_dir = results_dir + f"map_run_{map_run_id}/"
        with open(f"{results_dir}stats.pickle", "rb") as f:
            stats = pickle.load(f)
        return stats

    def compare_inliers_per_run(self, map_run_id=2):
        stats_1 = self.load_stats(self.results_paths[0], map_run_id)
        stats_2 = self.load_stats(self.results_paths[1], map_run_id)
        plt.figure()
        for live_id in stats_1.live_run_ids:
            p1 = plt.plot(stats_1.num_inliers[live_id], label=f"Run {live_id} - SVD")
            p2 = plt.plot(stats_2.num_inliers[live_id], label=f"Run {live_id} - SDPR")

        plt.xlabel("Time Index")
        plt.ylabel("Number of Inliers")
        plt.legend()

    def get_aggregate_stats(self):
        """Get the aggregate stats for all the map runs. Return dataframe.

        Args:
            map_ids (list, optional): _description_. Defaults to [2,11,16,17,23,28].
        """
        # Dictionary: [ map_run, live_run, pipeline, avg_num_inliers, rmse_tr, rmse_rot]
        df_list = []
        for i, results_path in enumerate(self.results_paths):
            for map_run_id in self.map_ids:
                stats = self.load_stats(results_path, map_run_id)
                for live_run_id in stats.live_run_ids:

                    num_inliers = np.round(
                        np.mean(stats.num_inliers[live_run_id])
                    ).astype(int)
                    if hasattr(stats, "run_times"):
                        runtime = np.mean(stats.run_times[live_run_id])
                    else:
                        runtime = np.nan
                    outputs_se3 = stats.outputs_se3[live_run_id]
                    targets_se3 = stats.targets_se3[live_run_id]
                    res_dict = {
                        "map_run": map_run_id,
                        "live_run": live_run_id,
                        "pipeline": self.labels[i],
                        "avg_num_inliers": num_inliers,
                        "runtime": runtime,
                    }
                    # If getting local optimizer results then filter local minima
                    filter = "local" in self.labels[i].lower()
                    # Get RMSE data
                    rmse_data, rmse_data_filt = rmse(
                        outputs_se3, targets_se3, filter=filter
                    )
                    # Store data
                    df_list.append(res_dict | rmse_data)
                    # if filtered data is available, also store
                    if rmse_data_filt is not None:
                        res_dict_filt = res_dict.copy()
                        res_dict_filt["pipeline"] = (
                            res_dict_filt["pipeline"] + " (filtered)"
                        )
                        df_list.append(res_dict_filt | rmse_data_filt)

        return df(df_list)

    def find_local_minima(self, map_run_id, live_run_id):
        """Search for indices of local minima in a given run

        Args:
            stats (Statistics): _description_
            live_run_id (_type_): _description_
        """
        # Assume the first set of results is the local solver
        stats = self.load_stats(self.results_paths[0], map_run_id)
        est = stats.get_outputs_se3()[live_run_id]
        trg = stats.get_targets_se3()[live_run_id]
        diffs = se3_log(torch.tensor(est).inverse().bmm(torch.tensor(trg)))
        local_min = torch.abs(diffs[:, 5]) > 0.1
        indices = [i for i in range(len(local_min)) if local_min[i]]
        return indices

    def get_poses_abs(self, stats: Statistics, live_run_id, ds_factor=10, add_inds=[]):
        """Get absolute poses, with origin at start frame of map run

        Args:
            stats (_type_): statistics object
            map_run_id (_type_): map run id
            live_run_id (_type_): live run id
            ds_factor : downsampling factor
            add_inds : list of additional indices to add (for local minima)
        """
        # get relative tranforms
        T_live_map_est = stats.get_outputs_se3()[live_run_id]
        T_live_map_gt = stats.get_targets_se3()[live_run_id]

        sample_ids = stats.get_sample_ids()[live_run_id]
        poses_est = []
        poses_gt = []

        for i, id in enumerate(sample_ids):
            if i % ds_factor == 0 or i in add_inds:
                # Parse id
                parsed_id = id.split("-")
                map_id, map_pose_id = int(parsed_id[-2]), int(parsed_id[-1])
                # Get closest map point
                T_iMap_0 = self.pose_graph.get_transform(
                    (map_id, 0), (map_id, map_pose_id)
                )
                T_iMap_0 = T_iMap_0.matrix
                # Get absolute tranforms
                T_iLive_0_est = T_live_map_est[i] @ T_iMap_0
                T_iLive_0_gt = T_live_map_gt[i] @ T_iMap_0
                T_0_iLive_est = np.linalg.inv(T_iLive_0_est)
                T_0_iLive_gt = np.linalg.inv(T_iLive_0_gt)
                # add to list
                poses_est.append(T_0_iLive_est)
                poses_gt.append(T_0_iLive_gt)

        return poses_est, poses_gt

    def plot_trajectories(
        self,
        map_run_id,
        live_run_id,
        ds_factor=10,
        frame_colors=None,
        inds=None,
        add_inds=[],
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if frame_colors is None:
            frame_colors = [None] * len(self.results_paths)
        for i, results_path in enumerate(self.results_paths):
            # load stats
            stats = self.load_stats(results_path, map_run_id)
            # get absolute trajectories
            poses_est, poses_gt = self.get_poses_abs(
                stats, live_run_id, ds_factor=ds_factor, add_inds=add_inds
            )
            # Limit run indices
            if inds is not None:
                poses_est = poses_est[inds]
                poses_gt = poses_gt[inds]

            self.plot_trajectory(
                ax,
                poses_est,
                axis_alpha=0.9,
                trace_color="m",
                trace_alpha=0.3,
                frame_color=frame_colors[i],
            )
        self.plot_trajectory(
            ax, poses_gt, axis_alpha=0.2, trace_color="k", trace_alpha=0.3
        )
        # Make box the right size
        ax.set_box_aspect(
            [
                np.ptp(np.stack(poses_gt)[:, 0, 3]),
                np.ptp(np.stack(poses_gt)[:, 1, 3]),
                np.ptp(np.stack(poses_gt)[:, 2, 3]),
            ]
        )
        # remove background of plot
        # ax.set_axis_off()

        return fig, ax

    @staticmethod
    def plot_trajectory(
        ax, poses, axis_alpha=0.7, trace_color="k", trace_alpha=0.5, frame_color=None
    ):
        """Plots a trajectory of poses. Assumes poses transform from world to vehicle frame."""

        if frame_color is None:
            frame_color = ["r", "g", "b"]
        origin_prev = None
        for i, T in enumerate(poses):
            x_axis = T[:3, 0]
            y_axis = T[:3, 1]
            z_axis = T[:3, 2]
            origin = T[:3, 3]
            # Plot the original and transformed coordinate axes
            length = 1.0
            ax.quiver(
                *origin,
                *x_axis,
                color=frame_color[0],
                alpha=axis_alpha,
                length=length,
            )
            ax.quiver(
                *origin,
                *y_axis,
                color=frame_color[1],
                alpha=axis_alpha,
                length=length,
            )
            ax.quiver(
                *origin,
                *z_axis,
                color=frame_color[2],
                alpha=axis_alpha,
                length=length,
            )

            # plot tracer line along trajectory
            if i > 0:
                line = np.array([origin_prev, origin])
                ax.plot(
                    line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color=trace_color,
                    alpha=trace_alpha,
                )
            # Store previous origin
            origin_prev = origin.copy()

    def savefig(self, fig, name, dpi=600):

        directory = f"{self.home_dir}/plots/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(f"{directory}/{name}.png", format="png", dpi=dpi, transparent=True)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def rmse(outputs_se3, targets_se3, filter=False):
    """
    Compute the rotation and translation RMSE for the SE(3) poses provided. Compute RMSE for ich live run
    individually.

    Args:
        outputs_se3 (dict): map from id of the localized live run to a list of estimated pose transforms
                            represented as 4x4 numpy arrays.
        outputs_se3 (dict): map from id of the localized live run to a list of ground truth pose transforms
                            represented as 4x4 numpy arrays.
    """

    out_mat = torch.from_numpy(np.stack(outputs_se3, axis=0))
    trg_mat = torch.from_numpy(np.stack(targets_se3, axis=0))

    # Get the difference in pose by T_diff = T_trg * inv(T_src)
    diff_mat = trg_mat.bmm(out_mat.inverse())
    diff_R = diff_mat[:, 0:3, 0:3]
    diff_tr = diff_mat[:, 0:3, 3].numpy()

    err_tr_sq = (
        (diff_tr[:, 0] * diff_tr[:, 0])
        + (diff_tr[:, 1] * diff_tr[:, 1])
        + (diff_tr[:, 2] * diff_tr[:, 2])
    )

    d = torch.acos(
        (0.5 * (diff_R[:, 0, 0] + diff_R[:, 1, 1] + diff_R[:, 2, 2] - 1.0)).clamp(
            -1 + 1e-6, 1 - 1e-6
        )
    )

    # Just get errors on important axes (as in actual runs)
    errors = se3_log(out_mat.inverse().bmm(trg_mat))
    errors[:, 3:6] = np.rad2deg(errors[:, 3:6])

    # Compute RMSE
    if filter:
        # filter local mins based on heading angle
        valid = torch.abs(errors[:, 5]) < 3.0
        rmse_dict_filt = {}
        test_errors_filt = np.sqrt(np.mean(errors[valid, :].numpy() ** 2, axis=0))
        rmse_dict_filt["x"] = test_errors_filt[0]
        rmse_dict_filt["y"] = test_errors_filt[1]
        rmse_dict_filt["yaw"] = test_errors_filt[5]
    else:
        rmse_dict_filt = None

    rmse_dict = {}
    test_errors = np.sqrt(np.mean(errors.numpy() ** 2, axis=0))
    rmse_dict["x"] = test_errors[0]
    rmse_dict["y"] = test_errors[1]
    rmse_dict["yaw"] = test_errors[5]

    # rmse_tr = np.sqrt(np.mean(err_tr_sq, axis=0))
    # rmse_rot = np.sqrt(np.mean(np.rad2deg(d.numpy()) ** 2, axis=0))
    # rmse_dict["rmse_tr"] = rmse_tr
    # rmse_dict["rmse_rot"] = rmse_rot

    return rmse_dict, rmse_dict_filt


def print_tables_RSS():
    results_paths = ["results/test_svd_v2/inthedark", "results/test_sdpr_v2/inthedark"]
    labels = ["SVD", "SDPR"]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [2, 11, 16, 17, 23, 28]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    stats = compare_stats.get_aggregate_stats()

    print("Full Table:")
    stats.sort_values(["map_run", "live_run"], inplace=True)
    # stats.drop("avg_num_inliers", axis=1, inplace=True)
    latex_tbl = stats.to_latex(float_format="%.3f", index=False)
    print(latex_tbl)

    print("Aggregated Table:")
    stats_avg = stats.groupby(["pipeline"]).mean().drop(labels=["map_run"], axis=1)
    latex_tbl = stats_avg.to_latex(float_format="%.3f")
    print(latex_tbl)


def print_tables_TRO():
    results_paths = [
        "results/TRO_test_baseline/inthedark",
        "results/TRO_test_sdp/inthedark",
        "results/TRO_test_lieopt/inthedark",
    ]
    labels = ["Baseline", "Global", "Local"]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [2, 11, 16, 17, 23, 28]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    stats = compare_stats.get_aggregate_stats()

    print("Full Table:")
    stats.sort_values(["map_run", "live_run"], inplace=True)
    stats.drop("avg_num_inliers", axis=1, inplace=True)
    latex_tbl = stats.to_latex(float_format="%.3f", index=False)
    print(latex_tbl)

    print("Aggregated Table:")
    stats_avg = stats.groupby(["pipeline"]).mean().drop(labels=["map_run"], axis=1)
    latex_tbl = stats_avg.to_latex(float_format="%.3f")
    print(latex_tbl)


def plot_traj_TRO_small(ds_factor=10):

    results_paths = [
        "results/TRO_test_lieopt_small/inthedark",
        "results/TRO_test_sdp_small/inthedark",
        "results/TRO_test_baseline_small/inthedark",
    ]
    labels = ["lieopt", "sdp", "baseline"]
    colors = [["b", "b", "b"], ["r", "r", "r"], ["g", "g", "g"]]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [0, 1, 2, 3]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    # Find local minima
    map_id, live_id = 0, "2"
    add_inds = compare_stats.find_local_minima(map_id, live_id)
    # full trajectory
    fig, ax = compare_stats.plot_trajectories(
        map_id, live_id, ds_factor=ds_factor, frame_colors=colors, add_inds=add_inds
    )
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.view_init(azim=-107.25, elev=52.12, roll=0.0)
    ax.set_axis_off()
    compare_stats.savefig(fig, "traj_map0live2", dpi=600)


def plot_traj_TRO(ds_factor=10):

    results_paths = [
        "results/TRO_test_lieopt/inthedark",
        "results/TRO_test_sdp/inthedark",
        "results/TRO_test_baseline/inthedark",
    ]
    labels = ["lieopt", "sdp", "baseline"]
    colors = [["b", "b", "b"], ["r", "r", "r"], ["g", "g", "g"]]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [2, 11, 16, 17, 23, 28]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    # Find local minima
    map_id, live_id = 2, "11"
    add_inds = compare_stats.find_local_minima(map_id, live_id)
    # full trajectory
    fig, ax = compare_stats.plot_trajectories(
        map_id, live_id, ds_factor=ds_factor, frame_colors=colors, add_inds=add_inds
    )
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.view_init(azim=-107.25, elev=52.12, roll=0.0)
    ax.set_axis_off()
    plt.show()
    compare_stats.savefig(fig, f"traj_map{map_id}live{live_id}", dpi=600)


def plot_local_mins():
    """Plot the first local minimum that can be found from the result set"""

    results_paths = [
        "results/TRO_test_lieopt/inthedark",
        "results/TRO_test_sdp/inthedark",
        "results/TRO_test_baseline/inthedark",
    ]
    labels = ["lieopt", "sdp", "baseline"]
    colors = [["b", "b", "b"], ["r", "r", "r"], ["g", "g", "g"]]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [2, 11, 16, 17, 23, 28]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    map_id, live_id = 2, "16"
    indices = compare_stats.find_local_minima(map_id, live_id)
    if len(indices) > 0:
        plt_inds = slice(indices[7] - 24, indices[7] + 24, 4)
        fig, ax = compare_stats.plot_trajectories(
            map_id, live_id, ds_factor=1, frame_colors=colors, inds=plt_inds
        )
        # Equalize axes
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.view_init(azim=-110.0, elev=29.5, roll=0.0)
        fig.set_size_inches(10, 10)
        plt.show()
        compare_stats.savefig(fig, f"local_min_map{map_id}live{live_id}")


if __name__ == "__main__":
    # plot_traj_TRO()
    # plot_local_mins()
    print_tables_RSS()
    # print_tables_TRO()
