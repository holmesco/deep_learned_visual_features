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
                    rmse_data = rmse(outputs_se3, targets_se3)
                    df_list.append(
                        {
                            "map_run": map_run_id,
                            "live_run": live_run_id,
                            "pipeline": self.labels[i],
                            "avg_num_inliers": num_inliers,
                            "rmse_tr": rmse_data["rmse_tr"],
                            "rmse_rot": rmse_data["rmse_rot"],
                            "x": rmse_data["x"],
                            "y": rmse_data["y"],
                            "yaw": rmse_data["yaw"],
                            "runtime": runtime,
                        }
                    )

        return df(df_list)

    def get_poses_abs(self, stats: Statistics, live_run_id, ds_factor=10):
        """Get absolute poses, with origin at start frame of map run

        Args:
            stats (_type_): _description_
            map_run_id (_type_): _description_
            live_run_id (_type_): _description_
        """
        # get relative tranforms
        T_live_map_est = stats.get_outputs_se3()[live_run_id]
        T_live_map_gt = stats.get_targets_se3()[live_run_id]

        sample_ids = stats.get_sample_ids()[live_run_id]
        poses_est = []
        poses_gt = []

        for i, id in enumerate(sample_ids):
            if i % ds_factor == 0:
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

    def plot_trajectories(self, map_run_id, live_run_id, inds=None):
        # load stats
        stats = self.load_stats(self.results_paths[0], map_run_id)
        # get absolute trajectories
        poses_est, poses_gt = self.get_poses_abs(stats, live_run_id)
        # Limit run indices
        if inds is not None:
            poses_est = poses_est[inds]
            poses_gt = poses_gt[inds]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self.plot_trajectory(
            ax, poses_gt, axis_alpha=0.1, trace_color="m", trace_alpha=0.3
        )
        self.plot_trajectory(
            ax, poses_est, axis_alpha=1.0, trace_color="k", trace_alpha=0.3
        )
        # Make box the right size
        ax.set_box_aspect(
            [
                np.ptp(np.stack(poses_est)[:, 0, 3]),
                np.ptp(np.stack(poses_est)[:, 1, 3]),
                np.ptp(np.stack(poses_est)[:, 2, 3]),
            ]
        )
        # remove background of plot
        # ax.set_axis_off()

        return fig

    @staticmethod
    def plot_trajectory(ax, poses, axis_alpha=0.7, trace_color="k", trace_alpha=0.5):
        """Plots a trajectory of poses. Assumes poses transform from world to vehicle frame."""
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
                color="r",
                alpha=axis_alpha,
                length=length,
            )
            ax.quiver(
                *origin,
                *y_axis,
                color="g",
                alpha=axis_alpha,
                length=length,
            )
            ax.quiver(
                *origin,
                *z_axis,
                color="b",
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


def rmse(outputs_se3, targets_se3):
    """
    Compute the rotation and translation RMSE for the SE(3) poses provided. Compute RMSE for ich live run
    individually.

    Args:
        outputs_se3 (dict): map from id of the localized live run to a list of estimated pose transforms
                            represented as 4x4 numpy arrays.
        outputs_se3 (dict): map from id of the localized live run to a list of ground truth pose transforms
                            represented as 4x4 numpy arrays.
    """
    rmse_dict = {}

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
    rmse_tr = np.sqrt(np.mean(err_tr_sq, axis=0))

    d = torch.acos(
        (0.5 * (diff_R[:, 0, 0] + diff_R[:, 1, 1] + diff_R[:, 2, 2] - 1.0)).clamp(
            -1 + 1e-6, 1 - 1e-6
        )
    )
    rmse_rot = np.sqrt(np.mean(np.rad2deg(d.numpy()) ** 2, axis=0))
    # Just get errors on important axes (as in actual runs)
    errors = se3_log(out_mat.inverse().bmm(trg_mat))
    errors[:, 3:6] = np.rad2deg(errors[:, 3:6])
    test_errors = np.sqrt(np.mean(errors.numpy() ** 2, axis=0))

    rmse_dict["rmse_tr"] = rmse_tr
    rmse_dict["rmse_rot"] = rmse_rot
    rmse_dict["x"] = test_errors[0]
    rmse_dict["y"] = test_errors[1]
    rmse_dict["yaw"] = test_errors[5]

    return rmse_dict


def print_tables_RSS():
    results_paths = ["results/test_svd/inthedark", "results/test_sdpr_v1/inthedark"]
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
    stats.drop(labels=["rmse_tr", "rmse_rot"], axis=1, inplace=True)
    latex_tbl = stats.to_latex(float_format="%.3f", index=False)
    print(latex_tbl)

    print("Aggregated Table:")
    stats_avg = stats.groupby(["pipeline"]).mean().drop(labels=["map_run"], axis=1)
    latex_tbl = stats_avg.to_latex(float_format="%.3f")
    print(latex_tbl)


def print_tables_TRO():
    results_paths = [
        "results/TRO_test_baseline/inthedark",
        "results/TRO_test_sdpr_nomat/inthedark",
        "results/TRO_test_sdpr/inthedark",
    ]
    labels = ["Baseline", "SDPR No Mat", "SDPR"]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [0, 1, 2]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    stats = compare_stats.get_aggregate_stats()

    print("Full Table:")
    stats.drop(labels=["rmse_tr", "rmse_rot"], axis=1, inplace=True)
    latex_tbl = stats.to_latex(float_format="%.3f", index=False)
    print(latex_tbl)

    print("Aggregated Table:")
    stats_avg = stats.groupby(["pipeline"]).mean().drop(labels=["map_run"], axis=1)
    latex_tbl = stats_avg.to_latex(float_format="%.3f")
    print(latex_tbl)


def plot_traj_TRO():

    results_paths = ["results/TRO_test_sdpr/inthedark"]
    labels = ["SDPR"]
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [0, 1, 2, 3]
    compare_stats = CompareStats(
        home,
        results_paths=results_paths,
        labels=labels,
        map_ids=map_ids,
    )
    # full trajectory
    fig = compare_stats.plot_trajectories(0, "1")
    # zoomed trajectory
    fig_zoom = compare_stats.plot_trajectories(0, "1", slice(70, 90))
    plt.show()
    print("done")


if __name__ == "__main__":
    plot_traj_TRO()
    # print_tables_RSS()
    # print_tables_TRO()
