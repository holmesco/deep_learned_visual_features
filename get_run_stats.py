import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame as df

from src.utils.lie_algebra import se3_log
from src.utils.statistics import Statistics


class CompareStats:
    def __init__(self, home_dir, results_path_old, results_path_new, map_ids):
        self.home_dir = home_dir
        self.results_path_old = f"{home_dir}/{results_path_old}/"
        self.results_path_new = f"{home_dir}/{results_path_new}/"
        self.plot_dir = f"{home_dir}/plots/"
        self.map_ids = map_ids

    def load_stats(self, results_dir, map_run_id):
        # Load the stats
        results_dir = results_dir + f"map_run_{map_run_id}/"
        with open(f"{results_dir}stats.pickle", "rb") as f:
            stats = pickle.load(f)
        return stats

    def compare_inliers_per_run(self, map_run_id=2):
        stats_1 = self.load_stats(self.results_path_old, map_run_id)
        stats_2 = self.load_stats(self.results_path_new, map_run_id)
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
        for map_run_id in self.map_ids:
            stats_old = self.load_stats(self.results_path_old, map_run_id)
            stats_new = self.load_stats(self.results_path_new, map_run_id)
            for live_run_id in stats_old.live_run_ids:
                num_inliers_old = np.round(
                    np.mean(stats_old.num_inliers[live_run_id])
                ).astype(int)
                num_inliers_new = np.round(
                    np.mean(stats_new.num_inliers[live_run_id])
                ).astype(int)
                outputs_se3_old = stats_old.outputs_se3[live_run_id]
                targets_se3_old = stats_old.targets_se3[live_run_id]
                outputs_se3_new = stats_new.outputs_se3[live_run_id]
                targets_se3_new = stats_new.targets_se3[live_run_id]
                rmse_old = rmse(outputs_se3_old, targets_se3_old)
                rmse_new = rmse(outputs_se3_new, targets_se3_new)
                df_list.append(
                    {
                        "map_run": map_run_id,
                        "live_run": live_run_id,
                        "pipeline": "SVD",
                        "avg_num_inliers": num_inliers_old,
                        "rmse_tr": rmse_old["rmse_tr"],
                        "rmse_rot": rmse_old["rmse_rot"],
                        "x": rmse_old["x"],
                        "y": rmse_old["y"],
                        "yaw": rmse_old["yaw"],
                    }
                )
                df_list.append(
                    {
                        "map_run": map_run_id,
                        "live_run": live_run_id,
                        "pipeline": "SDPR",
                        "avg_num_inliers": num_inliers_new,
                        "rmse_tr": rmse_new["rmse_tr"],
                        "rmse_rot": rmse_new["rmse_rot"],
                        "x": rmse_new["x"],
                        "y": rmse_new["y"],
                        "yaw": rmse_new["yaw"],
                    }
                )
        return df(df_list)


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


if __name__ == "__main__":
    results_path_old = "results/test_svd/inthedark"
    results_path_new = "results/test_sdpr_v2/inthedark"
    home = "/home/cho/projects/deep_learned_visual_features"
    map_ids = [2, 11, 16, 17, 23, 28]
    compare_stats = CompareStats(
        home,
        results_path_old=results_path_old,
        results_path_new=results_path_new,
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
