import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import data
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from src.dataset import Dataset
from src.model.pipeline import Pipeline
from src.model.unet import UNet, UNetVGG16
from src.utils.early_stopping import EarlyStopping
from src.utils.lie_algebra import se3_log
from src.utils.statistics import Statistics
from visualization.plots import Plotting

torch.backends.cudnn.benchmark = True


def execute_epoch(
    pipeline,
    net,
    data_loader,
    stats,
    epoch,
    mode,
    config,
    dof,
    optimizer=None,
    scheduler=None,
    std_out=None,
):
    """
    Train with data for one epoch.

    Args:
        pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                             computes pose and losses.
        net (torch.nn.Module): neural network module.
        data_loader (torch.utils.data.DataLoader): data loader for training data.
        stats (Statistics): an object for keeping track of losses.
        epoch (int): current epoch.
        mode (string): 'training' or 'validation'
        config (dict): parameter configurations.
        dof (list[int]): indices of the pose DOF (out of the 6 DOF) that we want to learn.
        optimizer (torch.optim.Optimizer or None): optimizer if training or None if validation.
        scheduler(torch.optim.lr_scheduler or None): scheduler if training or None if validation.

    Returns:
        avg_loss (float): the average loss for the epoch computed over all batches. The average for the weighted
                          sum of all loss types.
        stats (Statistics): object with recorded losses for all processed data samples.
    """
    start_time = time.time()

    if mode == "training":
        net.train()
    else:
        net.eval()

    epoch_losses = {}
    errors = None
    # We want to print out the avg./max target pose absolute values for reference together with the pose errors.
    targets_total = torch.zeros(6)
    targets_max = torch.zeros(6)
    batch_size = -1
    num_batches, num_examples = 0, 0

    with torch.set_grad_enabled(mode == "training"):
        for i, data in enumerate(tqdm(data_loader, file=std_out)):
            images, disparities, ids, pose_se3, pose_log = data

            losses = {}
            batch_size = images.size(0)
            if mode == "training":
                optimizer.zero_grad()
            with record_function("Run pipeline"):

                try:
                    losses, output_se3 = pipeline.forward(
                        net, images, disparities, pose_se3, pose_log, epoch
                    )

                    if mode == "training":
                        losses["total"].backward()

                except Exception as e:
                    print(e)
                    print(f"Ids: {ids}, at Iter {num_examples}")
                    continue

            if mode == "training":
                torch.nn.utils.clip_grad_norm(
                    net.parameters(), max_norm=2.0, norm_type=2
                )
                with record_function("Optimizer"):
                    optimizer.step()

            num_batches += 1
            num_examples += batch_size

            with record_function("compute 6dof error"):
                # Get the error in each of the 6 pose DOF.
                diff = (
                    se3_log(output_se3.inverse().bmm(pose_se3.cuda()))
                    .detach()
                    .cpu()
                    .numpy()
                )
                if num_batches == 1:
                    errors = diff
                else:
                    errors = np.concatenate((errors, diff), axis=0)
            with record_function("Record epoch losses"):
                # Save the losses during the epoch.
                for loss_type in losses.keys():
                    if loss_type in epoch_losses:
                        epoch_losses[loss_type] += losses[loss_type].item()
                    else:
                        epoch_losses[loss_type] = losses[loss_type].item()

            targets_total += torch.sum(torch.abs(pose_log), dim=0).detach().cpu()
            targets_max = torch.max(
                targets_max, torch.max(torch.abs(pose_log.detach().cpu()), dim=0)[0]
            )

    # Compute and print the statistics for the finished epoch.
    print(f"{mode}, epoch {epoch}")

    # Compute the average training losses.
    for loss_type in epoch_losses.keys():
        epoch_losses[loss_type] = epoch_losses[loss_type] / float(num_batches)
        print(f"Average {mode} loss, {loss_type}: {epoch_losses[loss_type]:.6f}")

    # Only store and print pose errors if we have started computing pose (and are not just computing keypoint loss).
    if epoch >= config["training"]["start_pose_estimation"]:

        # RMSE for each pose DOF.

        errors[:, 3:6] = np.rad2deg(errors[:, 3:6])
        epoch_errors = np.sqrt(np.mean(errors**2, axis=0)).reshape(1, 6)

        stats.add_epoch_stats(epoch_losses, epoch_errors)

        targets_avg = (targets_total * (1.0 / num_examples)).detach().cpu().numpy()
        targets_avg[3:6] = np.rad2deg(targets_avg[3:6])
        targets_max = targets_max.detach().cpu().numpy()
        targets_max[3:6] = np.rad2deg(targets_max[3:6])

        # Only print the results for the relevant DOF that we are learning.
        print(f"Pose RMSE by DOF: {epoch_errors[:, np.array(dof)]}")
        print(f"Average pose targets by DOF: {targets_avg[np.array(dof)]}")
        print(f"Max pose targets by DOF: {targets_max[np.array(dof)]}")

    print(f"Epoch duration: {time.time() - start_time} seconds. \n")

    # Run if we are training, and we dare using a scheduler.
    if mode == "training":
        scheduler.step()

    return epoch_losses["total"], stats


def plot_epoch_statistics(train_stats, validation_stats, plotting, dof):
    """
    Plot the losses from training and validation over epochs.

    Args:
        train_stats (Statistics): an object for keeping track of losses and errors for the training data.
        validation_stats (Statistics): an object for keeping track of losses and errors for the validation data.
        plotting (Plotting): an object to handle plotting.
        dof (list[int]): indices of the DOF to plot.
    """
    train_epoch_stats = train_stats.get_epoch_stats()
    valid_epoch_stats = validation_stats.get_epoch_stats()

    plotting.plot_epoch_losses(train_epoch_stats[0], valid_epoch_stats[0])
    plotting.plot_epoch_errors(train_epoch_stats[1], valid_epoch_stats[1], dof)


def train(
    pipeline,
    net,
    optimizer,
    scheduler,
    train_loader,
    validation_loader,
    start_epoch,
    min_validation_loss,
    train_stats,
    validation_stats,
    config,
    results_path,
    checkpoint_path,
    std_out,
):
    """
    Run the training loop.

    Args:
        pipeline (Pipeline): the training pipeline object that performs a forward pass of the training pipeline and
                             computes pose and losses.
        net (torch.nn.Module): neural network module.
        optimizer (torch.optim.Optimizer): training optimizer.
        scheduler (torch.optim.lr_scheduler): training scheduler.
        train_loader (torch.utils.data.DataLoader): data loader for training data.
        validation_loader (torch.utils.data.DataLoader): data loader for validation data.
        start_epoch (int): epoch we start training from, 0 if starting from scratch.
        min_validation_loss (float): the minimum validation loss during training.
        train_stats (Statistics): an object for keeping track of losses for the training data.
        validation_stats (Statistics): an object for keeping track of losses for the validation data.
        config (dict): configurations for model training.
        results_path (string): directory for storing results (outputs, plots).
        checkpoint_path (string): file path to store the checkpoint.
    """
    # Helper class for plotting results.
    plotting = Plotting(results_path)

    # Early stopping keeps track of when we stop training (validation loss does not decrease) and when we store a
    # checkpoint (each time validation loss decreases below or is equal to the minimum value).
    last_save_epoch = start_epoch - 1 if start_epoch > 0 else 0
    early_stopping = EarlyStopping(
        config["training"]["patience"], min_validation_loss, last_save_epoch
    )

    # Find which of the DOF of the 6 DOF pose we want to learn. Record the indices to keep in the pose vector.
    dof = [0, 1, 5] if "pose_plane" in config["loss"]["types"] else [0, 1, 2, 3, 4, 5]

    # The training loop.
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        print(f"Epoch: {epoch}, train:", file=std_out)
        with record_function("train epoch"):
            train_loss, train_stats = execute_epoch(
                pipeline,
                net,
                train_loader,
                train_stats,
                epoch,
                "training",
                config,
                dof,
                optimizer,
                scheduler,
                std_out,
            )
        with record_function("validate epoch"):
            print(f"Epoch: {epoch}, validate:", file=std_out)
            validation_loss, validation_stats = execute_epoch(
                pipeline,
                net,
                validation_loader,
                validation_stats,
                epoch,
                "validation",
                config,
                dof,
                None,
                None,
                std_out,
            )

        # Only start checking the loss after we start computing the pose loss. Sometimes we only compute keypoint loss
        # for the first few epochs.
        if epoch >= config["training"]["start_pose_estimation"]:

            # Plot the losses during training and validations.
            plot_epoch_statistics(train_stats, validation_stats, plotting, dof)

            stop, min_validation_loss = early_stopping.check_stop(
                validation_loss,
                net,
                optimizer,
                checkpoint_path,
                train_stats,
                validation_stats,
                epoch,
            )

            # Stop the training loop if we have exceeded the patience.
            if stop:
                break
        else:
            # Save the state in the first few rounds.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "train_stats": train_stats,
                    "valid_stats": validation_stats,
                },
                "{}_{}.pth".format(checkpoint_path, epoch),
            )  # Add epoch so we don't overwrite existing file.
            print(f"Saved checkpoint for epoch {epoch} prior to pose estimation.")

        if epoch == config["training"]["max_epochs"] - 1:
            # save on the final epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "train_stats": train_stats,
                    "valid_stats": validation_stats,
                },
                "{}_{}.pth".format(checkpoint_path, epoch),
            )  # Add epoch so we don't overwrite existing file.
            print(f"Saved checkpoint for epoch {epoch} prior to pose estimation.")


def main(config, profiler_on=False, debug=False):
    """
    Set up everything needed for training and call the function to run the training loop.

    Args:
        config (dict): configurations for training the network.
    """
    if profiler_on or debug:
        config["training"]["num_samples_train"] = 3
        config["training"]["num_samples_valid"] = 3
        config["training"]["max_epochs"] = 1
        config["training"]["start_pose_estimation"] = 0
        config["dataset_name"] = "dataset_inthedark_small"
        config["experiment_name"] = config["experiment_name"] + "_prf_dbg"
        # config["dataset_name"] = "dataset_inthedark_small"
        config["debug"] = True
    elif "debug" not in config:
        config["debug"] = False
    # Set random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Set up paths
    results_path = f"{config['home_path']}/results/{config['experiment_name']}/"
    checkpoints_path = f"{config['home_path']}/networks"
    data_path = f"{config['home_path']}/data"
    datasets_path = f"{config['home_path']}/datasets"

    checkpoint_name = config["checkpoint_name"]
    dataset_name = config["dataset_name"]

    dataset_path = f"{datasets_path}/{dataset_name}.pickle"
    checkpoint_path = f"{checkpoints_path}/{checkpoint_name}"

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoint_path)

    # Print outputs to files
    orig_stdout = sys.stdout
    if not profiler_on and not config["debug"]:
        fl = open(f"{results_path}out_train.txt", "w")
        sys.stdout = fl
        orig_stderr = sys.stderr
        fe = open(f"{results_path}err_train.txt", "w")
        sys.stderr = fe

    # Record the config settings.
    print(f"\nTraining parameter configurations: \n{config}\n")

    # Load the data.
    dataloader_params = config["data_loader"]
    dataset_params = config["dataset"]
    dataset_params["data_dir"] = data_path
    # Sampling for epochs (if profiling, reduce samples)
    num_samples_train = config["training"]["num_samples_train"]
    num_samples_valid = config["training"]["num_samples_valid"]

    print(
        f"Loading dataset from {dataset_path}",
    )
    localization_data = None
    with open(dataset_path, "rb") as handle:
        localization_data = pickle.load(handle)

    # Training data generator (randomly sample a subset of the full dataset for each epoch).
    train_set = Dataset(**dataset_params)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=num_samples_train
    )
    train_set.load_mel_data(localization_data, "training")
    train_loader = data.DataLoader(
        train_set, sampler=train_sampler, **dataloader_params
    )

    # Validation data generator (randomly sample a subset of the full dataset for each epoch).
    validation_set = Dataset(**dataset_params)
    validation_sampler = RandomSampler(
        validation_set, replacement=True, num_samples=num_samples_valid
    )
    validation_set.load_mel_data(localization_data, "validation")
    validation_loader = data.DataLoader(
        validation_set, sampler=validation_sampler, **dataloader_params
    )

    # Set up device
    device = torch.device(
        "cuda:{}".format(config["cuda_device"])
        if torch.cuda.device_count() > 0
        else "cpu"
    )
    torch.cuda.set_device(device)

    # Set training pipeline
    training_pipeline = Pipeline(config)
    training_pipeline = training_pipeline.cuda()

    # Set up the network, optimizer, and scheduler
    if config["network"]["type"] == "unet":
        net = UNet(
            config["network"]["num_channels"],
            config["network"]["num_classes"],
            config["network"]["layer_size"],
        )

    elif config["network"]["type"] == "unet_vgg16":
        net = UNetVGG16(
            config["network"]["num_channels"],
            config["network"]["num_classes"],
            config["network"]["pretrained"],
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"],
    )

    # Variables for keeping track of training progress.
    start_epoch = 0
    min_validation_loss = 10e9
    train_stats = Statistics()
    validation_stats = Statistics()

    # Resume training from checkpoint if available
    if os.path.exists(f"{checkpoint_path}.pth"):
        checkpoint = torch.load(f"{checkpoint_path}.pth")
        net.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        if (
            profiler_on or config["debug"]
        ):  # restart epoch so that we can just run one epoch
            start_epoch = 0
        else:
            start_epoch = checkpoint["epoch"] + 1 if "epoch" in checkpoint.keys() else 0
        min_validation_loss = (
            checkpoint["val_loss"] if "val_loss" in checkpoint.keys() else 10e9
        )
        train_stats = (
            checkpoint["train_stats"]
            if "train_stats" in checkpoint.keys()
            else Statistics()
        )
        validation_stats = (
            checkpoint["valid_stats"]
            if "valid_stats" in checkpoint.keys()
            else Statistics()
        )

    net.cuda()

    training_inputs = [
        training_pipeline,
        net,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        start_epoch,
        min_validation_loss,
        train_stats,
        validation_stats,
        config,
        results_path,
        checkpoint_path,
        orig_stdout,
    ]
    if profiler_on:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/deep_feat"),
            profile_memory=True,
        ) as prof:
            train(*training_inputs)
    else:
        train(*training_inputs)

    if not profiler_on and not config["debug"]:
        sys.stdout = orig_stdout
        fl.close()
        sys.stderr = orig_stderr
        fe.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, type=str, help="config file path (default: None)"
    )
    parser.add_argument("--profile", action="store_true", help="profile the code.")
    parser.add_argument("--debug", action="store_true", help="profile the code.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config, profiler_on=args.profile, debug=args.debug)
