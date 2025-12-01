import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


def convert_to_np(values):
    return [value.detach().cpu().numpy() for value in values]


def create_embeddings(predictions, split):
    _catalog = {}

    for col, value in predictions["catalog"].items():
        _catalog[col] = convert_to_np(value)

    _catalog["z"] = convert_to_np(predictions["z"])

    predictions.pop("catalog")
    predictions.update(_catalog)

    del _catalog

    return Dataset.from_dict(predictions, split=split)


def create_samples(samples, split):
    _catalog = {}

    for col, value in samples["catalog"].items():
        _catalog[col] = convert_to_np(value)

    _catalog["z"] = convert_to_np(samples["z"])

    _catalog["z_prime"] = convert_to_np(samples["z_prime"])

    samples.pop("catalog")
    samples.update(_catalog)

    del _catalog

    return Dataset.from_dict(samples, split=split)


def create_timestep_embeddings(samples, split):
    _dset = {}

    """for col, value in samples["catalog"].items():
        _dset[col] = convert_to_np(value)"""

    _dset["z"] = convert_to_np(samples["z"])

    _dset["ode_cond_trajectory_20"] = samples["output_cond"]["sol"][19].numpy().tolist()
    # this gives t = 80
    _dset["ode_uncond_trajectory_20"] = (
        samples["output_uncond"]["sol"][19].numpy().tolist()
    )

    _dset["ode_cond_trajectory_40"] = samples["output_cond"]["sol"][39].numpy().tolist()
    _dset["ode_uncond_trajectory_40"] = (
        samples["output_uncond"]["sol"][39].numpy().tolist()
    )

    _dset["ode_cond_trajectory_60"] = samples["output_cond"]["sol"][59].numpy().tolist()
    _dset["ode_uncond_trajectory_60"] = (
        samples["output_uncond"]["sol"][59].numpy().tolist()
    )

    _dset["ode_cond_trajectory_80"] = samples["output_cond"]["sol"][79].numpy().tolist()
    _dset["ode_uncond_trajectory_80"] = (
        samples["output_uncond"]["sol"][79].numpy().tolist()
    )

    _dset["ode_cond_trajectory_end"] = (
        samples["output_cond"]["sol"][-1].numpy().tolist()
    )
    _dset["ode_uncond_trajectory_end"] = (
        samples["output_uncond"]["sol"][-1].numpy().tolist()
    )

    _dset["y"] = convert_to_np(samples["y"])

    return Dataset.from_dict(_dset, split=split)


def wandb_format(embeddings, x_ds):
    X_collection = []
    recon_collection = []
    z_collection = []

    if x_ds["type"] == "image":
        for X in embeddings["X"]:
            X_collection.append(wandb.Image(torch.tensor(X)))

        if "recon" in embeddings.column_names:
            for recon in embeddings["recon"]:
                recon_collection.append(wandb.Image(torch.tensor(recon)))

    if "z" in embeddings.column_names:
        for z in embeddings["z"]:
            z_collection.append(torch.tensor(z))

    embeddings = embeddings.to_pandas()
    if X_collection:
        embeddings["X"] = X_collection
    if recon_collection:
        embeddings["recon"] = recon_collection
    if z_collection:
        embeddings["z"] = z_collection
    return embeddings


def get_ckpt_files(ckpt_dir):
    if not ckpt_dir.exists():
        msg = f"Checkpoint directory {ckpt_dir} does not exist."
        raise FileNotFoundError(msg)

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    # get idx, job_id and job_num from the ckpt_files
    # and sort by job_num

    if len(ckpt_files) == 0:
        msg = f"No checkpoint files found in {ckpt_dir}."
        raise FileNotFoundError(msg)
    return ckpt_files


def create_lightning_loader(cfg):
    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_files = get_ckpt_files(ckpt_dir)
    current_job_num = int(HydraConfig.get().job.num)

    job_ids = []
    job_nums = []

    for ckpt_file in ckpt_files:
        if len(ckpt_file.stem.split("_")) > 1:
            job_id, job_num = ckpt_file.stem.split("_")

        else:
            job_id = ckpt_file.stem
            job_num = 0  # for single job runs.

        if "-v" in job_num:
            # skip versioned files
            continue

        job_ids.append(job_id)
        job_nums.append(int(job_num))

    job_ids = set(job_ids)

    if len(job_ids) > 1:
        msg = (
            f"Multiple job ids found in {ckpt_dir}. "
            "Only one job id should exist for this run."
        )
        raise ValueError(msg)

    try:
        ckpt_idx = job_nums.index(current_job_num)
    except Exception as e:
        msg = f"Job number {current_job_num} not found in {ckpt_dir}. {e}"
        raise ValueError(msg) from e

    ckpt_path = Path(ckpt_files[ckpt_idx])
    OmegaConf.update(cfg, "lightning_loader.ckpt_path", str(ckpt_path))

    return (
        cfg,
        ckpt_path.stem,
        int(HydraConfig.get().job.id),
    )


def create_info_file(cfg, job_id, job_num, dir_path):
    job_num = int(HydraConfig.get().job.num)
    sweep = list(cfg.sweeps.keys())[job_num]
    sweep_info = dict(cfg.sweeps[sweep])
    sweep_info["job_id"] = job_id
    sweep_info["job_num"] = job_num

    if "vae_ckpt_path" in cfg.lightning_loader:
        sweep_info["vae_ckpt_path"] = cfg.lightning_loader.vae_ckpt_path

    if "ckpt_path" in cfg.lightning_loader:
        sweep_info["ckpt_path"] = cfg.lightning_loader.ckpt_path

    output_path = os.path.join(dir_path, f"{job_id}_info.yaml")
    OmegaConf.save(config=sweep_info, f=output_path)

    return


def run_test(test, data):
    model = hydra.utils.instantiate(test.model)
    X = np.array(data[test.variables.X])
    y = np.array(data[test.variables.y]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test.test_split, random_state=test.random_state
    )
    model.fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)


def run_tests(df, tests, embed_folder):
    _scores = {}
    for test in tests.values():
        train_score, test_score = run_test(test, df)
        test["train_score"] = train_score
        test["test_score"] = test_score
        test["embed_folder"] = embed_folder
    return tests


def run_fm_test(test, data):
    model = hydra.utils.instantiate(test.model)
    # X = np.array(data[test.variables.X])
    ode_traj = np.array(data["ode_trajectory"]).transpose(1, 0, 2)
    y = np.array(data[test.variables.y]).ravel()
    train_scores = []
    test_scores = []

    for idx in range(ode_traj.shape[0]):
        X = ode_traj[idx, :, :]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test.test_split, random_state=test.random_state
        )
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    return train_scores, test_scores


def run_fm_tests(df, tests, embed_folder):
    _scores = {}
    for test in tests.values():
        train_score, test_score = run_fm_test(test, df)
        test["train_score"] = train_score
        test["test_score"] = test_score
        test["embed_folder"] = embed_folder
    return tests


def get_embedding_folder(cfg, job_num):
    embed_dir = Path(cfg.paths.embed_dir)
    embed_folders = list(embed_dir.iterdir())
    current_job_num = int(HydraConfig.get().job.num)
    job_ids = []
    job_nums = []

    for embed_folder in embed_folders:
        fp_components = embed_folder.name.split("_")
        job_id = fp_components[0]
        job_num = fp_components[1]

        job_ids.append(job_id)
        job_nums.append(int(job_num))

    job_ids = set(job_ids)
    if len(job_ids) > 1:
        msg = (
            f"Multiple job ids found in {embed_dir}. "
            "Only one job id should exist for this run."
        )
        raise ValueError(msg)
    try:
        ckpt_idx = job_nums.index(current_job_num)
    except Exception as e:
        msg = f"Job number {current_job_num} not found in {embed_dir}. {e}"
        raise ValueError(msg) from e

    return embed_folders[ckpt_idx]


def save_tests(tests, cfg, embed_folder, split):
    stat_test_dir = Path(
        cfg.paths.experiment_path + "/stat_tests/" + embed_folder + f"/{split}"
    )

    if not stat_test_dir.exists():
        print(f"Creating stat test directory {stat_test_dir}")
        stat_test_dir.mkdir(parents=True)

    test_names = tests.keys()

    test_summary = [
        {
            "model_id": embed_folder,
            "X": output["variables"]["X"],
            "y": output["variables"]["y"],
            "train_score": output["train_score"],
            "test_score": output["test_score"],
            "name": output["name"],
            "drop_variables": str(cfg.data.y_catalog.drop_variables),
        }
        for output in tests.values()
    ]

    df = pd.DataFrame(test_summary)
    df["test_key"] = test_names

    df.to_csv(stat_test_dir / "tests.csv", index=False)
    return df
