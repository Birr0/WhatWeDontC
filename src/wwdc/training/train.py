import logging
import os

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from wwdc.training.modules import track_weights

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="experiment/RGBMNIST_VAE/train",
)
def main(cfg):
    """The main training function."""
    seed_everything(cfg.seed, workers=True)

    try:
        """#Add movement to scratch here
        try:
            #breaks with error code 11
            #believe it is clashes with I/O on parallel jobs
            subprocess.run(
                [
                    "rsync",
                    "-avz",
                    cfg.meta.data_path,
                    cfg.meta.scratch_data_path,
                ],
                capture_output=True,
                check=True
            )
            msg = f"Data in {cfg.meta.data_path} moved to scratch."
            log.info(msg)

        except Exception as e:
            msg = f"Error in moving the data to scratch: {e}."
            log.error(msg)
            raise Exception(msg) from e"""

        data = hydra.utils.instantiate(cfg.data.loader)
        data.setup()
        train_dataloader = data.train_dataloader()
        val_dataloader = data.val_dataloader()
        test_dataloader = data.test_dataloader()

        log.info("Data loaders initialized.")
    except Exception as e:
        msg = f"Error in instantiating the data loader: {e}."
        log.error(msg)
        raise Exception(msg) from e

    """try:
        for key, value in cfg.paths.items():
            if key not in ["scratch_data_dir", "data_dir", "embed_dir"]:
                print(key, value)
                OmegaConf.update(
                    cfg,
                    f"paths.{key}",
                    f"{cfg.paths.scratch_data_dir}{value}"
                )
                print(cfg.paths)
    except Exception as e:
        msg = f"Error in printing the paths: {e}."
        log.error(msg)
        raise Exception(msg) from e"""

    try:
        if "run_id" in cfg:
            msg = f"Resuming training from run_id: {cfg.run_id}."
            log.info(msg)
            if os.path.exists(f"{cfg.paths.experiment_path}/ckpts/{cfg.run_id}.ckpt"):
                OmegaConf.update(
                    cfg,
                    "lightning_loader.vae_ckpt_path",
                    f"{cfg.paths.experiment_path}/ckpts/{cfg.run_id}.ckpt",
                )
        lightning_loader = hydra.utils.instantiate(cfg.lightning_loader)
        log.info("Lightning loader initialized.")
    except Exception as e:
        msg = f"Error in instantiating the lightning loader: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer = hydra.utils.instantiate(cfg.trainer)
        log.info("Trainer initialized.")
    except Exception as e:
        msg = f"Error in instantiating the trainer: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        if "run_id" in cfg:
            OmegaConf.update(cfg, "logger.wandb.id", f"{cfg.run_id}")
        wandb = hydra.utils.instantiate(cfg.logger.wandb)
        if "run_id" in cfg:
            wandb._wandb_init["resume"] = "must"
        wandb.log_hyperparams(cfg)
        log.info("Wandb logger initialized.")
    except Exception as e:
        msg = f"Error in instantiating the wandb logger: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer.fit(
            model=lightning_loader,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.trainer_ckpt_path if cfg.trainer_ckpt_path else None,
        )
        log.info("Model fitting completed.")

    except Exception as e:
        msg = f"Failed to train the model for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer.test(
            model=lightning_loader,
            dataloaders=test_dataloader,
        )
        log.info("Model testing completed.")

    except Exception as e:
        msg = f"Failed to test the model for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        job_id = HydraConfig.get().job.id
        track_weights(cfg, job_id)
        log.info("Weights tracking on local machine completed.")
    except Exception as e:
        msg = f"Failed to track the weights for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    """# move experiment path back to original directory.
    experiment_path = "/" +
    os.path.relpath(cfg.paths.experiment_path, start=cfg.paths.scratch_data_dir)
    experiment_path = str(pathlib.Path(experiment_path).parent)

    subprocess.run(
        [
            "rsync",
            "-avz",
            cfg.paths.experiment_path,
            experiment_path,
        ]
    )
    OmegaConf.update(
        cfg,
        "paths.experiment_path",
        experiment_path
    )"""

    return


if __name__ == "__main__":
    # Enable multirun by default for SLURM launcher
    GlobalHydra.instance().clear()
    main()

    """job_num = int(HydraConfig.get().job.num)
    rerun = (cfg.rerun_job_num is not None) #and (cfg.failed_job_id is not None)
    job_rerun = (job_num in cfg.rerun_job_num)

    if rerun and not job_rerun:
        print(f"Job {cfg.failed_job_id}_{job_num} has already been run,
        skipping this job.")
        return

    elif rerun and job_rerun:
        job_id = cfg.failed_job_id
        print(f"Re-running job {cfg.failed_job_id}_{HydraConfig.get().job.num}.")

    else:
        job_id = HydraConfig.get().job.id"""
 