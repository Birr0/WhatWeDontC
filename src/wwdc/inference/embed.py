import logging
import os

import hydra
from datasets import concatenate_datasets
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import seed_everything

from wwdc.inference.modules import (
    create_lightning_loader,
    create_timestep_embeddings,
    wandb_format,
    # create_embeddings,
    # need to add option for this for
    # non-timestep embeddings
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="experiment/RGBMNIST_FLOW/embed",
)
def main(cfg):
    """The main inference function."""
    seed_everything(cfg.seed, workers=True)

    try:
        data = hydra.utils.instantiate(cfg.data.loader)
        data.setup()
        split_dataloaders = {
            "test": {"loader": data.test_dataloader, "call": False},
            "train": {"loader": data.train_dataloader, "call": False},
            "val": {"loader": data.val_dataloader, "call": False},
        }

        for split in cfg.splits:
            split_dataloaders[split]["dataloader"] = split_dataloaders[split][
                "loader"
            ]()
            split_dataloaders[split]["call"] = True

        msg = f"Data loaders instantiated: {split_dataloaders!s}"
        log.info(msg)

    except Exception as e:
        msg = f"Error instantiating data loaders: {e}"
        log.error(msg)
        raise e

    try:
        cfg, model_id, job_id = create_lightning_loader(cfg)
        print(f"cfg: {cfg}. Model ID: {model_id}. Job ID: {job_id}")

        lightning_loader = hydra.utils.instantiate(cfg.lightning_loader)
        msg = "Lightning loader instantiated."
        log.info(msg)
    except Exception as e:
        msg = f"Error instantiating lightning loader: {e}"
        log.error(msg)
        raise e

    try:
        cfg.logger.wandb.tags.append(str(model_id))
        cfg.logger.wandb.id = model_id
        cfg.logger.wandb.name = f"{model_id}_{cfg.meta.experiment_name}"
        wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)
        msg = "Wandb logger instantiated."
        log.info(msg)

    except Exception as e:
        msg = f"Error instantiating wandb logger: {e}"
        log.error(msg)
        raise e

    for name, split in split_dataloaders.items():
        embeddings = []
        if split["call"]:
            for idx, batch in enumerate(split["dataloader"]):
                if (cfg.batch_limit is not None) and (idx >= cfg.batch_limit):
                    break
                try:
                    predictions = lightning_loader.predict_step(batch)

                    # Now need to save these correctly...
                    # save log probs to etc...
                    embeddings.append(
                        # create_embeddings(predictions, name) # _timestep
                        create_timestep_embeddings(predictions, name)
                    )  # create_embeddings
                except Exception as e:
                    msg = f"Error creating embeddings: {e}"
                    log.error(msg)
                    raise e

            try:
                if not os.path.exists(cfg.paths.embed_dir):
                    os.makedirs(cfg.paths.embed_dir)

                embed_dir = cfg.paths.embed_dir + f"/{model_id}/{name}"
                embeddings = concatenate_datasets(embeddings)
                embeddings.save_to_disk(embed_dir)

            except Exception as e:
                msg = f"Error saving embeddings: {e}"
                log.error(msg)
                raise e

            try:
                print(f"wandb_logger: {wandb_logger}")
                print(
                    f"wandb_logger._wandb_init['mode']:\
                        {wandb_logger._wandb_init['mode']}"
                )
                if wandb_logger._wandb_init["mode"] == "online":
                    wandb_logger.log_table(
                        key=f"{model_id}_{name}_embeddings",
                        dataframe=wandb_format(embeddings, cfg.data.x_ds),
                    )
                    print("Logged embeddings to wandb.")
                    msg = "Logged embeddings to wandb."
                    log.info(msg)
            except Exception as e:
                msg = f"Error logging embeddings to wandb: {e}"
                log.error(msg)
                raise e

    log.info("Inference complete.")
    return


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
