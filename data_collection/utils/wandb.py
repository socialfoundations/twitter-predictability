import wandb
import os


def init_wandb_run(config, job_type, mode="online", log_code=True):
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity="social-foundations",
        save_code=True,
        job_type=job_type,
        config=config,
        mode=mode,
    )
    if log_code:
        run.log_code()  # save code
