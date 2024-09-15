import wandb
import os


def init_wandb_run(config, job_type, mode="online", log_code=True, **kwargs):
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity="social-foundations",
        save_code=True,
        job_type=job_type,
        config=config,
        mode=mode,
        **kwargs
    )
    if log_code:
        run.log_code()  # save code
