import ray
from ray_tpu import RayTpuManager
from ray.job_submission import JobSubmissionClient
from trainer import MaxTextTrainer

import logging
import os
import argparse
import pyconfig
from typing import Sequence, Optional
from absl import app



#### Configurations
# Flags that go into MaxText
MAXTEXT_CONFIG = dict(
    tokenizer_path="assets/tokenizer",
)
# Enables verbose TPU logging.
TPU_VERBOSE_ENV_VARS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
}

# Default env vars that run on all TPU VMs.
MACHINE_ENV_VARS = {
    "ENABLE_PJRT_COMPATIBILITY": "true",
    "TPU_SLICE_BUILDER_DUMP_CHIP_FORCE": "true",
    "TPU_SLICE_BUILDER_DUMP_ICI": "true",
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto",  # Dumps HLOs for debugging
}


def setup_loggers():
  """Sets up loggers for Ray."""
  logging.basicConfig(level=logging.INFO)


def get_job_submission_id() -> str:
  """Returns the Ray job submission ID."""
  c = JobSubmissionClient()
  current_job_id = ray.get_runtime_context().get_job_id()
  jobs = c.list_jobs()
  return [job.submission_id for job in jobs if job.job_id == current_job_id][0]


def main(argv: Sequence[str]):
  ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
  run_name = get_job_submission_id()
  logging.info("Got args: %s", argv)
  logging.info("This run name: %s", run_name)

  tpu_resources = RayTpuManager.get_available_resources()
  num_detected_tpu_types = len(tpu_resources.keys())
  if num_detected_tpu_types == 0:
    logging.error("Did not detect any TPUs in the cluster, check your Ray cluster setup: %s", ray.available_resources())

  tpu_type = list(tpu_resources.keys())[0]
  if num_detected_tpu_types > 1:
    logging.warning(
        "Detected %d TPUs in the cluster. MaxText does not support clusters with multiple TPU pod slices - falling back to using %s",
        num_detected_tpu_types,
        tpu_type,
    )

  logging.info("Running on pod slice type %s.", tpu_type)

  logging.info("Creating Ray actors with multislice.")

  actors = RayTpuManager.remote(
      tpus=tpu_resources[tpu_type], actor_or_fn=MaxTextTrainer, multislice=True, env=MACHINE_ENV_VARS, argv=argv 
  )

  try:
    # Keep initializations separately so we can catch errors.
    logging.info("Initializing actors.")
    ray.get([actor.initialize.remote(run_name) for actor in actors])
  except Exception as e:
    logging.error("Caught error during initializations: %s", e)
    logging.error("Shutting down...")
    ray.shutdown()
    raise e

  logging.info("Initialization complete. Starting MaxText training...")
  total_steps = 50 #int(args.total_steps)
  steps_per_loop = 100 #int(args.steps_per_loop)
  steps = 0

  while steps < total_steps:
    logging.info("Training from step %d to %d.", steps, steps_per_loop)

    try:
      r = ray.get([actor.train.remote(num_steps=steps_per_loop) for actor in actors])
      steps = r[0]
    except Exception as e:
      logging.error("Caught error during training: %s", e)
      logging.error("Shutting down...")
      ray.shutdown()
      raise e

  logging.info("Training complete!")
  ray.shutdown()


if __name__ == "__main__":
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  app.run(main)
