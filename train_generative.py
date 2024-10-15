
import os
import yaml

import generative

import torch
import ignite.distributed as idist

# os.environ['NCCL_P2P_DISABLE'] = str(1)  
# torch.backends.cudnn.enabled = False


def main():

    with open("params_generative.yml", 'r') as f:
        params = yaml.safe_load(f)

    # Remove SLURM_JOBID to prevent ignite assume we are using SLURM to run multiple tasks.
    os.environ.pop("SLURM_JOBID", None)

    if params['distributed']:
        # Run distributed
        with idist.Parallel(
                backend="nccl",
                nproc_per_node=torch.cuda.device_count(),
                master_addr="127.0.0.1",
                master_port=27182) as parallel:
            parallel.run(generative.train_ddpm, params)
    else:
        generative.train_ddpm(0, params)


if __name__ == "__main__":
    main()
