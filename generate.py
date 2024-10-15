
import os
import yaml

import evaluation


def main():

    with open("params_generative.yml", 'r') as f:
        params = yaml.safe_load(f)

    # Remove SLURM_JOBID to prevent ignite assume we are using SLURM to run multiple tasks.
    os.environ.pop("SLURM_JOBID", None)

    evaluation.ms_eval(0, params)

    
if __name__ == "__main__":
    main()
