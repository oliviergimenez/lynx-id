# DP-SCR_Identify-and-estimate-density-lynx-population

## Running MegaDetector in Python code

- Requesting computing resources via SLURM, for example with: `srun --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --nodes=1 --hint=nomultithread --qos=qos_gpu-dev --account=ads@v100 --cpus-per-task 8 --pty bash
`
- `source setup_env.sh`  
- You can then use the `batch_detection` function, for example, by executing the file `test_megadetector.py`.

**Note**: this does not work for notebooks.
