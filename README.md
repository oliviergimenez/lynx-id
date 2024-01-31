# DP-SCR_Identify-and-estimate-density-lynx-population

## Running code outside the `lynx_id` module

To run code outside `lynx_id` folder that uses this module (for example, files in the `test` folder), you need to **install the project locally in editable mode**.  
On Jean-Zay, run a command similar to this one at the root of the project: 
- `pip install --editable . --user --no-cache-dir`  

You can then run code that does `from lynx_id. ... import ...`.

## Running MegaDetector in Python code

**Currently not used** because generating bounding boxes on the fly is too time-consuming.

- Requesting computing resources via SLURM, for example with: `srun --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --nodes=1 --hint=nomultithread --qos=qos_gpu-dev --account=ads@v100 --cpus-per-task 8 --pty bash
`
- `source setup_env.sh`  
- You can then use the `batch_detection` function, for example, by executing the file `test_megadetector.py`.

**Note**: this does not work for notebooks.
