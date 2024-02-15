# DP-SCR_Identify-and-estimate-density-lynx-population

## Running code outside the `lynx_id` module

To run code outside `lynx_id` folder that uses this module (for example, files in the `test` folder), you need to **install the project locally in editable mode**.  
On Jean-Zay, run a command similar to this one at the root of the project: 
- `pip install --editable . --user --no-cache-dir`  

You can then run code that does `from lynx_id. ... import ...`.



## Preprocess

To prepare the data, perform the following instructions in the following order:
- `./lynx_id/preprocess/extract_archives.sh`
- `./lynx_id/preprocess/rename.sh`
- Running the notebook `./lynx_id/preprocess/preprocess_dataset.ipynb` with a V100 GPU
