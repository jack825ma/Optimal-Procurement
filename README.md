# Learning Multi-Item Procurement Auctions

This is the official implementation of our neural-netowrk based approach for optimal procurement auctino design. It contains all the necessary code to reproduce our experiments and baselines, along with the pre-trained models that we used in obtaining our results.
## Scripts

### Running Experiments

To run experiments using ProcFormer, use the command:

`python main.py --setting=run_configs/setting`

Do not add `.py` to the end of the command. All settings are located in `run_configs` folder.

### Running Baselines

We provide scripts for running baselines (Reverse Vickrey and Itemwise Reverse Vickrey). Use the command:

`python main.py --setting=mechanisms/mechanism.py`

## Trained Models
We provide the pre-trained models we used in our experiments. Models are run for 10000 iterations with seller costs drawn from $U_{[0,1]}$. The corresponding model for a setting can be found in `runs/setting/model_iter` where `iter` is the desired iteration. Models for settings 2x1, 3x1, and 2x2 are TBA.
