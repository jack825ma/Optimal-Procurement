# Learning Multi-Item Procurement Auctions

This is the official implementation of our neural-network based approach for optimal procurement auction design. It contains all the necessary code to reproduce our experiments and baselines, along with the pre-trained models that we used in obtaining our results.
## Scripts

### Running Experiments

To run experiments using ProcFormer, use the command:

`python main.py --setting=run_configs/setting`

Do not add `.py` to the end of the command. All settings are located in `run_configs` folder.

### Running Baselines

We provide scripts for running baselines (Reverse Vickrey and Itemwise Reverse Vickrey). Use the command:

`python main.py --setting=mechanisms/{mechanism}.py` where `mechanism` is the desired baseline.

## Pre-Trained Models
We provide the pre-trained models we used in our experiments. Models are run for 10000 iterations with seller costs drawn from $U_{[0,1]}$. The corresponding model for a setting can be found in `runs/setting/model_{iter}` where `iter` is the number of iterations the model is trained for.
