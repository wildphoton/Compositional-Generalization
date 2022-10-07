# CompGen
This repository contains the official code for the paper: "Compositional Generalization in Unsupervised Representation Learning: From Disentanglement to Emergent Language" (NeurIPS2022). 

This work propose a new protocol of evaluating compositional generalization of learned representations. Our protocol focus on whether or not it is easy to train a simple
model for downstream tasks on top of the learned representation that generalizes to new combinations of compositional factors. We systematically studied $\beta$-VAE, $\beta$-TCVAE and emergent language autoencoders.



## Dependencies
```
torch == 1.8.1
torchvision == 0.9.1
pytorch-lightning == 1.5.8
wandb == 0.12.10
scikit-learn == 0.22
disentanglement-lib == 1.5
tensorflow == 1.15.0
tensorflow == 1.15.0             
tensorflow-datasets == 4.2.0              
tensorflow-estimator ==1.15.1             
tensorflow-hub == 0.4.0              
tensorflow-metadata == 0.30.0             
tensorflow-probability == 0.6.0  
```
## Data
Two public available datasets [dSprites](https://github.com/deepmind/dsprites-dataset) and [MPI3D](https://github.com/rr-learning/disentanglement_dataset) are used in our work.

## Run Experiments
- Set configuations in ```.yaml``` files under ```scripts/configs``` or directly overload arguments in experimental scripts e.g. ```run_{MODEL_NAME}.py```. 

- Run Pretrain and finetune by
```
python run_vae.py -g 0 -ft
python run_tcvae.py -g 0 -ft 
python run_el.py -g 0 -ft
```
- By default, linear readout models are used. Add `-gbt` to use GBT read models for evaluation.

- If the a pretraining model with the same config exists, it will skip the pretraining use the previous saved model unless adding ```--overwrite``` tag.

- Evaluate the disentanglement/compositionality metric of pretrained models
```
python run_{MODEL_NAME}.py -g 0 --compmetric 
```

Add `--nowb` to disable wandb logger.
