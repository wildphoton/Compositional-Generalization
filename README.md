# CompGen
Compositional Generalization in Unsupervised Representation Learning: From Disentanglement to Emergent Language

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

## Run Experiments
Set configuations in run_{model}.py scripts first. 

Train and evaluate models by
```
python run_vae.py -g 0 -sk
python run_tcvae.py -g 0 -sk 
python run_el.py -g 0 -sk 

```
Add `-gbt` to use GBT read models for evaluation.

Measure compositional metrics on trained models" 
```
python run_el.py -g 0 --compmetric --notrain
python run_tcvae.py -g 0 --compmetric --notrain
python run_el.py -g 0 --compmetric --notrain
```

Add `--nowb` to disable wandb logger.
