# Hyperparameter tuning

All public tuners use Optuna's TPE sampler and MedianPruner. A trial trains on
the training split, selects and early-stops on validation, and never evaluates
the test split. The selected configuration is then reinitialized for final
testing. The default budget is 200 trials.

Shared neural-network space:

| Parameter | Space |
|---|---|
| hidden size | `{64,128,256,512}` (subject to model constraints) |
| depth | integer `[2,10]` |
| dropout | continuous `[0,0.5]` |
| activation | `{ReLU,PReLU,SiLU/Swish}` |
| normalization | `{none,batch_norm,layer_norm}` |
| readout | `{mean,max,sum,attn}` when supported |
| residual / virtual node | `{false,true}` when supported |
| learning rate | log-uniform `[1e-4,1e-2]` |
| weight decay | `{0,1e-6,1e-5,1e-4,5e-4,1e-3}` |
| scheduler | `{none,ReduceLROnPlateau,CosineRestart,StepLR}` |

Model-specific spaces are in the correspondingly named `*_tuning.py` module.
They cover FFN/RWSE for Plus models; PNA towers; GraphGPS attention and PE;
pooling depth/ratio for EdgePool, TopK, and GMT; HMN colouring; AK/ECS/I2GNN
subgraphs; MO/HGP/VIB structure learning; and RGC/MVG/GCA contrastive choices.

```bash
python -m tuning.run --dataset NCI1 --model GINE --n_trials 200 --gpu 0
```
