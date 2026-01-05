# CryoPromptSeg

## Requirements

torch 2.0.1,python3.8,mmcv 

## Training

1.Train the SFI_Promter

```
python main.py --config cryoPoint.py --output_dir feature_fusion --use-wandb 
```

2.Train the SAM_Mona

```
python train.py --lr_scheduler 1  
```

## Test

Generate prompts using the trained SFI_Prompt.

```
python predict_prompts.py --config cryoPoint.py --resume checkpoint/feature_fusion/best.pth
```

