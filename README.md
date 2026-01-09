# CryoPromptSeg

## Requirements

torch 2.0.1,python3.8,mmcv 

Download **`SAM model checkpoint:`** **[ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)** and put it at /SAM_Mona/pretrained/

1. **Clone project**

   ```
   git clone https://github.com/347251369/CryoPromptSeg.git
   cd CryoPromptSeg
   ```

2. **Create conda environment**

   ```
   conda create -n CryoPromptSeg python=3.8 -y
   conda activate CryoPromptSeg
   pip install -r requirements.txt
   ```

## Dataset

You can download datasets from [CryoPPP](http://calla.rnet.missouri.edu/cryoppp).

**Training Data Statistics:**

| **EMPIAR ID** | **Type of Protein**     | **Image Size** | **Training Images** | **Validation Images** |
| ------------- | ----------------------- | -------------- | ------------------- | --------------------- |
| 10005         | TRPV1 Transport Protein | (3710, 3710)   | 23                  | 6                     |
| 10059         | TRPV1 Transport Protein | (3838, 3710)   | 232                 | 59                    |
| 10075         | Bacteriophage MS2       | (4096, 4096)   | 239                 | 60                    |
| 10077         | Ribosome (70S)          | (4096, 4096)   | 240                 | 60                    |
| 10096         | Viral Protein           | (3838, 3710)   | 240                 | 60                    |
| 10289         | Transport Protein       | (3710, 3838)   | 240                 | 60                    |
| 10406         | Ribosome (70S)          | (3838, 3710)   | 191                 | 48                    |
| 10444         | Membrane Protein        | (5760, 4092)   | 236                 | 60                    |
| 10590         | TRPV1 Transport Protein | (3710, 3838)   | 236                 | 60                    |
| 10737         | Membrane Protein        | (5760, 4092)   | 233                 | 59                    |
| 10760         | Membrane Protein        | (3838, 3710)   | 240                 | 60                    |
| 10816         | Transport Protein       | (7676, 7420)   | 240                 | 60                    |
| 10852         | Signaling Protein       | (5760, 4092)   | 274                 | 69                    |
| 11051         | Transcription/DNA/RNA   | (3838, 3710)   | 240                 | 60                    |
| 11183         | Signaling Protein       | (5760, 4092)   | 240                 | 60                    |
| Total         |                         |                | 3344                | 881                   |

**Test Data Statistics:**

| **EMPIAR ID** | **Type of Protein** | **Image Size** | **Number of Images** |
| ------------- | ------------------- | -------------- | -------------------- |
| 10017         | β -galactosidase    | (4096, 4096)   | 84                   |
| 10028         | Ribosome (80S)      | (4096, 4096)   | 300                  |
| 10081         | Transport Protein   | (3710, 3838)   | 300                  |
| 10093         | Membrane Protein    | (3838, 3710)   | 295                  |
| 10345         | Signaling Protein   | (3838, 3710)   | 295                  |
| 10532         | Viral Protein       | (4096, 4096)   | 300                  |
| 11056         | Transport Protein   | (5760, 4092)   | 305                  |
| Total         |                     |                | 1,879                |

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

