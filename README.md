## Kaggle Competition Solution (6th)
# Google - Fast or Slow? Predict AI Model Runtime 
https://www.kaggle.com/competitions/predict-ai-model-runtime/

For discussion, please refer to:  
https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456084


## 1. Hardware  
- GPU: 2x Nvidia Quadro RTX 8000, each with VRAM 48 GB
- CPU: Intel® Xeon(R) Gold 6240 CPU @ 2.60GHz, 72 cores
- Memory: 376 GB RAM

## 2. OS 
- ubuntu 18.04.5 LTS


## 3. Set Up Environment
- Install Python >=3.10.9
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── solution
    ├── src 
    ├── results
    ├── data
    |   ├── predict-ai-model-runtime
    |       ├── sample_submission.csv
    │       ├── npz_all
    │            ├── npz
    │                 ├── layout 
    │                 │     ├── nlp
    │                 │     │    ├── default : train/valid/test
    │                 │     │    ├── random : train/valid/test
    │                 │     ├── xla
    │                 │          ├── default : train/valid/test
    │                 │          ├── random : train/valid/test
    |                 ├── tile
    |                       ├── xla : train/valid/test      
    ├── LICENSE 
    ├── README.md 
```

- The dataset "predict-ai-model-runtime" can be downloaded from Kaggle:  
https://www.kaggle.com/competitions/predict-ai-model-runtime/data


## 4. Training the model

### Warning !!! training output will be overwritten to the "solution/results" folder
Please run the following python scripts to output the model files

``` 
>> python src/1a_run_res_graphsage4_layout.py
output model:
- results/final-01/model/4x-graphsage-pair2/layout/nlp-default/checkpoint/swa.pth
- results/final-01/model/4x-graphsage-pair2/layout/nlp-random/checkpoint/swa.pth
- results/final-01/model/4x-graphsage-pair2/layout/xla-default/checkpoint/swa.pth
- results/final-01/model/4x-graphsage-pair2/layout/xla-random/checkpoint/swa.pth

>> python src/1b_run_res_gin4_layout.py
output model:
- results/final-01/model/4x-gin-pair2/layout/xla-default/checkpoint/swa.pth

>> python src/2_run_res_gatconv4_tile.py
output model:
- results/final-01/model/4x-gatconv-listmle/tile/xla/checkpoint/00010013.pth
``` 

Local validation results are also output:  
- 4x-graphsage-pair2


|             | opa     | kendall_tau |
|-------------|---------|-------------|
| nlp-default | 0.76969 | 0.53938     |
| nlp-random  | 0.96327 | 0.92654     |
| xla-default | 0.72754 | 0.45508     |
| xla-random  | 0.83563 | 0.67127     |
 
- 4x-gin-pair2  

|             | opa     | kendall_tau |
|-------------|---------|-------------|
| xla-default | 0.72978 | 0.45957     | 

- 2_run_res_gatconv4_tile 

|     | slowndown1 | slowndown5 | slowndown10 |
|-----|------------|------------|-------------|
| xla | 0.89052    | 0.97462    | 0.98351     |


## 5. Submission csv 

Please run the following script:

```
>> python src/3_run_make_kaggle_submission.py
output file:
- results/final-01/submission_06.csv
```

|                   | public lb | private lb |
|-------------------|-----------|------------|
| submission_06.csv | 0.69424   | 0.70549    |


## 6. Reference trained models and validation results
- Reference results can be found in the zip file "final-01.zip". It includes the weight files, train/validation logs.
- Please download from share google drive: [https://drive.google.com/drive/folders/13zgzaB-kl9CnPcXfibCfxnUY5WtHJLbQ?usp=sharing](https://drive.google.com/drive/folders/13zgzaB-kl9CnPcXfibCfxnUY5WtHJLbQ?usp=sharing)
  

## Authors

- https://www.kaggle.com/hengck23

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

"We extend our thanks to HP for providing the Z8-G4 Data Science Workstation, which empowered our deep learning experiments. The high computational power and large GPU memory enabled us to design our models swiftly."
