# MasterThesis: Data Trustworthiness Assessment for Traffic Condition Participatory Sensing Scenario  

## SUMO  
Part of the code is quoted from the large-scale and high-accuracy traffic simulation scenario for SUMO (Simulator of Urban MObility), version 0.32.0 described in the following paper:

    Juan Jesus Gonzalez-Delicado, Javier Gozalvez, Jesús Mena-Oreja, Miguel Sepulcre, Baldomero Coll-Perales
    "Alicante-Murcia Freeway Scenario: A High-Accuracy and Large-Scale  Traffic Simulation Scenario 
    generated using a Novel Traffic Demand Calibration Method in SUMO",
    IEEE Access, November 2021. DOI: 10.1109/ACCESS.2021.3126269
    https://ieeexplore.ieee.org/abstract/document/9606704
    
### Files
Common files:
* `Alicante-Murcia_MW_LIMPIO_GR_CRECIENTE_beforeApril16.net.xml` SUMO road network file.
* `detectores_Ali-Mur_CREC_bAp16_Ord_567OK_best82_LIGEROS.xml` SUMO detector file for Light Vehicles.
* `detectores_Ali-Mur_CREC_bAp16_Ord_567OK_best82_PESADOS.xml` SUMO detector file for Heavy Vehicles.
* `detectores_Ali-Mur_CREC_bAp16_Ord_567OK_best82_MIX.xml` SUMO detector file for all vehicles.
* `rerouters.xml` SUMO additional file to include rerouters in the road network.
* `vtypes.xml` SUMO additional file to define vehicle types.
* `runner.py` TraCI startup file.

Each 12 hour period is calibrated independently, having its own demand file and configuration file:
* `altRoutesOutput_XXX.cal.xml` SUMO demand file.
* `iteration_XXX.sumocfg` SUMO configuration file.
    
    
    
## Data trustworthiness assessment  

Part of the code is quoted from the DCRNN work described in the following paper:
    ```
    @inproceedings{li2018dcrnn_traffic,
      title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
      author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
      booktitle={International Conference on Learning Representations (ICLR '18)},
      year={2018}
    }
    ```
    
### Requirements
* `scipy>=0.19.0
* `numpy>=1.12.1
* `pandas>=0.19.2
* `pyaml
* `statsmodels
* `tensorflow>=1.3.0

### Data Preparation
Run the following commands to generate train/test/val dataset
```bash
# Create data directories
python dataprep.py
python generate_training_data.py  # dataset used for training DCRNN and DCRNN-NoCov
python generate_testing_data.py  # real-time data  
dataset_attacker.ipynb  # attacker data for user distinction
```

### Graph Construction
```bash
gen_adj_mx.ipynb
```

### Model Training 
```bash
# DCRNN
python dcrnn_train.py --config_filename=dcrnn_train.yaml

# DCRNN-NoCov
python dcrnn_train.py --config_filename=dcrnn_nocov_train.yaml
```

### Run the Pre-trained Model on testing data
```bash
python run_prediction.py --config_filename=pretrained_model/DCRNN/config_23.yaml
python run_prediction.py --config_filename=pretrained_model/DCRNN_NoCov/config_11.yaml

or go to run_prediction.ipynb for GPU

ARIMA.py
```

### Experiments
```bash
Experiments and visualization.ipynb
```
