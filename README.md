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
    
    
    
    
## Requirements
