# Project details
Authors: Andrei Shelest, Zofia Bracha

University of Warsaw  

Project is based on a paper found here: https://www.sciencedirect.com/science/article/abs/pii/S0957417419300995  
title: "Combining Principal Component Analysis, Discrete Wavelet Transform  
and XGBoost to trade in the financial markets"   
Authors: João Nobre, Rui Ferreira Neves  

Main parts include
  1. PCA
  2. Discrete Wavelet Transform
  3. XGBoost with hyperparameter tunning using MOOGA.
  4. Applying generated signals in trading
  5. Evaluation

# Setting Up the Environment

The project was created in Python with Anaconda, therefore Anaconda is the preferred way to run the project.
You can restore the conda environment from the *environment.yml* file included in the project:

- The file was created by running ```conda env export``` command
- To create new conda environment with *environment.yml* file, please run ```conda env create --name envname --file=environment.yml```

# How to Run the Project

There are several essential scripts to be run in order to download and prepare the data for the modelling. In order to facilitate the process, the first four steps can be performed one by one with ```./src/data_generation.py``` script.

- Firstly, execute ```./src/data/tickers_loader.py``` module to download ticker data from Yahoo Finance.
- Next, generate target feature (Buy or Sell) by executing ```./src/data/target_feature.py```.
- Then execute ```./src/features/technical_analysis.py``` in order to generate technical indicators.
- After that, execute ```./src/models/train_test_split.py``` so that the appropriate split into train, validation and test datasets is created.
- The actual modelling is performed in ```./src/modelling.py```.
- Strategies are executed by ```./src/strategy_execution.py```.
- And, finally, in the notebook ```1.0-strategy_performance.ipynb``` you can plot the performance of strategies.

For each step configuration please refer to ```./project_config.json``` file, where you can see the paths where generated files are downloaded or stored, as well as configuration of algorithms.

