# Out-of-Vocabulary Entities in Link Prediction Datasets
This open-source project contains the implementation of our analysis on the impact of out-of-vocabulary entities 
in link prediction datasets.

# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/OOV-In-Link-Prediction
unzip KGs
# Create anaconda virtual enviroment
conda env create -f environment.yml
```

# Describe OOV entities
```
python describe_out_of_vocab_entity.py
python descriptive_statistics.py
```

# Reproduce experiments
```
git clone https://github.com/uma-pi1/kge.git
cd kge
pip install -e .
# download and preprocess datasets
cd data
sh download_all.sh
cd ..
cd ..
# Create pretrained_models/FB15K-237
# Create pretrained_models/WN18RR
# Download pretrained models from LibKGE.
# To reproduce link prediction results on WN18RR and FB15K-237
python link_prediction_wn18rr.py
python link_prediction_fb15k237.py
# To reproduce link prediction results on WN18RR* and FB15K-237*
python link_prediction_wn18rr_star.py
python link_prediction_fb15k237_star.py
```

# Acknowledgement
In our experiments, we used [LibKGE](https://github.com/uma-pi1/kge). We would like to thank for the readable codebase