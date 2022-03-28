# sHINGE

**sHINGE** is a schema-aware hyper-relational KG embedding model, which directly learns from both hyper-relational facts and their corresponding schema information in a KG. Please see the details in our paper below:

# How to run the code 

###### Data preprocessing
```
python builddata.py --data_dir <PATH>/<DATASET>/
python builddata.py --data_dir <PATH>/<DATASET>/ --if_permutate True --bin_postfix _permutate
```
###### Train and evaluate model (suggested parameters for both JF17k and Wiki dataset)
check the script `sHINGE/run_all_experiments.sh`

# Python lib versions
Python: 3.7.2

torch: 1.9.1

numpy: 1.18.1

tensorflow-gpu: 2.2.0

# Reference
If you use our code or datasets, please cite:
