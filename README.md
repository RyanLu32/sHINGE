# sHINGE (Schema-aware Hyper-relational Knowledge Graph Embedding)

**sHINGE** is a schema-aware hyper-relational KG embedding model, which directly learns from both hyper-relational facts and their corresponding schema information in a KG. sHINGE captures not only the primary structural information of the KG encoded in the triplets and their associated key-value pairs, but also the schema information encoded by entity-typed triplets and their associated entity-typed key-value pairs. Please see the details in our paper below:

# How to run the code 

###### Data preprocessing
```
python builddata.py --data_dir <PATH>/<DATASET>/
python builddata.py --data_dir <PATH>/<DATASET>/ --if_permutate True --bin_postfix _permutate
```
###### Train and evaluate model (suggested parameters for both JF17k and Wiki dataset)
check the script `sHINGE/run_all_experiments.sh`

###### Parameter setting:
In `main_hinge.py`, you can set:
`--indir`: input file directory

`--epochs`: number of training epochs

`--batchsize`: batch size of training set

`--embsize`: embedding size

`--learningrate`: learning rate

`--outdir`: where to store HINGE model

`--load`: load a pre-trained HINGE model and evaluate

`--num_negative_samples`: number of negative samples

`--gpu_ids`: gpu to be used for train and test the model

`--num_filters`: number of filters used in the CNN


# Python lib versions
Python: 3.7.2

torch: 1.9.1

numpy: 1.18.1

tensorflow-gpu: 2.2.0

# Reference
If you use our code or datasets, please cite:
