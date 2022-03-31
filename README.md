# sHINGE (Schema-aware Hyper-relational Knowledge Graph Embedding)

**sHINGE** is a schema-aware hyper-relational KG embedding model, which directly learns from both hyper-relational facts and their corresponding schema information in a KG. sHINGE captures not only the primary structural information of the KG encoded in the triplets and their associated key-value pairs, but also the schema information encoded by entity-typed triplets and their associated entity-typed key-value pairs.

# How to run the code 

###### Data preprocessing
```
python builddata.py --data_dir <PATH>/<DATASET>/
python builddata.py --data_dir <PATH>/<DATASET>/ --if_permutate True --bin_postfix _permutate
```
###### Train and evaluate model (suggested parameters for both JF17k and Wiki dataset)
check the script `sHINGE/run_all_experiments.sh`

