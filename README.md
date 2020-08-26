Collaborators: don't distribute without permission.

# Presentation
[Slides](https://docs.google.com/presentation/d/1pLQjgdYT3jEJYzpIW5BcekAlCagWzhv8FiPQ3m8OTjI/edit?usp=sharing)

# Resource

* GWAS (180 used in Choobdar et al. 2019 Nature Methods/the DREAM challenge paper) `/cellar/users/f6zheng/Data2/GWAS/4_gwas_datasets`
* GWAS (more based on [EBI-GWAS catalog](https://www.ebi.ac.uk/gwas/docs/file-downloads)) `/cellar/users/ecdymore/GWAS_Context/Gene_by_Trait`

* BioGRID (abstracts associated with interactions), use papers generating >=5 interactions:`/cellar/users/f6zheng/Data2/text_mining/biogrid` 
* STRING (with PPI and coexp scores, etc.) (`/cellar/users/f6zheng/Data/Public_data/stringdb/2019-07-17`)

# Plan

Fig. 1: Data, Method and Workflow  
Fig. 2: Show basic heterogeneous network, with STRING or humanNet; compare to homogeneous network (predict GWAS)  
Fig. 3: Show that interactions can be literature dependent. Some pleiotropic proteins interact with a lot of things but interactions are specific to particular biological topics  
Fig. 4: Use doc vectors as edge embedding to predict GWAS  
Fig. 5: Intepret predictions with attention  

# Workflow

* Use `gwas_2_npy.py` to create network-specific numpy array
* Use `gen_training_data_HAN,py` to generate a pickle file that is similar to the ACM dataset used in original HAN paper `~/.dgl/ACM3025.pkl`
* Put the created `pkl` file in `~/.dgl/ppi`
* Run `python han/main.py --ds [xxx.pkl]`

# Notes

- Aug. 12; code run well. only have a test run (not really meaningful class and edges), don't overintepret, but seemingly class imbalance and feature initialization may be problems. Also need to further understand the code to generate prediction outputs, tune hyperparameters and output attention values.
- Aug. 25: run HAN on all GWAS set, tried a few different class weights. Don't expect the performance will be amazing; need some baseline (network propagation to decide)
