import argparse, os
import pandas as pd
import numpy as np

def gwas_2_genelist(df, nodes):
    genes = []
    for _, row in df.iterrows():
        gene=row[0]
        if ',' in gene:
            gene = [x.strip() for x in gene.split(',')]
        elif ' - ' in gene:
            gene = [x.strip() for x in gene.split(' - ')]
        else:
            gene = [gene]
        genes.extend(gene)
    genes = sorted(set(genes).intersection(nodes))
    return genes

par = argparse.ArgumentParser()
par.add_argument('--node', required=True, help='a file of all node names in the network')
par.add_argument('--gwas_list', required=True)
par.add_argument('--nmin', default=100, type=int)
par.add_argument('--nmax', default=1000, type=int)
par.add_argument('--out', required=True)
args = par.parse_args()

df_doc = pd.read_csv(args.gwas_list, index_col=0)
df_s = df_doc.loc[(df_doc['Number of Alleles'] < args.nmax) & (df_doc['Number of Alleles'] > args.nmin), :]
df_s.reset_index(inplace=True)

nodes = [l.strip() for l in open(args.node).readlines()]
node_dict = {x:i for i,x in enumerate(nodes)}

mat = np.zeros((len(nodes), df_s.shape[0]))
for i, row in df_s.iterrows():
    df_gene = pd.read_csv(os.path.dirname(args.gwas_list) + '/{}.tsv'.format(row['File Index']), sep='\t', header=None)
    genes = gwas_2_genelist(df_gene, nodes)
    genes_idx = np.array([node_dict[g] for g in genes])
    mat[genes_idx, i] = 1

np.save(args.out, mat)

