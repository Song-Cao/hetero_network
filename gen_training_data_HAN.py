import numpy as np
import pandas as pd
import argparse, sys
import pickle
from scipy.sparse import *

par = argparse.ArgumentParser()
par.add_argument('--nodelist', required=True, help='a file with a list of nodes, the union of nodes in each edge-type network; this file also can optionally followed by initial node features')
par.add_argument('--nodelabel', required=True, help='a numpy matrix')
par.add_argument('--geneset_id', type=int, required=True, help='use to select a column in the numpy matirx')
par.add_argument('--nets', nargs='+', help='a list of files for each edge-type network')
par.add_argument('--enames', required = True, help='a list of edge-type names')
par.add_argument('--split', help='an optional list of splitting nodes into train/val/test; if not provided, produce randomly on-fly')
# maybe need to specify a dimension value
par.add_argument('--dim', default=32, type=int)
par.add_argument('--out', help='output pickle file')
args = par.parse_args()

df_nodes = pd.read_csv(args.nodelist, sep='\t', header=None)
nodes = df_nodes[0].tolist()
node_dict = {nodes[i]:i for i in range(len(nodes))}
node_labels = np.load(args.nodelabel)[:, args.geneset_id].astype(int)
assert df_nodes.shape[0] == len(node_labels)

if df_nodes.shape[1] > 1:
    try:
        features = np.array(df_nodes.iloc[:, 1:]).astype(float)
    except:
        sys.exit("non-numerical features")
else:
    features = np.random.random((len(nodes), args.dim)) # TODO: thinking about initialization later
features = csr_matrix(features)
# ACM by default is sparse; probably make sense to create dense here

enames = [l.strip() for l in open(args.enames).readlines()]
assert len(enames) == len(args.nets)

mat_label = coo_matrix((np.ones(df_nodes.shape[0],), (np.arange(df_nodes.shape[0]),node_labels)))
mat_label = mat_label.tocsr()

# create train, val, test
if args.split != None:
    train, val, test = [], [], []
    split = [f.strip() for f in open(args.split).readlines()]
    for i in range(split):
        if split[i]=='train':
            train.append(i)
        if split[i]=='val':
            val.append(i)
        if split[i]=='test':
            test.append(i)
else:
    arr = np.arange(len(node_labels))
    np.random.shuffle(arr)
    train, val, test = np.split(arr, [int(0.6*len(arr)), int(0.8*len(arr))])
    train.sort()
    val.sort()
    test.sort()
train = np.array(train).reshape(1, len(train))
val = np.array(val).reshape(1, len(val))
test = np.array(test).reshape(1, len(test))

# read edge-type specific networks
edge_sparse_mats = []
for net in args.nets:
    nodei, nodej = [], []
    df_net = pd.read_csv(net, sep='\t', header=None)
    for i, row in df_net.iterrows():
        nodei.append(node_dict[row[0]])
        nodej.append(node_dict[row[1]])
    nodei2 = nodei + nodej
    nodej2 = nodej + nodei
    nodei2 = np.array(nodei2)
    nodej2 = np.array(nodej2)
    mat_edge = coo_matrix((np.ones(len(nodei2),), (nodei2, nodej2)))
    mat_edge = mat_edge.tocsr()
    edge_sparse_mats.append(mat_edge)

# pickle objects

out_dict = {'label':mat_label,
            'train_idx': train,
            'val_idx':val,
            'test_idx':test,
            'feature':features,
            'HetNet':{}}

for i in range(len(edge_sparse_mats)):
    out_dict['HetNet'][enames[i]] = edge_sparse_mats[i]
pickle.dump(out_dict, open(args.out, 'wb'))