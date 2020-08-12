import os
import argparse
import numpy as np
import gensim
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

par = argparse.ArgumentParser()
par.add_argument('--cor', required=True, help='the corpus file')
par.add_argument('--dim', default=50, type=int, help='doc2vec parameter')
par.add_argument('--min_count', default=2, type=int, help='doc2vec parameter')
par.add_argument('--epoch', default=40, type=int, help='doc2vec parameter')
args = par.parse_args()

train_corpus = list(read_corpus(args.cor))
model = gensim.models.doc2vec.Doc2Vec(vector_size=args.dim,
                                      min_count=args.min_count,
                                      epochs=args.epoch)
model.build_vocab(train_corpus)
model.train(train_corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs)

# output the embedding of documents, and document similarity
doc_vec = np.zeros((len(train_corpus), args.dim))
rank_sim = np.zeros((len(train_corpus), len(train_corpus)))
mat_sim = np.zeros((len(train_corpus), len(train_corpus)))

for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    doc_vec[doc_id, :] = np.array(inferred_vector)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = np.array([d for d, _ in sims])
    simi = np.array([sim for _, sim in sims])
    rank_sim[doc_id, :] = rank
    mat_sim[doc_id, :] = simi
    # print(doc_id)

np.save('doc_vec', doc_vec)
np.save('doc_simrank', rank_sim)
np.save('doc_sim', mat_sim)