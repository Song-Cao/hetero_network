import argparse
from Bio import Entrez

par = argparse.ArgumentParser()
par.add_argument('--infile', required=True, help='each line is a PMID')
args = par.parse_args()

Entrez.email = 'f.zheng.libra@gmail.com'

pmids = [int(l.strip()) for l in open(args.infile).readlines()]
handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pmids)),
                       rettype="xml", retmode="text")
records = Entrez.read(handle)

abstract_dict = {}
for pubmed_article in records['PubmedArticle']:
    pmid = int(str(pubmed_article['MedlineCitation']['PMID']))
    article = pubmed_article['MedlineCitation']['Article']
    if 'Abstract' in article:
        abstract_text = article['Abstract']['AbstractText']
        abstract_text = '\n'.join(abstract_text)
        # multiple sections in abstract?
        abstract_dict[pmid] = abstract_text

for pmid, abs in abstract_dict.items():
    outf = str(pmid) + '_abstract.txt'
    with open(outf, 'w') as fh:
        fh.write(abs)