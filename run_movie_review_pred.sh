# GAT based on consituency graph
nohup python -u examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/movie/gat_bi_fuse_constituency.yaml --gpu 1 > gat_const.log &

# GAT based on dependency graph
python -u examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/movie/gat_bi_fuse_dependency.yaml --gpu 2 > gat_dep.log &

# GraphSAGE based on dependency graph
python -u examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/movie/graphsage_bi_fuse_dependency.yaml --gpu 3 > sage_const.log &

# GraphSAGE based on consituency graph
python -u examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/movie/graphsage_bi_fuse_constituency.yaml --gpu 4 > sage_dep.log &

# evaluate
# python -u examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/movie/graphsage_bi_fuse_dependency.yaml --gpu 4 --eval
