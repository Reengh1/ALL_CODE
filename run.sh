device=0
data=/data/guangchen_li/datasets/amazon/
batch=4
n_head=4
n_layers=4
d_model=512
d_rnn=64
d_inner=1024
d_k=512
d_v=512
dropout=0.1
lr=1e-4
smooth=0.1
epoch=20
log=log.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python /home/guangchen_li/dev/THP_Source_Code/Transformer-Hawkes-Process-master/Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log
