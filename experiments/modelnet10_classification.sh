dir_name='test'
dataset='ModelNet10'
model='MetricClassificationNet'
num_vertices=1500
out_features=10
n_layers=5
n_hidden=100
embedding_dim=2
n_epochs=300
batch_size=1
seed=1
lr=1e-3
min_lr=1e-3

python train/classification.py \
    --seed ${seed} \
    --dir-name ${dir_name} \
    --dataset ${dataset} \
    --classes ${classes} \
    --model ${model} \
    --num-vertices ${num_vertices} \
    --out-features ${out_features} \
    --n-layers ${n_layers} \
    --n-hidden ${n_hidden} \
    --embedding-dim ${embedding_dim} \
    --n-epochs ${n_epochs} \
    --batch-size ${batch_size} \
    --lr ${lr} \
    --min-lr ${min_lr}

