
## Text Summarization


### Requirements
+ Pytorch version >= 1.2
+ Fairseq version >= 0.9.0
+ Python version >= 3.6



### Datasets
+ [CNN / Daily Mail](https://drive.google.com/file/d/1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1/view)
+ [Gigaword](https://drive.google.com/file/d/1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV/view?usp=drive_open)

We follow [ProphetNet](https://github.com/microsoft/ProphetNet) for data pre-processed and post-processed.
Below, we take Gigaword as an example to show the training process.


### Pre-training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,
work_dir=./text_summarization
DATA_DIR=path2_yourdata
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
TENSORBOARD_LOGDIR=$work_dir/ggw/finetune_ggw_tensorboard
PRETRAINED_MODEL=$work_dir/pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt
fairseq_path=fairseq

USER_DIR=$work_dir/prophetnet
SAVE_DIR=$work_dir/ggw/pretrained_model_name

python $fairseq_path/train_epoch_ss.py $DATA_DIR \
    --fp16 \
    --user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
    --lr 1e-4 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 64 --max-sentences 4 \
    --num-workers 8  \
    --load-sep \
    --load-from-pretrained-model $PRETRAINED_MODEL \
    --ddp-backend=no_c10d --max-epoch 10 \
    --max-source-positions 512 --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --keep-last-epochs 10 \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    --seed 1 \
```


### Finetuning

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,
work_dir=./text_summarization
DATA_DIR=path2_yourdata
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
TENSORBOARD_LOGDIR=$work_dir/ggw/finetune_ggw_tensorboard
PRETRAINED_MODEL=$work_dir/ggw/pretrained_model_name/checkpoint10.pt
fairseq_path=fairseq

USER_DIR=$work_dir/prophetnet
SAVE_DIR=$work_dir/ggw/finetune_model_name

python $fairseq_path/train_epoch_ss.py $DATA_DIR \
    --fp16 \
    --user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
    --lr 1e-4 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 64 --max-sentences 4 \
    --num-workers 8  \
    --load-sep \
    --load-from-pretrained-model $PRETRAINED_MODEL \
    --ddp-backend=no_c10d --max-epoch 10 \
    --max-source-positions 512 --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --keep-last-epochs 10 \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    --seed 1 \
    --decodingstep_schduled_sampling_strategy exp \
    --decodingstep_sigmoid_k 50 \
    --decodingstep_exp_radix 0.99
```
