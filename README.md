# ss_on_decoding_steps
Codes for "Scheduled Sampling Based on Decoding Steps for Neural Machine Translation" (long paper of EMNLP-2022).

## NMT

### Datasets
+ [WMT14 EN-DE](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh)
+ [WMT14 EN-FR](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh)
+ [WMT19 ZH-EN](http://www.statmt.org/wmt19/translation-task.html)


### Pre-training

```
code_dir=./nmt/THUMT
data_dir=path2yourdata
work_dir=./nmt
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export PYTHONPATH=$work_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=pretrained_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=100000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True
```



### Finetuning
```
code_dir=./nmt/THUMT-schedule_composite
work_dir=./nmt
data_dir=path2yourdata

export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

signature=finetune_model_name

output_dir=$work_dir/train/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $output_dir \
  --input $data_dir/train.src $data_dir/train.trg \
  --vocabulary $data_dir/dict.src.txt $data_dir/dict.trg.txt \
  --checkpoint $work_dir/train/pretrained_model_name
  --parameters=device_list=[0,1,2,3,4,5,6,7],eval_steps=90000000,train_steps=300000,batch_size=4096,max_length=128,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,num_encoder_layers=6,num_decoder_layers=6,layer_preprocess=none,layer_postprocess=layer_norm,update_cycle=1,hidden_size=512,filter_size=2048,num_heads=8,label_smoothing=0.1,warmup_steps=4000,learning_rate=1.0,save_checkpoint_steps=5000,keep_checkpoint_max=200,position_info_type=absolute,shared_embedding_and_softmax_weights=True,shared_source_target_embedding=True,mle_rate=0,zero_step=False,trainstep_scheduled_sampling_strategy=sigmoid,timestep_scheduled_sampling_strategy=exp,timestep_exp_epsilon=0.99,trainstep_sigmoid_k=20000
```



## Text Summarization

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

