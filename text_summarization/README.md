
## Text Summarization


### Requirements
+ Pytorch version >= 1.2
+ Fairseq version >= 0.9.0
+ Python version >= 3.6



### Datasets
+ [CNN / Daily Mail](https://drive.google.com/file/d/1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1/view)
+ [Gigaword](https://drive.google.com/file/d/1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV/view?usp=drive_open)

We follow [ProphetNet](https://github.com/microsoft/ProphetNet/tree/master/ProphetNet_En) for data pre-processed and post-processed.
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

python $fairseq_path/train.py $DATA_DIR \
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

USER_DIR=$work_dir/prophetnet_decoding_step_ss_mle_absolute
SAVE_DIR=$work_dir/ggw/finetune_model_name

python $fairseq_path/train.py $DATA_DIR \
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


### Inference

```
export CUDA_VISIBLE_DEVICES=0,1

work_dir=./text_summarization
DATA_DIR=path2_yourdata
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
TENSORBOARD_LOGDIR=$work_dir/ggw/finetune_ggw_tensorboard
PRETRAINED_MODEL=$work_dir/ggw/pretrained_model_name/checkpoint10.pt
fairseq_path=fairseq


signature=finetune_model_name

output_dir=$work_dir/ggw/results/$signature
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

for idx in `seq 1 1 10`; do
  BEAM=4
  LENPEN=1.0
  SUFFIX=_ck"$idx"_pelt"$LENPEN"_test_beam"$BEAM"
  CHECK_POINT=$work_dir/ggw/$signature/checkpoint"$idx".pt
  python3 $fairseq_path/generate.py $data_dir --path $CHECK_POINT --user-dir $work_dir/prophetnet --task translation_prophetnet --batch-size 150 --gen-subset test --beam $BEAM --num-workers 6 --lenpen $LENPEN 2>&1 --results-path $output_dir
  cp $output_dir/generate-results.txt $output_dir/out$SUFFIX
  grep ^H  $output_dir/out$SUFFIX | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" >  $output_dir/out"$SUFFIX"_sorted
done
```


### Evaluation

We use [pyrouge](https://github.com/bheinzerling/pyrouge) for evaluation and follow [ProphetNet](https://github.com/microsoft/ProphetNet/tree/master/ProphetNet_En) for data post-processed.

+ CNN / Daily Mail
```
python ./evaluation/cnndm/postprocess_cnn_dm.py \
  --generated $model_output \
  --golden ./evaluation/cnndm/cnndm.test.summary > $model_output.score
```

+ Gigaword
```
python ./evaluation/ggw/eval_ggw.py \
      --pred $model_output \
      --gold ./evaluation/ggw/ggw.test.summary  --perl > $model_output.score

```


