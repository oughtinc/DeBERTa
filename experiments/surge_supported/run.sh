#!/bin/bash
#
# This is an example script to show how to made customized task
#
#
if [[  $# -lt 3 ]] ; then
	echo "Usage: $0 <output dir> <train_batch_size>"
	exit 1
fi

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/
data_dir=/content/drive/MyDrive/code/data/surge_supported

Task=SurgeSupported

init=base-mnli-scitail
tag=$init

output_dir=$1
train_batch_size=$2

# case ${init,,} in
# 	base-mnli)
# 	parameters=" --num_train_epochs 6 \
# 	--warmup 100 \
# 	--learning_rate 2e-5 \
# 	--cls_drop_out 0.2 \
# 	--max_seq_len 320"
# 		;;
# 	large-mnli)
# 	parameters=" --num_train_epochs 6 \
# 	--warmup 100 \
# 	--learning_rate 8e-6 \
# 	--cls_drop_out 0.2 \
# 	--max_seq_len 320"
# 		;;
# 	xlarge-mnli)
# 	parameters=" --num_train_epochs 6 \
# 	--warmup 100 \
# 	--learning_rate 7e-6 \
# 	--cls_drop_out 0.2\
# 	--fp16 True \
# 	--max_seq_len 320"
# 		;;
# 	xlarge-v2-mnli)
# 	parameters=" --num_train_epochs 6 \
# 	--warmup 100 \
# 	--learning_rate 4e-6 \
# 	--cls_drop_out 0.2 \
# 	--fp16 True \
# 	--max_seq_len 320"
# 		;;
# 	xxlarge-v2-mnli)
# 	parameters=" --num_train_epochs 6 \
# 	--warmup 100 \
# 	--learning_rate 3e-6 \
# 	--cls_drop_out 0.2 \
# 	--fp16 True \
# 	--max_seq_len 320"
# 		;;
# 	*)
# 		echo "usage $0 <Pretrained model configuration> <output dir>"
# 		echo "Supported configurations"
# 		echo "base-mnli - Pretrained DeBERTa v1 model with 140M parameters (12 layers, 768 hidden size)"
# 		echo "large-mnli - Pretrained DeBERta v1 model with 380M parameters (24 layers, 1024 hidden size)"
# 		echo "xlarge-mnli - Pretrained DeBERTa v1 model with 750M parameters (48 layers, 1024 hidden size)"
# 		echo "xlarge-v2-mnli - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
# 		echo "xxlarge-v2-mnli - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
# 		exit 0
# 		;;
# esac

python -m DeBERTa.apps.run \
	--model_config '/content/drive/MyDrive/code/checkpoints/base-mnli/model_config.json' \
	--init_model '/content/drive/MyDrive/code/checkpoints/base-mnli/pytorch.model-014000.bin' \
	--tag $tag \
	--do_train \
	--task_dir . \
	--task_name $Task \
	--data_dir  $data_dir \
	--output_dir $output_dir/$tag/$task \
	--train_batch_size $train_batch_size \
	--num_train_epochs 6 \
	--warmup 100 \
	--learning_rate 2e-5 \
	--cls_drop_out 0.2 \
	--max_seq_len 512