cd /storage/longjing/run/dimsum/DimSum/src
python train.py -task "ext" -encoder "roberta" -mode "test" -bert_data_path "/storage/longjing/run/dimsum/DimSum/roberta-data/cnndm"  -test_from "/storage/longjing/run/dimsum/DimSum/models/roberta-result/cnndm/model_step_48000.pt" -result_path "" -test_batch_size 20
