LR_FOLDER=/mnt/HDD1/home/mtlong/workspace/Data_Storage/Datasets/CUFED5
DICT_FOLDER=/mnt/HDD1/home/mtlong/workspace/Data_Storage/Datasets/CUFED_Dictionary
OUTPUT_FOLDER=/mnt/HDD1/home/mtlong/workspace/Data_Storage/Results/Aggregative_Learning/TTSR/CUFED5_Test_AllRef_Big

### test
python main.py --save_dir $OUTPUT_FOLDER \
               --reset True \
               --log_file_name test.log \
               --test False \
               --test_FixedRef False\
               --test_AllRef True\
               --num_workers 1 \
               --lr_path $LR_FOLDER \
               --ref_path $DICT_FOLDER \
               --model_path ./TTSR-rec.pt

            