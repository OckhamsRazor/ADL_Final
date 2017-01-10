wget http://speech.ee.ntu.edu.tw/~wjohn1483/ADL_final_QA/ADL_final_QA_dataset.zip
unzip ADL_final_QA_dataset.zip
mv Squad/training_data.json data/s_train.json
mv TOEFL_QA/train.json data/t_train.json
rm -r Squad TOEFL_QA
rm ADL_final_QA_dataset.zip
