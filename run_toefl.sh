input_file=$1
output_file=$2
python dmn_test.py -i $input_file
mv answer.txt $output_file
