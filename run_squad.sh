input_file=$1
output_file=$2

if [ ! -d "weights" ]; then
    bash download_all.sh
fi

python squad_test.py -i $input_file
mv answer.txt $output_file
