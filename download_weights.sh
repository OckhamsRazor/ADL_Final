#wget https://github.com/c62066206/Model/releases/download/1.0/weights.tar.gz
wget https://www.dropbox.com/s/n2rlq0plm8xs37i/weights.tar.gz
tar zxvf weights.tar.gz 
mv weights/task6.weights weights/SQUAD.weights
wget https://www.dropbox.com/s/8isy862j3hzwlry/toefl.weights
mv toefl.weights weights/TOEFL.weights
rm -f weights.tar.gz

