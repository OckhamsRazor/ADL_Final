wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm -rf data
mkdir data
mkdir data/glove
mkdir data/glove/glove.6B
mv glove.6B.100d.txt data/glove/glove.6B
rm -f glove.6B*
