echo "----------------------"
echo "Installs datasets"
mkdir -p datasets
cd datasets/
curl -L "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.zip" -o spambase.zip
curl -L "https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/trec05p-1.tgz" -o trec05p-1.tgz
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00259/rcv1rcv2aminigoutte.tar.bz2" -o rcv.tar.bz2
echo "----------------------"
echo "Downloads complete"
echo "De-compressing"
mkdir -p spambase
unzip -d spambase spambase.zip
tar xf rcv.tar.bz2

