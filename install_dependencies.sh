#!/bin/bash

sudo apt install python3 -y
sudo apt install python3-pip -y
sudo apt install git -y 
sudo apt install autoconf -y
sudo apt install automake -y
sudo apt install bison -y
sudo apt install gettext -y
sudo apt install gperf -y
sudo apt install perl -y
sudo apt install gzip -y
sudo apt install rsync -y
sudo apt install tar -y
sudo apt install texinfo -y
sudo apt install autopoint -y
sudo apt install glibc-source -y
git clone git://git.sv.gnu.org/coreutils
cd coreutils
git clone git://git.savannah.gnu.org/gnulib.git
cd ..
pip3 install nltk
pip3 install sklearn
pip3 install matplotlib
pip3 install pandas
