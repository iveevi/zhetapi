mkdir $HOME/usr
export PATH="$HOME/usr/bin:$PATH"
wget https://cmake.org/files/v3.14/cmake-3.14.2-Linux-x86_64.sh
chmod +x cmake-3.14.2-Linux-x86_64.sh
./cmake-3.14.2-Linux-x86_64.sh --prefix=$HOME/usr --exclude-subdir --skip-license

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install libboost-dev gcc-8 g++-8 tree
