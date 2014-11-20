echo escape '^t^t' > .screenrc
sudo apt-get update
sudo apt-get install emacs ghc ghc-prof haskell-mode ruby ruby-dev libxml2-dev libxslt-dev g++ rake make libsqlite3-dev poppler-utils python-pip unzip cabal-install libgsl0-dev pkg-config
sudo pip install awscli
aws configure
cabal update
wget https://github.com/nushio3/UFCORIN/archive/master.zip -O ufcorin.zip
unzip ufcorin.zip
(cd UFCORIN-master; cabal install)
echo 'PATH=$PATH:/home/ubuntu/.cabal/bin' >> .bashrc
