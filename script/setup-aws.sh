echo escape '^t^t' > .screenrc
sudo apt-get update
sudo apt-get install emacs ghc ghc-prof haskell-mode ruby ruby-dev libxml2-dev libxslt-dev g++ rake make libsqlite3-dev poppler-utils python-pip unzip gmp-devel
sudo pip install awscli
# wget https://github.com/s3tools/s3cmd/archive/master.zip -O s3cmd.zip 
# unzip s3cmd.zip
# (cd s3cmd-master; sudo python setup.py install)
# sudo gem update
# s3cmd --configure
aws --configure

# wget https://www.haskell.org/platform/download/2014.2.0.0/haskell-platform-2014.2.0.0-unknown-linux-x86_64.tar.gz -O haskell.tar.gz
# (cd /; sudo tar xvf /home/ec2-user/haskell.tar.gz; sudo /usr/local/haskell/ghc-7.8.3-x86-64/bin/activate-hs)
# 


cabal update
wget https://github.com/nushio3/UFCORIN/archive/master.zip -O ufcorin.zip
unzip ufcorin.zip
(cd UFCORIN-master; cabal install)
