echo escape '^t^t' > .screenrc
sudo yum update
sudo yum install emacs ghc ghc-prof haskell-mode ruby ruby-dev libxml2-dev libxslt-dev g++ rake make libsqlite3-dev poppler-utils python-pip unzip
sudo pip install awscli
wget https://github.com/s3tools/s3cmd/archive/master.zip -O s3cmd.zip 
unzip s3cmd.zip
'(cd s3cmd-master; sudo python setup.py install)'
sudo gem update

