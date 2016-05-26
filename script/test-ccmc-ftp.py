#!/usr/bin/env python

from ftplib import FTP
import StringIO

ftp = FTP('hanna.ccmc.gsfc.nasa.gov')     # connect to host, default port
ftp.login()                     # user anonymous, passwd anonymous@
ftp.cwd('pub/FlareScoreboard/in/UFCORIN_1')               # change into "debian" directory
str_file = StringIO.StringIO("hello")
#ftp.storlines('STOR test.txt', str_file)
ftp.delete('test.txt')
ftp.quit()

# from ftplib import FTP_TLS
# ftps = FTP_TLS('ftp://hanna.ccmc.gsfc.nasa.gov')
#
# /pub/FlareScoreboard/in/UFCORIN_1/
#
# ftps.login()           # login anonymously before securing control channel
# ftps.prot_p()          # switch to secure data connection
# ftps.retrlines('LIST') # list directory content securely
# ftps.quit()
