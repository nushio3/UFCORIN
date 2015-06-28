#!/usr/bin/env python
import subprocess
def system(cmd):
    subprocess.call(cmd, shell=True)


type = 'hmi.M_45s'
start = '2013.05.31_00:00:00'
end = '2013.05.31_24:00:00'
interval = '2h'

query = type+'['+start+'-'+end+'@'+interval+']'
command = "./exportfile_mod.csh '{}' {}".format(query, 'muranushi@gmail.com')
print command
system(command)
