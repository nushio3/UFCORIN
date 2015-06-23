import os

type = 'hmi.M_45s'
start = '2013.01.01_00:00:00'
end = '2013.01.01_02:00:00'
interval = '30m'

query = type+'['+start+'-'+end+'@'+interval+']'
command = "./exportfile.csh "+query
print command
os.system(command)
