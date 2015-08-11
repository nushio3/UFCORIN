while :
do
    ./download-hmi-nrt.py -m  $1
    echo 'waiting...'
    sleep 360
done
