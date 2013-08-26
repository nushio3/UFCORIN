hadoop fs -rm -f -r hadoop-result

cp dist/build/main/main main-mapper

hadoop jar  /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -input resource/hdfs-hmi-2011-01.txt \
    -output hadoop-result \
    -mapper main-mapper \
    -reducer "cat" \
    -file main-mapper

#    -input /user/nushio/filelist/ \
