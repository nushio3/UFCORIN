hadoop fs -rm -f -r test-hadoop-result

hadoop jar  /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -input resource/hdfs-hmi-2011-01.txt \
    -output test-hadoop-result \
    -mapper dist/build/test-mapper/test-mapper \
    -reducer dist/build/test-reducer/test-reducer
