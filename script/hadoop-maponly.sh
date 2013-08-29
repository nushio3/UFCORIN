hadoop fs -rm -f -r hadoop-result

cp dist/build/main/main main-mapper

hadoop jar  /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -input /user/nushio/filelist/ \
    -output hadoop-result \
    -mapper main-mapper
    -reducer "cat" \
    -files main-mapper
