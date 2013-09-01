OUTPUT_DIR=wavelet-result-`date +'%Y%m%d-%H%M%S'`

hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.3.0.jar \
    -input filelist/ \
    -output $OUTPUT_DIR \
    -mapper  "dist/build/main/main +RTS -N4 -RTS" \
    -reducer cat \
    -file    dist/build/main/main \


#    -input resource/hdfs-hmi-2011-01.txt \

#    -input resource/hdfs-hmi-single.txt \
#    -input resource/hdfs-hmi-2011-01-01.txt \
