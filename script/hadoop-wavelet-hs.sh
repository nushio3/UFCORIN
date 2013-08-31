OUTPUT_DIR=hadoop-wavelet-hs-result

hadoop fs -rm -f -r $OUTPUT_DIR

sleep 1

hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.3.0.jar \
    -input resource/hdfs-hmi-single.txt \
    -output $OUTPUT_DIR \
    -mapper  "dist/build/main/main +RTS -N4 -RTS" \
    -reducer dist/build/hs-wordcount-reducer/hs-wordcount-reducer \
    -file    dist/build/main/main \
    -file    dist/build/hs-wordcount-reducer/hs-wordcount-reducer 

#    -input resource/hdfs-hmi-2011-01-01.txt \