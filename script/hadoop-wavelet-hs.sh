OUTPUT_DIR=hadoop-wavelet-hs-result

hadoop fs -rm -f -r $OUTPUT_DIR

hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.3.0.jar \
    -input resource/hdfs-hmi-2011-01.txt \
    -output $OUTPUT_DIR \
    -mapper  dist/build/main/main \
    -reducer dist/build/hs-wordcount-reducer/hs-wordcount-reducer \
    -file    dist/build/main/main \
    -file    dist/build/hs-wordcount-reducer/hs-wordcount-reducer 

