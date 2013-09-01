OUTPUT_DIR=hadoop-wordcount-hs-result

hadoop fs -rm -f -r $OUTPUT_DIR

hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.3.0.jar \
    -input resource/example.txt \
    -output $OUTPUT_DIR \
    -mapper  dist/build/hs-wordcount-mapper/hs-wordcount-mapper  \
    -reducer dist/build/hs-wordcount-reducer/hs-wordcount-reducer \
    -file    dist/build/hs-wordcount-mapper/hs-wordcount-mapper \
    -file    dist/build/hs-wordcount-reducer/hs-wordcount-reducer 


