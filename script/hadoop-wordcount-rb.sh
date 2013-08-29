OUTPUT_DIR=hadoop-wordcount-rb-result

hadoop fs -rm -f -r $OUTPUT_DIR

hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.3.0.jar \
    -input resource/example.txt \
    -output $OUTPUT_DIR \
    -mapper  script/rb-wordcount-mapper.rb  \
    -reducer script/rb-wordcount-reducer.rb \
    -file  script/rb-wordcount-mapper.rb \
    -file  script/rb-wordcount-reducer.rb
