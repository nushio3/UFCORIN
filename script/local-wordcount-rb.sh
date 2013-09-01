cat resource/example.txt | \
    ./script/rb-wordcount-mapper.rb | \
    sort | \
    ./script/rb-wordcount-reducer.rb 
