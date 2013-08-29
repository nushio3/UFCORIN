cat resource/example.txt | \
    ./dist/build/hs-wordcount-mapper/hs-wordcount-mapper | \
    sort | \
    ./dist/build/hs-wordcount-reducer/hs-wordcount-reducer