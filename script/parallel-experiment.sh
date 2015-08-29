parallel -j4 --joblog parallel.log  'echo python realtime_forecast.py -g {%} {}' :::: argument-list.txt

