parallel --eta -j4 --joblog parallel.log  'echo python realtime_forecast.py -q -g {%} {} | bash' :::: argument-list.txt

