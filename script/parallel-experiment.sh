parallel --eta -j4 --joblog parallel.log  'echo python realtime_forecast.py -g {%} {} | bash' :::: argument-list.txt

