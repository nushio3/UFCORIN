# for GPU execution
# parallel --eta -j5 --joblog parallel.log  'echo python realtime_forecast.py -q -g {%} {} | bash' :::: argument-list.txt

# for CPU execution
OPENBLAS_NUM_THREADS=1 parallel --eta -j28 --joblog parallel.log  'echo python realtime_forecast.py -q {} | bash' :::: argument-list.txt

