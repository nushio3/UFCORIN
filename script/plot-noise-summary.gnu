set term postscript landscape enhanced color 20
set grid
set log x

set ylabel "True Skill Statistics"

set xlabel "Time Noise"

set out 'survey-noise/XClass--tn-summary.eps'
plot \
   'survey-noise/XClass[--tn-summary]-00.txt' w line t "X Class"   lw 4 , \
   'survey-noise/MClass[--tn-summary]-00.txt' w line t ">=M Class" lw 4 , \
   'survey-noise/CClass[--tn-summary]-00.txt' w line t ">=C Class" lw 4 lt 4 lc rgb "blue", \
   'survey-noise/XClass[--tn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0, \
   'survey-noise/MClass[--tn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0 lc rgb "green", \
   'survey-noise/CClass[--tn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0 lc rgb "blue"

set xlabel "Amplitude Noise"

set out 'survey-noise/XClass--sn-summary.eps'
plot \
   'survey-noise/XClass[--sn-summary]-00.txt' w line t "X Class"   lw 4 , \
   'survey-noise/MClass[--sn-summary]-00.txt' w line t ">=M Class" lw 4 , \
   'survey-noise/CClass[--sn-summary]-00.txt' w line t ">=C Class" lw 4 lt 4 lc rgb "blue", \
   'survey-noise/XClass[--sn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0, \
   'survey-noise/MClass[--sn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0 lc rgb "green", \
   'survey-noise/CClass[--sn-summary]-00.txt' w yerr t ""          lw 4 lt 1 pt 0 lc rgb "blue"
