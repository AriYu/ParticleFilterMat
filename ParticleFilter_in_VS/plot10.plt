##--------------------------------
## This script for test_PFM10.
##--------------------------------
#x�Ɗϑ��l�݂̂��v���b�g
plot "result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "first sensor(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#����l�A�ϑ��l��MMSE���v���b�g
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 3 w lines t "PF[MMSE](x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#����l�A�ϑ��l��EPVGM���v���b�g
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 4 w lines t "EPVGM(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#����l�A�ϑ��l��MMSE,EPVGMA���v���b�g
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 6 w lines lw 1.5 t "EPVGMA(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#����l�A�ϑ��l��PFMAP���v���b�g
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 5 w lines t "PFMAP(x1)"
pause -1 "press [Enter] key or [OK] button to quit"

#����l���v���b�g
plot "result1.dat" u 1 w lines t "true(x1)","result1.dat" u 3 w lines t "PF[MMSE](x1)","result1.dat" u 4 w lines t "EPVGM(x1)","result1.dat" u 5 w lines t "PFMAP(x1)"
pause -1 "press [Enter] key or [OK] button to quit"
reset