#x1のみをプロット
plot "result.dat" u 1 w lines t "true(x1)", "result.dat" u 4 w lines t "kalman(x1)" ,"result.dat" u 7 w lines t "PF[MMSE](x1)"
pause -1 "press [Enter] key or [OK] button to quit"
#x2のみをプロット
plot "result.dat" u 2 w lines t "true(x2)", "result.dat" u 5 w lines t "kalman(x2)" ,"result.dat" u 8 w lines t "PF[MMSE](x2)"
pause -1 "press [Enter] key or [OK] button to quit"
#x3のみをプロット
plot "result.dat" u 3 w lines t "true(x3)", "result.dat" u 6 w lines t "kalman(x3)" ,"result.dat" u 9 w lines t "PF[MMSE](x3)"
pause -1 "press [Enter] key or [OK] button to quit"

reset
#pause -1 "press [Enter] key or [OK] button to quit"
