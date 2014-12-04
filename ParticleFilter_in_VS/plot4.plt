#x1‚Ì‚İ‚ğƒvƒƒbƒg
plot "result1.dat" u 1 w lines t "true(x1)", "result1.dat" u 2 w lines t "observed(x1)" ,"result1.dat" u 3 w lines t "PF[MMSE](x1)"
pause -1 "press [Enter] key or [OK] button to quit"
reset
#set isosamples 50
splot "result_particle.dat" pt 7 ps 0.5 w lines
reset
#pause -1 "press [Enter] key or [OK] button to quit"
