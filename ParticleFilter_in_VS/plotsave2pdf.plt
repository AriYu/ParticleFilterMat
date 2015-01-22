##--------------------------------
## This script for test_PFM10, test_PFM11, test_PFM12
##--------------------------------
#x�Ɗϑ��l�݂̂��v���b�g
set grid
cd "./graph/estimation"
set terminal pdf

set output "true-obs.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)", "../../result1.dat" u 2 w lines t "first sensor(x1)","../../result1.dat" u 8 w lines t "second sensor(x1)"

#����l�A�ϑ��l��MMSE���v���b�g
set output "mmse.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 3 w lines t "PF[MMSE](x1)"

# #����l�A�ϑ��l��ML���v���b�g
set output "ml.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 6 w lines lw 1.5 t "ML(x1)"

#����l�A�ϑ��l��EPVGM���v���b�g
set output "epvgm.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 4 w lines t "EPVGM(x1)"# ,"result1.dat" u 2 w lines t "first sensor(x1)"

#����l�A�ϑ��l��PFMAP���v���b�g
set output "pfmap.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 5 w lines t "PFMAP(x1)"

#����l�A�ϑ��l�APFMS��mmse���v���b�g
set output "meanshift.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 7 w lines t "PFMS(x1)","../../result1.dat" u 3 w lines t "PF[MMSE](x1)"

#����l���v���b�g
set output "all.pdf"
plot "../../result1.dat" u 1 w lines t "true(x1)","../../result1.dat" u 3 w lines t "PF[MMSE](x1)","../../result1.dat" u 4 w lines t "EPVGM(x1)","../../result1.dat" u 5 w lines t "PFMAP(x1)","../../result1.dat" u 7 w lines t "PFMS(x1)"

reset