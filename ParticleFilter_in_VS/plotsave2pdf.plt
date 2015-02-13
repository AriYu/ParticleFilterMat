##--------------------------------
## This script for test_PFM10, test_PFM11, test_PFM12
##--------------------------------
#x�Ɗϑ��l�݂̂��v���b�g
set grid
set xtics nomirror
set ytics nomirror

cd "./graph/estimation"
set terminal pdf

set output "true-obs.pdf"
plot "../../result1.dat" u 1 w lines t "true state", "../../result1.dat" u 2 w lines t "measurement","../../result1.dat" u 8 w lines t "second sensor(x1)","../../result1.dat" u 9 w lines t "third sensor(x1)","../../result1.dat" u 10 w lines t "forth sensor(x1)","../../result1.dat" u 11 w lines t "fifth sensor(x1)"

#����l�A�ϑ��l��MMSE���v���b�g
set output "mmse.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 3 w lines t "PF[MMSE](x1)"

# #����l�A�ϑ��l��ML���v���b�g
set output "ml.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 6 w lines lw 1.5 t "ML(x1)"

#����l�A�ϑ��l��EPVGM���v���b�g
set output "epvgm.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 4 w lines t "EPVGM(x1)"# ,"result1.dat" u 2 w lines t "first sensor(x1)"

#����l�A�ϑ��l��PFMAP���v���b�g
set output "pfmap.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 5 w lines t "PFMAP(x1)"

#����l�A�ϑ��l�APFMS��mmse���v���b�g
set output "meanshift.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 7 w lines t "PFMS(x1)"

#����l���v���b�g
set output "all.pdf"
plot "../../result1.dat" u 1 w lines t "true state","../../result1.dat" u 3 w lines t "PF[MMSE](x1)","../../result1.dat" u 4 w lines t "EPVGM(x1)","../../result1.dat" u 5 w lines t "PFMAP(x1)","../../result1.dat" u 7 w lines t "PFMS(x1)"

# ����l�Ɛ^�l�̍��̃O���t���v���b�g
set output "diff.pdf"
plot "../../result2.dat" u 2 w lines t "PFMS(x1)","../../result2.dat" u 1 w lines t "PF[MMSE](x1)"

#���x�̐���l���v���b�g�itest_PFM14�j
set output "mmse_velocity.pdf"
plot "../../result1.dat" u 12 w lines t "true(v1)","../../result1.dat" u 13 w lines t "PF[MMSE](v1)", "../../result1.dat" u 14 w lines t "PF[EP-VGM](v1)", "../../result1.dat" u 15 w lines t "PF[pf-MAP](v1)"


reset