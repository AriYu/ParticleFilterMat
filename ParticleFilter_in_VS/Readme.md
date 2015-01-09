# ParticleFilterMat

```bash
$ git clone https://github.com/AriYu/ParticleFilterMat.git .
$ cd ParticleFilterMat
$ mkdir bin
$ mkdir -p graph/particles
$ mkdir -p graph/viterbi
$ cd bin
$ cmake ..
$ make
$ cd ..
$ ./testPFM*
```

ヒストグラムをプロット
```bash
$ filter(x,y)=int(x/y)*y
$ plot "result_particle.dat" ind 29 u (filter($1,0.1)):(1) smooth frequency with boxes, "result_particle.dat" ind 29 u 1:2
```
# TODO
- リサンプリング後のパーティクルのヒストグラムを表示する
