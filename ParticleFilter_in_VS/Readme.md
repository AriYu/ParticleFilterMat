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
事前分布と事後分布のヒストグラムを同時にプロット
```bash
$ plot "result_particle.dat" ind 77 u (filter($1,1)):(1) smooth frequency with boxes t "proposal","result_after_particle.dat" ind 77 u (filter($1,1)):(1) smooth frequency with boxes t "posterior"i
```

# TODO
- リサンプリング後のパーティクルのヒストグラムを表示する
- 事後分布をクラスタリングしてヒストグラムを表示する
- 事後分布が多峰性かどうかを判定する
- 多峰性の場合はクラスタリングを行って、パーティクルの多いクラスタの重み付き平均を取る
