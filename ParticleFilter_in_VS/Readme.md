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
$ ./test_PFM*
$ source plots.sh
```
## test_PFM12
非線形、多峰性モデル

## plots.sh
以下のスクリプトを一気に実行するシェルスクリプト．

```sh
$ source plots.sh >& plot.log 
```

エラー出力は`plot.log`に書き出される.
### plotestimation.plt
各推定手法による推定結果をプロット

### plothistogram.plt
リサンプリング前とリサンプリング後のパーティクルのヒストグラムをプロット
クラスタリング後のパーティクルもプロット

### plotparticles.plt
サンプリング後のパーティクルの位置と更新後の重みをプロット

### plotlastparticles.plt
サンプリング後のパーティクルの位置と更新前の重みをプロット



# TODO
- リサンプリング後のパーティクルのヒストグラムを表示する
- 事後分布をクラスタリングしてヒストグラムを表示する
- 事後分布が多峰性かどうかを判定する
- 多峰性の場合はクラスタリングを行って、パーティクルの多いクラスタの重み付き平均を取る
