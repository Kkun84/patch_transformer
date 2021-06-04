# patch_transformer

## Colabで動かす方法

`colab/patch_transformer.ipynb`をGoogle Colabで動かせます．学習や推論が簡単に試せます．

## 学習する

`src/train.py`を実行すれば学習できます．`runs/`に出力が保存されます．

## Dockerを動かす方法

Dockerを簡単に動かせるシェルスクリプトを`docler/`に配置しています．

| 事例                              | コマンド                |
| --------------------------------- | ----------------------- |
| Dockerイメージを作りたい          | `./docker/build.sh`     |
| Dockerコンテナを作りたい          | `./docker/run.sh`       |
| GPUなしのDockerコンテナを作りたい | `./docker/run-nogpu.sh` |
| Dockerコンテナに入りたい          | `./docker/exec.sh`      |
| Dockerコンテナを停止したい        | `./docker/stop.sh`      |

基本的には，
1. イメージを作成
2. コンテナを作成
3. コンテナに入る
4. コンテナ内で作業
5. （コンテナの利用が終われば）コンテナを停止
のようにします．
