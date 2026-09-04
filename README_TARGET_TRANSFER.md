# 学習済みソースからのAdvantage転移実験

## 今回の実験

Space Invadersで学習済みの物体入力モデルを固定ソースとし、Galaxianで
新規初期化したPPOを学習する。ソースを再学習せず、ソースの重みを
ターゲットへコピーするfine-tuningでもない。

今回指定されたソースは次のディレクトリの最終モデルとする。

```text
/home/yui-nagae/reserch/sourceobject/research-results/representation_comparison/objects/seed_0/model.pt
```

旧形式の`model.pt`には一部の環境設定が含まれないため、同じディレクトリの
元の`config.json`も残すこと。ソースの成績は本実装では判定しない。
このPCにある動作確認用モデルと、研究用PCの学習済みモデルは別物である。

## 手法は従来のまま

```text
A_T     = Target CriticによるGAE
A_S     = r_train + gamma * V_source(s_next) * (1-done) - V_source(s)
A_total = (1-alpha) * A_T + alpha * A_S
```

- Source Criticは評価モードで完全固定する。
- Target Actorの更新に`A_total`を使う。
- Value lossの教師値は従来どおりターゲットのGAEから作るreturnであり、混合しない。
  ActorとCriticは共通backboneを持つため、表現へのActor勾配の影響はある。
- Target GAEとSource TDを個別に正規化せず混合し、PPO更新時に混合後の
  Advantageだけを一度標準化する。scale matching・状態依存係数は導入しない。
- `alpha=0`は通常PPO、`alpha=0.5`は従来の固定係数条件。今回は減衰なし。
- 同じ学習報酬とterminal maskを両信号に使う。life-loss terminal、
  最大episode長、reset時のFIRE動作は現在の学習基盤の設定を使用する。
- 係数0.5は更新への寄与が必ず半々という意味ではない。両Advantageの
  スケール差は`updates.csv`で確認する。

旧`ocatari_transfer_full_experiment.py`はソースから再学習する旧一括実験、
`ppo_transfer_galaxian.py`は以前の画像用実装である。今回は次の新しい入口を使用する。

## 設定の継承と公平な比較

`ocatari_target_transfer.py`はソースモデルと隣接`config.json`から入力方式、
物体エンコーダ、画像前処理、frameskip、報酬条件、PPO設定を読み込む。
指定したソースが`scaled_raw`・0.1なら、ターゲットも同じ条件となる。
ソースのseed・学習step数・出力先は引き継がない。

`--total-steps`等でターゲット側の予算を指定できるが、入力表現・報酬単位・
gamma等の不一致は実行前に拒否する。表現次元だけ同じでスロット定義が違う
モデルも拒否する。Source Criticは状態価値だけを利用するので、ソース方策の
行動数とターゲットの行動数を同一にする必要はない。

同じtarget seedでalphaだけを変えると、初期Targetパラメータと乱数初期化が
揃う。Sourceモデル構築による乱数消費も復元し、初期Target重みのSHA-256を
各runの`config.json`に保存する。ソースのファイルハッシュも保存する。

まず`alpha=0`でGalaxianに通常学習が成立するか確認する。Baselineが改善しない
場合、転移結果だけから負の転移を断定しない。今回のsource seed 0固定・
複数target seed比較は、**このソースモデルを条件とした**転移検証である。
ソース学習のseed変動まで一般化するには、後続実験で複数ソースを使う。

## Linuxでの実行

リポジトリの最新版を研究用PCへ反映後、そのディレクトリへ移動し、
既存のvenvを有効にした状態で実行する。新しい依存パッケージは不要。
`python`は有効なvenvのものを使う。`.venv/bin/python`に固定しない。

### 1. ソース確認（学習しない）

```bash
SOURCE_MODEL="/home/yui-nagae/reserch/sourceobject/research-results/representation_comparison/objects/seed_0/model.pt"
RESULT_ROOT="/home/yui-nagae/reserch/sourceobject/research-results/transfer_galaxian_objects_seed0"

test -f "$SOURCE_MODEL" || { echo "source model.pt がありません"; exit 1; }
python -u ocatari_target_transfer.py \
  --source-checkpoint "$SOURCE_MODEL" \
  --inspect-source
```

入力方式・source step数・reward・PPO設定を確認する。`--inspect-source`は
メタデータを表示するだけで、モデル性能や環境動作の検証ではない。
`.pt`は任意コードを含み得るため、自分の信頼できる研究用モデルだけを指定する。

### 2. 短い実行確認

```bash
python -u ocatari_target_transfer.py \
  --source-checkpoint "$SOURCE_MODEL" \
  --transfer-alpha 0.5 \
  --device cuda \
  --quick \
  --output-dir "${RESULT_ROOT}_smoke"
```

最大2,048 environment stepsの動作確認であり、学習効果を判定する実験ではない。
この出力は本実験と分離する。CPUで確認する場合は`--device cpu`にする。

### 3. 転移なし／転移ありの比較

```bash
SOURCE_CHECKPOINT="$SOURCE_MODEL" \
OUTPUT_ROOT="$RESULT_ROOT" \
TOTAL_STEPS=1000000 \
SEEDS="0 1 2" \
ALPHAS="0 0.5" \
DEVICE=cuda \
bash run_target_transfer.sh
```

上の100万stepsは初回比較用の予算例であり、十分な学習を保証しない。
本実験の予算は全条件で事前に揃える。3 target seeds・2条件なら合計600万
target environment stepsとなる。まず1 seedだけ動かす場合は`SEEDS="0"`にする。
任意の中間係数を追加する場合は`ALPHAS="0 0.25 0.5"`にする。
`bash`で起動するためシェルスクリプトへの実行権限付与は不要。

各runはrandom/初期方策評価の後に学習し、進捗・steps/s・ETAを表示する。
最初の評価中にはまだ学習進捗が出ない。標準出力は`-u`でバッファしない。

### 4. 再開

同じ設定で上のスクリプトを再実行すると、存在する`checkpoint_latest.pt`から
再開する。ソースをターゲットの`--resume`へ渡してはいけない。

```bash
python -u ocatari_target_transfer.py \
  --source-checkpoint "$SOURCE_MODEL" \
  --resume "$RESULT_ROOT/alpha_0.5/seed_0/checkpoint_latest.pt" \
  --total-steps 1000000 \
  --device cuda
```

再開時は保存済みのalpha・seed・学習設定が復元される。sourceハッシュが
変わった場合は停止する。同じモデルを移動した場合は新しいパスを明示できる。
環境のエミュレータ状態までは保存しないため、中断なしの実行と完全同一ではない。
学習率減衰の分母が`total-steps`なので、途中で予算を延長した結果を最初から
その予算で学習した結果と同一扱いしない。

ソース保存先またはその親をtarget出力先に指定すると停止する。
新規実行では既存の非空targetディレクトリも上書きしない。

## 出力とグラフ

```text
transfer_galaxian_objects_seed0/
  alpha_0/seed_0/       # 通常PPO
  alpha_0.5/seed_0/     # 固定alphaの転移
  ...
  comparison/
    transfer_evaluation_stochastic.png
    paired_metrics_stochastic.csv
    comparison_stochastic.json
```

各runには従来と同じ`model.pt`、`checkpoint_latest.pt`、定期評価の最良
`checkpoint_best.pt`、`episodes.csv`、`updates.csv`、`periodic_evaluation.csv`、
`progress.json`、学習曲線が保存される。`transfer_metadata.json`にはソースの
環境・学習steps・seed・ハッシュ・手法を保存する。ソースファイルは読み取り専用。

`updates.csv`の追加列は、alpha、Source Advantageの平均・標準偏差、混合後の
平均・標準偏差、Source/Targetの符号一致率と相関。既存`advantage_mean/std`は
Target GAEの混合前統計である。alpha=0ではTeacherを評価しないためsource列は空欄。

比較スクリプトは同じtarget seed・初期重み・sourceハッシュ・評価時点・予算・
学習条件を検証し、固定seed評価曲線と、各seedの最終raw return／区間長で割った
AUC／Baselineとの差を出力する。1 seedではSEMを描かない。複数seedのSEMは
target学習seed間の値で、評価episode間の誤差ではない。

```bash
python compare_transfer_runs.py --root "$RESULT_ROOT"
python compare_transfer_runs.py --root "$RESULT_ROOT" --policy deterministic
```

## 学習済みターゲットのMP4

```bash
python render_trained_agent.py \
  --model "$RESULT_ROOT/alpha_0.5/seed_0/model.pt" \
  --device cuda --episodes 3 --policy deterministic \
  --output-dir "$RESULT_ROOT/alpha_0.5/seed_0/playback"
```

Galaxianと入力方式は保存メタデータから取得する。転移元モデルは動画作成時には不要。
転移なしも同じ評価seed・方策選択で再生し、学習曲線・定量評価と併せて判断する。
