# OCAtari source-task PPO

転移学習へ進む前に、OCAtari REMの物体中心表現だけでSpace Invadersの
PPO学習が成立するかを確認するためのプログラムである。

## 実験条件

- 環境: `ALE/SpaceInvaders-v5`
- 物体抽出: OCAtari RAM Extraction Mode（REM）
- 入力: 91次元/フレーム × 4フレーム = 364次元
- 物体スロット: 前フレームとの最近傍対応を取る`temporal`
- 学習報酬の既定値: `scaled_raw`（raw reward × 0.1）
- 評価・判定に使う報酬: 加工前のraw return
- 転移学習: 実行しない

## Linux研究用PCの準備

Python 3.10～3.12を使用する。Python 3.14はOCAtari 2.2.1が要求する
NumPy 1.26.4と互換性がない。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CUDA版PyTorchが必要な場合は、研究用PCのCUDAに対応するPyTorchを先に
インストールする。その後で残りの依存関係を導入する。

```bash
python -m pip install -r requirements_ocatari_transfer.txt
```

## 実行確認

これは2,048 environment stepsのパイプライン確認であり、学習効果を
判断する実験ではない。

```bash
python ocatari_source_ppo.py \
  --quick \
  --device cpu \
  --output-dir source_smoke
```

単体テスト:

```bash
python -m unittest -v test_ocatari_source_ppo.py
```

## ソースタスク本学習

1 seed:

```bash
python ocatari_source_ppo.py \
  --env ALE/SpaceInvaders-v5 \
  --object-mode ram \
  --total-steps 1000000 \
  --seed 0 \
  --device cuda \
  --reward-mode scaled_raw \
  --reward-scale 0.1 \
  --slot-strategy temporal \
  --frameskip 4 \
  --rollout-steps 1024 \
  --eval-interval 100000 \
  --eval-episodes 10 \
  --random-eval-episodes 20 \
  --checkpoint-interval 100000 \
  --output-dir source_baseline_runs/seed_0
```

3 seedsを順次実行:

```bash
chmod +x run_source_seeds.sh
PYTHON_BIN=.venv/bin/python DEVICE=cuda TOTAL_STEPS=1000000 \
  REWARD_MODE=scaled_raw REWARD_SCALE=0.1 \
  SEEDS="0 1 2" ./run_source_seeds.sh
```

スクリプトは全seed終了後に`aggregate_source_runs.py`を実行し、
`source_baseline_runs/aggregate`へseed間の表、評価曲線、判定用summaryを
保存する。個別実行した結果は次のコマンドでも集計できる。

```bash
python aggregate_source_runs.py \
  --root source_baseline_runs \
  --output-dir source_baseline_runs/aggregate
```

## 再開

`--total-steps`には再開後の追加量ではなく、学習開始時点からの合計量を
指定する。

```bash
python ocatari_source_ppo.py \
  --resume source_baseline_runs/seed_0/checkpoint_latest.pt \
  --total-steps 1000000 \
  --device cuda
```

`--output-dir`を省略すると、checkpointと同じディレクトリへ再開結果を
保存する。再開時はモデル、optimizer、environment steps、PPO更新数、
ログ、Python/NumPy/PyTorch/CUDAの乱数状態を復元する。ALE内部状態は
移植性のため保存せず、新しいエピソードから再開する。

## 報酬モード

- `raw`: ゲーム本来の報酬
- `clipped`: raw rewardを`[-1, 1]`へクリップ
- `scaled_raw`: raw rewardへ`--reward-scale`を乗算（既定）
- `scaled_survival`: スケーリングした得点へALEフレーム単位の生存報酬と
  life lossペナルティを加算
- `shaped`: 旧実装の再現用。environment step単位の生存ボーナスと
  死亡ペナルティを加えてから全体をscale

すべてのモードで`episodes.csv`と評価ファイルにはraw returnを別に保存する。

### 報酬スケーリング

既定値は次の式である。

```text
training_reward = raw_reward × 0.1
```

正の定数倍なので得点同士の相対関係を維持したまま、Value targetと
Advantageの大きさ・分散を縮小できる。`clipped`と異なり、5点と30点の
違いも保持する。

### フレーム生存報酬

`scaled_survival`では次の式を使用する。

```text
score_component
  = raw_reward × reward_scale

survival_component
  = 実際に進んだALEフレーム数 × survival_reward_per_frame

life_loss_component
  = life loss時のみ life_loss_penalty

training_reward
  = score_component
  + survival_component
  + life_loss_component
```

推奨初期値:

```text
reward_scale = 0.1
survival_reward_per_frame = 0.001
life_loss_penalty = -1.0
```

frameskip 4なら通常の1 environment step当たり生存報酬は`0.004`である。
Space Invadersの5点報酬はscale後`0.5`なので、生存信号を与えつつ得点信号を
直ちに上回らない大きさにしている。

生存報酬は方策の目的を変えるため、`scaled_raw`と混ぜて一条件だけを
実行しない。まず`scaled_raw`を実施し、同じseed・steps・評価seedで
`scaled_survival`を別条件として比較する。

```bash
PYTHON_BIN=.venv/bin/python DEVICE=cuda TOTAL_STEPS=1000000 \
  REWARD_MODE=scaled_survival REWARD_SCALE=0.1 \
  SURVIVAL_REWARD_PER_FRAME=0.001 LIFE_LOSS_PENALTY=-1.0 \
  OUTPUT_ROOT=source_survival_runs SEEDS="0 1 2" \
  ./run_source_seeds.sh
```

`updates.csv`には変換前後の報酬平均・標準偏差を保存する。
`episodes.csv`にはscore、生存、life lossの累積成分を分離して保存する。
これにより、スケーリングで分散が実際に縮小したか、生存報酬が得点報酬を
支配していないかを確認できる。

## 出力

- `config.json`: 全実行条件、ライブラリ版、行動一覧
- `model.pt`: 転移実験で読み込む最終モデル
- `checkpoint_latest.pt`: optimizerと学習状態を含む再開用checkpoint
- `checkpoint_best.pt`: 定期deterministic評価が最良だったcheckpoint
- `episodes.csv`: raw return、学習報酬return、報酬成分、長さ、life loss
- `updates.csv`: PPO loss、KL、clip fraction、勾配norm、explained variance
  および変換前後の報酬分散
- `periodic_evaluation.csv`: 固定seedによるrandom/初期/定期/最終評価
- `action_distribution.csv`: PPO更新ごとの行動回数と割合
- `object_categories.csv`: REMカテゴリの出現数と出現フレーム率
- `object_diagnostics.json`: 自機・敵・飛翔物の検出率など
- `observation_statistics.csv`: 364入力次元ごとの統計
- `object_samples.json`: 最初のフレームの物体一覧と変換後ベクトル
- `object_detection_snapshot.png`: REM物体位置の可視化
- `raw_return_curve.png`: 学習中raw return
- `reward_components_curve.png`: score・生存・life loss報酬成分
- `reward_std_comparison.png`: rolloutごとの変換前後の報酬標準偏差
- `periodic_evaluation.png`: 固定seed評価曲線
- `summary.json`: 学習前後とrandom baselineの主要比較

複数seed集計では次を追加する。

- `aggregate/per_seed_summary.csv`
- `aggregate/evaluation_curve_aggregate.csv`
- `aggregate/evaluation_curve_across_seeds.png`
- `aggregate/aggregate_summary.json`

## Baseline成立の判断

Quick実行は判断材料にしない。最低3 seeds、各500,000～1,000,000
environment stepsで次を確認する。

1. 最終stochastic評価のseed間平均がrandom baselineを明確に上回る。
2. 最終stochastic評価が初期stochastic評価を上回る。
3. 固定seedの定期評価が一時的な上振れではなく、複数評価点で改善する。
4. 学習終盤20%のraw returnが学習序盤より高い。
5. `player_detection_rate`と`enemy_detection_rate`がほぼ1である。
6. PPO指標にNaN/Inf、持続的に大きいKL、行動の完全な単一化がない。

この条件を複数seedで満たしてから、GalaxianへのAdvantage転移比較へ進む。
