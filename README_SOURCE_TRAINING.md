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

## 学習進捗と残り時間

学習中は既定で10 PPO updatesごと、または前回表示から60秒経過時に、
次の情報を標準出力へ表示する。

```text
[progress] 42.60% steps=426,000/1,000,000 remaining=574,000
updates=420 episodes=812 speed=153.2 steps/s
elapsed=00:46:21 eta=01:02:27 finish=2026-07-28T23:14:00+09:00
```

表示間隔は変更できる。

```bash
python ocatari_source_ppo.py \
  --progress-interval-updates 5 \
  --progress-interval-seconds 30 \
  ...
```

各出力ディレクトリの`progress.json`は実行中も更新される。別terminalから
次のように確認できる。

```bash
watch -n 5 cat source_baseline_runs/seed_0/progress.json
```

`progress.csv`には履歴を保存する。主な項目:

- 現在steps／総steps／残りsteps
- 進捗率
- PPO update数、完了episode数
- 今回のprocessの経過時間
- 平均・直近steps/秒
- ETAと推定終了日時
- 直近100 episodesの平均raw return

resume時はcheckpointまでのstepsを引き継ぎ、ETAは再開processで実測した
速度から再計算する。

## 学習時間ベンチマーク

`benchmark_source_training.py`は実際のREM抽出、物体中心入力変換、方策推論、
GAE、PPO更新、メモリ上の診断処理を一定時間実行する。その実測速度から、
指定時間で学習可能なenvironment stepsと、目標stepsの所要時間を計算する。

Linux研究用PCでは本学習と同じCUDA・PPO条件で実行する。

```bash
python benchmark_source_training.py \
  --device cuda \
  --benchmark-seconds 120 \
  --warmup-updates 2 \
  --rollout-steps 1024 \
  --ppo-epochs 4 \
  --minibatch-size 256 \
  --reward-mode scaled_raw \
  --reward-scale 0.1 \
  --project-hours 1 6 12 24 48 \
  --target-steps 500000 1000000 5000000 \
  --output benchmark_results/source_cuda.json
```

ローカル動作確認:

```bash
python benchmark_source_training.py \
  --quick \
  --device cpu \
  --output benchmark_results/smoke.json
```

出力JSONには次を保存する。

- environment steps/秒、ALE frames/秒
- rollout収集時間とPPO最適化時間
- 指定時間ごとの予測steps
- 指定stepsごとの予測時間
- CPU/GPU、ライブラリ版、全ベンチマーク条件

定期評価、checkpoint、グラフなどのディスクI/Oは測定から除外する。その
オーバーヘッド用に既定で10%を予約し、実測値に加えてconservative予測を
出力する。実際の学習計画にはconservative側を使用する。

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

## 学習済みモデルのプレイ画像

`render_trained_agent.py`は最終モデル、最良モデル、最新checkpointのいずれも
読み込める。

```text
model.pt
checkpoint_best.pt
checkpoint_latest.pt
```

最終モデルをdeterministic方策で再生する例:

```bash
python render_trained_agent.py \
  --model source_baseline_runs/seed_0/model.pt \
  --episodes 1 \
  --seed 100000 \
  --policy deterministic \
  --device cpu \
  --capture-every 4 \
  --max-representative-frames 20 \
  --columns 4 \
  --scale 2 \
  --output-dir source_baseline_runs/seed_0/playback_final
```

定期評価で最良だったモデル:

```bash
python render_trained_agent.py \
  --model source_baseline_runs/seed_0/checkpoint_best.pt \
  --episodes 3 \
  --policy deterministic \
  --output-dir source_baseline_runs/seed_0/playback_best
```

出力:

- `episode_001_contact_sheet.png`: episode全体の時系列コンタクトシート
- `episode_001_final.png`: episode終了時の画面
- `episode_001_frames/*.png`: episode全体から抽出した代表フレーム
- `episode_001_trajectory.csv`: action、raw reward、Value、方策確率、life
- `playback_summary.json`: モデルSHA-256、seed、方策、return、再生条件

既定ではREM物体の枠とカテゴリ名を重ねる。ゲーム画面だけを保存する場合:

```bash
python render_trained_agent.py \
  --model source_baseline_runs/seed_0/model.pt \
  --no-overlay-objects
```

`deterministic`は各状態で最大確率の行動を選ぶため、固定seedで比較画像を
作る用途に適している。学習時のように方策分布からサンプルする場合は
`--policy stochastic`を使用する。

最終`model.pt`にはモデル重みとencoder設定が含まれ、同じディレクトリの
`config.json`からREM、frameskip、最大episode長などを復元する。
`checkpoint_best/latest.pt`は必要な実行設定もcheckpoint内に保持している。

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
- `progress.json`: 実行中の最新進捗、速度、ETA
- `progress.csv`: 進捗履歴
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
