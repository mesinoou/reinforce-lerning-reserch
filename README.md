# Reinforcement Learning Research

強化学習とゲーム間転移に関する研究実装を共有するリポジトリです。

現在の中心課題は、Space Invadersで画像入力とOCAtari RAM Extraction
Modeの物体リスト入力を同条件比較し、その後GalaxianへのAdvantage転移を
公平に評価することです。

## 現在の実験

- ソース環境: `ALE/SpaceInvaders-v5`
- 入力A: OCAtari REM、91次元/フレーム × 4 = 364次元
- 入力B: 84×84グレースケール × 4フレーム
- 学習アルゴリズム: PPO
- 本学習環境: Linux + CUDA

実行方法、報酬条件、出力、Baseline成立基準については
[README_SOURCE_TRAINING.md](README_SOURCE_TRAINING.md)を参照してください。

主要ファイル:

- `ocatari_source_ppo.py`: 転移前のソースタスク単独学習
- `run_source_seeds.sh`: Linux上での複数seed実行
- `run_representation_comparison.sh`: 2入力条件の対応seed実行
- `compare_representations.py`: 物体・画像条件の対応比較とグラフ生成
- `benchmark_source_training.py`: 学習時間とstepsの実測・換算
- `render_trained_agent.py`: 最終／最良モデルのMP4動画・プレイ画像生成
- `aggregate_source_runs.py`: seed間集計
- `test_ocatari_source_ppo.py`: 単体テスト
- `ocatari_transfer_full_experiment.py`: 既存の一括転移実験

学習中は各runの`progress.json`から進捗率、処理速度、残りsteps、ETA、
推定終了時刻を確認できます。
