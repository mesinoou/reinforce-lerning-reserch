# Changelog

## 2026-07-28

### Source-task baseline

- 転移を実行しない`ocatari_source_ppo.py`を追加。
- OCAtariの既定モードをVEMからREM（`ram`）へ変更。
- 364次元の共通入力を維持しつつ、距離順位の入れ替わりを抑える
  `temporal`物体スロットを追加。旧`distance`方式も再現用に残した。
- `raw`、`clipped`、`scaled_raw`、`scaled_survival`、`shaped`の
  報酬モードを追加。
- raw returnを学習報酬と独立して常時記録。
- 線形学習率減衰、value clipping、target KL早期終了を追加。
- clip fraction、explained variance、gradient norm、entropy、value統計、
  Advantage統計、行動割合を記録。
- life lossのGAE terminal化、最大エピソード長、FIRE resetを個別設定化。
- random、初期、定期、最終についてstochastic/deterministic評価を追加。
- 物体カテゴリ、検出率、入力次元統計、人間確認用サンプルを追加。
- 定期checkpoint、best checkpoint、乱数状態を含むresume機能を追加。
- ヘッドレスLinux向けにMatplotlibの`Agg`バックエンドを使用。
- Linux用3 seeds実行スクリプトとソース学習手順書を追加。
- 複数seedの評価曲線、random/初期/最終差、簡易成立基準をまとめる
  `aggregate_source_runs.py`を追加。
- Python 3.14のNumPy非互換を避けるため、対応環境をPython 3.10～3.12、
  NumPy 1.26.4として明記。

### Reward stabilization

- raw rewardを一律`0.1`倍する`scaled_raw`を既定条件へ変更。
- 実際に進んだALEフレーム数に基づく`scaled_survival`を追加。
- 生存報酬の保守的な初期値を`0.001/ALE frame`、life lossペナルティを
  `-1.0`とした。
- 旧environment-step単位の`shaped`報酬は再現用として維持。
- episodeごとにscore、生存、life lossの報酬成分を分離して記録。
- PPO更新ごとにraw rewardとtraining rewardの平均・標準偏差を記録し、
  スケーリング効果と生存報酬の支配を診断可能にした。

### Progress and throughput benchmark

- 学習中に進捗率、現在・残りsteps、平均速度、経過時間、ETA、推定終了日時を
  定期表示する機能を追加。
- 実行中に安全に読み取れる`progress.json`と履歴用`progress.csv`を追加。
- resume後は残りstepsと再開processの実測速度からETAを再計算。
- REM抽出、物体変換、方策推論、GAE、PPO更新、診断処理を含めて実測する
  `benchmark_source_training.py`を追加。
- 指定時間から予測stepsを、目標stepsから予測所要時間を計算。
- 評価・checkpoint等の除外時間を考慮するconservative予測を追加。
