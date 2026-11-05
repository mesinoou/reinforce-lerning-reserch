# Space Invadersの物体チャネルと画像CNNの比較

`pixels_256`と`object_channels_256`を、同じPPO設定で各1,400万環境ステップ学習する。画像は84×84のグレースケール4時点、物体マップは5種類×4時点の20チャネル。物体チャネル順は `Player, Shield, Bullet, Satellite, Alien` で、履歴は古い時点から連結する。物体の外接矩形を画面内に収め、開始端を切り下げ、終了端を切り上げて二値マップにする。

## Linux研究用PCで実行

リポジトリを開き、学習に使うvenvを有効にする。下の各コマンドは1行ずつ実行する。

```bash
# 計画の確認
python run_all_objects.py --conditions pixels_256 object_channels_256 --gpus 0,1 --seeds 0 --total-steps 14000000 --output-root object_channels_vs_pixels_14m --dry-run

# 各2,048ステップでGPU動作・評価・出力を確認
python run_all_objects.py --conditions pixels_256 object_channels_256 --gpus 0,1 --seeds 0 --smoke --status-interval 5 --output-root object_channels_vs_pixels_smoke

# 各1,400万ステップの本比較
python run_all_objects.py --conditions pixels_256 object_channels_256 --gpus 0,1 --seeds 0 --total-steps 14000000 --output-root object_channels_vs_pixels_14m
```

中断後は同じコード、出力先、条件、seed、総ステップ数で本実験コマンドに `--resume` を付ける。新しいseedを含む実験計画は別の出力先を使う。各GPUの進捗とETAは親プロセスに表示され、詳しいログは各runの `console.log` に保存される。メモリはロールアウト観測配列だけで画像約110 MiB、物体チャネル約551 MiBで、GPU上にも観測テンソルが作られる。試走で両条件の速度とGPUメモリを確認する。

## 集計と再生

本実験完了後の自動集計は `comparison/` に保存される。主指標は0〜200万stepの正規化AUC。全区間AUC、1,000万〜1,400万stepの区間平均、最終評価得点、処理速度も出力する。報酬は確率的方策の固定seed評価の元報酬を使い、差は物体チャネルから画像を引く。`per_seed.csv` は条件別、`paired_differences.csv` は対応seedの差、`evaluation_curve.csv` は描画用の曲線を収録する。

```bash
python summarize_input_hypotheses.py --root object_channels_vs_pixels_14m --partial
python render_trained_agent.py --model object_channels_vs_pixels_14m/object_channels_256/seed_0/model.pt --device cuda --episodes 3 --policy deterministic --seed 100000 --output-dir object_channels_vs_pixels_14m/object_channels_256/seed_0/playback_final
```

途中集計は両条件に揃った評価点だけを使用し、`comparison_partial/` に保存する。物体マップの確認図は物体チャネルrunの `object_detection_snapshot.png` に保存される。試走と単一seedの比較は、それぞれ動作確認と予備的な観測として扱う。
