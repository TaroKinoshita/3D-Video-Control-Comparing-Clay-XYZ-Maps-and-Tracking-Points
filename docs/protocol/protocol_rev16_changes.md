# protocol rev.15 → rev.16 更新差分（W11、2026-11-01）— ⑤ RQ2（編集の漏れ）の事前登録

**実行前に凍結する。** 測り方・判定式・予測は、RQ2 の生成結果を見てから変えない。
RQ1 の結果（§42.5：3 ペアとも主張不可）は既知。RQ2 の判定式は、RQ1 と同じ形にそろえる。

---

## 変更1：RQ2 の測り方

**問い（§1）**：Target の物体軌道を反実仮想的に変えたとき、各表現は指定した編集をどこまで実現し、
Blender GT 上で不変な領域にどれだけ変化を漏らすか。

| 項目 | 決定 |
|---|---|
| LoRA | 各 arm の S1（`g1_arm{A,B,C}/step-500`）。学習はしない（予算 5 / 5 消費済み） |
| control | Scene C の Original（`arm{X}_original.mkv`）と Edited（`arm{X}_edited.mkv`）。Edited は index 12–23 で停止、24–35 で追いつく（§10 / §26） |
| seed | **G1〜G4**（Taro 決定、2026-11-01） |
| 生成 | 3 arm × 4 seed × 2 control = **24 本** |
| 参照画像・prompt | Phase 0.8 と同じ（Edited も frame 0 は同一なので参照画像は 1 枚のまま） |

**Edited を学習に入れていない**（§33.3）ので、Edited の control は反実仮想のまま。
Original の control は**学習動画そのもの**であり、gen_orig は学習動画の再生に近いはず。これは 3 arm 共通の条件。

---

## 変更2：代表指標 = spill_mae（Taro 決定、2026-11-01）

```
spill_mae = M_invariant 内の |gen_edit − gen_orig| の平均（8bit、RGB 3 チャンネル平均）
            index 13–35 の 23 フレームの M_invariant 画素をまとめた平均（pooled）
            同じ arm・同じ seed の 2 本で比べる
```

- **M_invariant**：§11 の定義どおり。`data/points/m_invariant_R0S0.bin`（`make_rq2_inputs.py`）。
  index 13–35 で平均 **379,672.7 px/frame（95.070%）**。§32.1 の照合値 379,673 を再現した
- 小さいほど漏れが少ない
- **報告のみ**：PSNR / p95 / index 0–12・36–48 の全画面差（時間方向の漏れの目安）/ 編集の実現度（gen_edit の Target 中心と Edited GT の median 誤差）/ 個数 ≠ 1 のフレーム
- LPIPS / SSIM / DINO は今回測らない（マスク付きの実装と追加モデルが必要。9 月以降）

### 採点器の検証（実行前に済ませた）

| 確認 | 結果 |
|---|---|
| GT 同士（tv original vs tv edited）の spill_mae = **F_eval_spill** | **0.0003** |
| 全画面に ±8 の一様ノイズを足した偽の生成 | 4.229（理論値 4.24） |
| 編集の実現度：tv edited を「生成」として採点 | median 3.27 px（Evaluator floor と同程度） |

---

## 変更3：入力側の編集信号の大きさ（§15「latent edit-signal magnitude」）

control の original と edited の差（index 13–35、画素空間）。

| arm | 変化した画素の割合 | 変化量 \|d\| の平均（全画素） |
|---|---|---|
| A | 3.20% | 1.872 |
| B | 2.33% | 1.498 |
| C | 1.65% | 2.877 |

**arm ごとに入力側の編集信号の大きさが違う。** 面積では A > B > C、平均変化量では C > A > B。
§15 に従い、spill_mae の差が出ても、それを表現の性質だけで説明しない。
spill_mae の順位が、入力の編集信号のどちらかの順位と同じ場合は、そのことを必ず併記する。

**限界**：§15 が言う「latent」の編集信号（VAE エンコード後の差）は今回測っていない。画素空間の値で代用する。

---

## 変更4：Claim Gate（RQ2）

RQ1（§42.3）と同じ形にする。

```
arm の代表値   s_X = G1〜G4 の spill_mae の平均
床            F_seed_spill = 各 arm の G1〜G4 の spill_mae の（最大 − 最小）、その 3 arm の最大
              F_eval_spill = 0.0003
              F_det = 0（RQ1 で再生成が bit 一致することを確認済み。§42.5）
判定          |s_X − s_Y| > max(F_seed_spill, F_eval_spill, F_det) のときだけ主張する
比較          Primary = B vs C、Secondary = A vs B、A vs C
```

- **破綻した生成も除外しない**（RQ1 と同じ。結果を見てから除外規則を作らない）
- Training-run 変動は測れない（予算切れ）。**F_train を含まないことを限界として書く**
- 編集の実現度は報告のみ。漏れが少なくても編集が実現していなければ意味が薄いので、併記する

---

## 変更5：予測（実行前に記録）

| # | 予測 | 根拠 |
|---|---|---|
| Q1 | 全 12 ペアで spill_mae は 0.5〜6（8bit） | 拡散は clip 全体を一度に作るので、不変領域も少しは動く。ただし背景は静止で control も同一 |
| Q2 | index 0–12 / 36–48 の全画面差（outside_mae）も 0 にはならない（0.3 以上） | 同上。control が同じフレームでも、clip 全体の生成が変わる |
| Q3 | 順位は **C が最大**、A と B は近い | Arm C は背景を 168 点の疎な点でしか拘束しない。A / B は全画素を拘束する |
| Q4 | 編集の実現度：A / B の median ≤ 8 px、C ≤ 12 px（RQ1 の C′ と同じ傾向） | RQ1 で 3 arm とも control に追従した |
| Q5 | Claim Gate：**B vs C と A vs C は主張可**、A vs B は主張不可 | Q3 の順位と、seed のブレが RQ1 の A / B 程度（0.4〜1.1）なら差が超える |
| Q6 | 生成が破綻する（Target の複製など）ペアが 12 ペア中 0〜1 | RQ1 では 12 本中 1 本 |

---

## 実行（Pod）

`scripts/pod/run_rq2.sh`。生成 24 本、約 1 時間。学習 run は使わない。
