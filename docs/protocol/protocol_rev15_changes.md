# protocol rev.14 → rev.15 更新差分（W10、2026-11-01）— ④ 床の測定の事前登録

**実行前に凍結する。** ここに書いた測り方・判定式・予測は、④の結果を見てから変えない。
変える場合は理由と日付を記録し、論文で事後変更として明記する（§0）。

前提：Phase 0.8 の結果（S1 / G1）は既に見ている。
T_control median は A 1.98 / B 3.89 / C 6.62 px（worklog W9）。
**この結果を見たあとに決めているのは、代表 arm と判定式である。** 下で理由を明記する。

---

## 変更1：Training-run 変動の代表 arm は Arm C（§9 / §15）

決定者：Taro（2026-11-01）。

**理由（結果と無関係なもの）**：Arm C は Phase 0.8 の結果が出る前（2026-08-10）から
「代表 1 arm」として偵察 run に使っていた arm である（§33.7 / rev.11）。
§9 の「Training 変動測定：代表 1 arm × S1, S2, S3」の代表は、結果を見る前に事実上決まっていた。
結果を見たあとに A や B を選ぶと、都合のよい arm を選んだ疑いが残る。

**限界**：Training 変動は Arm C でしか測らない。Arm A / B の変動が C と同程度だとは保証されない。
F_train を全 arm に当てはめるのは仮定であり、論文に書く。

---

## 変更2：④ の測り方（§15 の Pipeline floor 4 種 + Training 変動）

すべて Scene C′、step-500、tracker と GT は rev.14（`score_t_control_v2.py`）。
床はすべて **T_control median（px）の差** で表す。ゲートと同じ統計量に揃えるため。

| 床 | やること | 生成本数 | 床の値の定義 |
|---|---|---|---|
| **F_det** Determinism | Phase 0.8 と同条件（S1 / G1）で再生成。A / B を新規、C は W9 の試運転を流用 | 2 | 3 arm の \|median(再生成) − median(元)\| の最大 |
| **F_enc** Encoding | control をデコード → FFV1 再保存 → 再デコード | 0（GPU 不要） | 再保存で画素が 1 つも変わらなければ **F_enc = F_det**（DiT 入力が bit 一致するため。§36.6） |
| **F_micro** Micro-perturbation | control を右へ 1 px（左端列は複製）。G1 | 3 | 3 arm の \|median(1px) − median(G1)\| の最大 |
| **F_seed** Seed variability | G2 / G3 / G4（seed 1 / 2 / 3）。S1 の LoRA | 9 | 各 arm の G1〜G4 の median の（最大 − 最小）、その 3 arm の最大 |
| **F_train** Training-run 変動 | Arm C を S2 / S3 で学習し G1 で生成 | 学習 2 + 生成 2 | Arm C の S1〜S3 の median の（最大 − 最小） |
| **F_eval** Evaluator | 測定済み（§41.4）。C′ の正解動画そのものを採点 | 0 | **1.81 px**（median） |
| F_gpu 数値再現性 | **今回は測らない**。全工程を同じ GPU 型（RTX PRO 6000 Blackwell）で行うため | 0 | 記録のみ |

**F_enc の実測（W10、GPU 不要）**：3 arm とも再保存前後で content sha(B) が一致した（A `3666302824d68ee5` / B `5a08e10cc228d230` / C `fde80efbe48dcb51`）。よって F_enc = F_det。

1 px ずらした control（`make_floor_controls.py`、FFV1 bitexact）：

| arm | sha(B) | 変化した画素の割合 |
|---|---|---|
| A | `de9c8d09168c6a0b` | 9.17% |
| B | `23e1cd10325dff9a` | 11.51% |
| C | `be3181967aabeb3c` | 1.27% |

**注意**：1 px の摂動で変わる画素の割合は arm ごとに違う（C は点なので少ない）。
「同じ 1 px」でも入力の変化量は揃っていない。F_micro は arm 間の公平な比較ではなく、床の大きさの目安として使う。

---

## 変更3：Claim Gate の判定式（§15 の具体化）

**arm の代表値**：m_X = G1〜G4 の T_control median の平均（S1 の LoRA）。
G1 の 1 本だけを使わない理由：seed の偶然で 1 本だけ良い / 悪い場合に引きずられるため。

**判定**：arm X と arm Y の比較で、

```
|m_X − m_Y| > max(F_det, F_enc, F_micro, F_seed, F_train, F_eval)
```

を満たすときに限り「表現による差がある」と主張する。満たさなければ「この条件では差を主張できない」と報告する。

- 比較の対象：**Primary = B vs C**、Secondary = A vs B、A vs C（§2。Phase 0.8 の結果を見てから新しい比較を足さない）
- p95 と T_coherence は報告のみ。判定には使わない（§37 の案 6-C と同じ）
- 学習の変動を表すのは F_train の 1 つだけ。3 seed なので幅の推定は粗い。これも限界として書く

---

## 変更4：予測（実行前に記録。外れたら訂正を書く）

| # | 予測 | 根拠 |
|---|---|---|
| P1 | 再生成（A / B）は **bit 一致**し、F_det = 0 | Arm C の試運転が bit 一致した（W9）。同じ GPU 型・同じ seed |
| P2 | F_micro ≤ 1.5 px（3 arm とも \|Δmedian\| ≤ 1.5） | 1 px の摂動に素直に追従すれば中心が約 1 px 動くだけ |
| P3 | F_seed：A ≤ 1.5 px、B 1〜4 px、C 2〜6 px。3 arm の最大は C | Phase 0.8 で p95 が A 4.7 / B 16.2 / C 20.4 と、C ほど誤差の裾が広い |
| P4 | 12 本の生成すべてで T_control PASS（≤ 47 px） | Phase 0.8 の 3 本は最大でも 25.97 px |
| P5 | T_coherence は 12 本中 **1〜4 本で FAIL**（不良 3 本以上） | Phase 0.8 の 3 本がすべて上限ちょうどの 2 本だった |
| P6 | Arm C の S2 / S3 の median は 4〜10 px。F_train は F_seed（C）より大きい | 学習の乱数は LoRA 初期化とノイズを全部変えるので、seed だけより動くはず |
| P7 | Claim Gate：**A vs C は主張できる**。A vs B と B vs C は主張できない | Phase 0.8 の差（A–C 4.6、B–C 2.7、A–B 1.9）と、予想される床 2〜4 px の比較 |

---

## 実行（Pod）

`scripts/pod/run_phase4.sh`。生成 16 本 + 学習 2 本、約 3 時間。
学習 run 予算はこれで 5 / 5 を使い切る。
