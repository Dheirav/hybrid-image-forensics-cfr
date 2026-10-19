## 1. Claimed against stored (primary metric, test split, authentic + splice + copy_move)

| variant | claimed | recomputed mean over seeds | per seed | difference |
|---|---|---|---|---|
| single_a | 0.9936 | 0.9936 | 0.9966, 0.9931, 0.9912 | +0.0000 |
| single_b | 0.9108 | 0.9108 | 0.9141, 0.9069, 0.9115 | +0.0000 |
| joint_concat | 0.9929 | 0.9929 | 0.9903, 0.9953, 0.9930 | -0.0000 |
| ml | 0.9791 | 0.9791 | 0.9793, 0.9812, 0.9767 | -0.0000 |
| ml_hem | 0.9844 | 0.9844 | 0.9844, 0.9855, 0.9832 | -0.0000 |
| cfr_adaptive | 0.9824 | 0.9824 | 0.9817, 0.9835, 0.9821 | +0.0000 |

## 2. External numbers (data/external, zero-shot; a different dataset, not comparable with section 1)

| quantity | ai_edited vs real | ai_generated vs real |
|---|---|---|
| brief | 0.554 | 0.594 |
| cfr_adaptive, mean of per-seed AUROC | 0.552 | 0.591 |
| cfr_adaptive, AUROC of seed-averaged scores | 0.554 | 0.594 |

The brief's figures are the seed-averaged-score AUROC; the per-seed mean is 0.002 to 0.003 lower. Same model, same images, different averaging.

## 3. Second code path (eval_manifest.forward_all) against stored probabilities, COCO-Forge test split

| run | primary AUROC stored | primary AUROC re-scored | max abs prob difference | label flips at 0.5 |
|---|---|---|---|---|
| runs/single_a_s0 | 0.9966 | 0.9966 | 0.0000 | 0 of 500 |
| runs/single_a_s1 | 0.9931 | 0.9931 | 0.0000 | 0 of 500 |
| runs/single_a_s2 | 0.9912 | 0.9912 | 0.0000 | 0 of 500 |
| runs/cfr_adaptive_s0 | 0.9817 | 0.9817 | 0.0000 | 0 of 500 |
| runs_extra/joint_concat_s0 | 0.9903 | 0.9903 | 0.0000 | 0 of 500 |

## 4. Determinism: single_a seed 0 retrained

primary AUROC original 0.9966, retrained 0.9961, difference -0.0005 (seed sd across seeds was 0.0027)

validation AUROC per epoch

| epoch | original | retrained |
|---|---|---|
| 1 | 0.9225 | 0.9225 |
| 2 | 0.9642 | 0.9616 |
| 3 | 0.9647 | 0.9670 |
| 4 | 0.9706 | 0.9693 |
| 5 | 0.9813 | 0.9802 |
| 6 | 0.9827 | 0.9814 |
| 7 | 0.9852 | 0.9816 |
| 8 | 0.9860 | 0.9857 |
| 9 | 0.9846 | 0.9858 |
| 10 | 0.9859 | 0.9869 |
| 11 | 0.9871 | 0.9872 |
| 12 | 0.9869 | 0.9875 |
