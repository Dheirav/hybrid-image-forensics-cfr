## 1. Do external real photos look different to the models than COCO-Forge authentic crops?

One 384 px window per image (no tiling, so the numbers are comparable). COCO-Forge authentic = test split. Threshold 0.5.

| variant | FPR external real | FPR COCO-Forge authentic | mean score external real | mean score COCO-Forge authentic | central-window AUROC edited | central-window AUROC generated |
|---|---|---|---|---|---|---|
| single_a | 0.025 | 0.029 | 0.029 | 0.034 | 0.6 | 0.583 |
| ml_hem | 0.044 | 0.029 | 0.249 | 0.239 | 0.546 | 0.593 |
| cfr_adaptive | 0.062 | 0.053 | 0.218 | 0.193 | 0.565 | 0.611 |

## 2. Is the real/fake information in the frozen embeddings? Leave-one-generator-out probes on the central 384 px window

Mean AUROC over held-out generators and the two real-image halves (sd in brackets). Chance is 0.5. Trained head = the model's own output on the same window, for reference.

| features | ai_edited probe AUROC | ai_generated probe AUROC |
|---|---|---|
| ImageNet EfficientNet-B0, never fine-tuned | 0.929 (0.008) | 0.892 (0.046) |
| Model A trained alone, seed 0 | 0.811 (0.015) | 0.824 (0.028) |
| Model A trained alone, seed 1 | 0.811 (0.029) | 0.796 (0.045) |
| Model A trained alone, seed 2 | 0.807 (0.009) | 0.811 (0.060) |
| Model B trained alone, seed 0 | 0.639 (0.021) | 0.677 (0.055) |
| A seed 0 + B seed 0 (concatenated) | 0.814 (0.009) | 0.818 (0.027) |
| trained head: single_a (3 seeds, central window) | 0.600 | 0.583 |
| trained head: cfr_adaptive (3 seeds, central window) | 0.565 | 0.611 |

## 3. AI-edited images by size of the edited region (tiled score, seed-averaged probabilities)

Terciles of mask area: small 0.010 to 0.054, medium 0.055 to 0.175, large 0.177 to 0.893 of the frame.

| variant | edit size | n | AUROC vs real | recall at 0.5 |
|---|---|---|---|---|
| single_a | small | 133 | 0.608 | 0.053 |
| single_a | medium | 133 | 0.61 | 0.083 |
| single_a | large | 134 | 0.589 | 0.052 |
| cfr_adaptive | small | 133 | 0.545 | 0.105 |
| cfr_adaptive | medium | 133 | 0.565 | 0.158 |
| cfr_adaptive | large | 134 | 0.552 | 0.164 |

## 4. Tiled max over windows against one central window (3 seed mean AUROC)

| variant | tiled max, ai_edited | tiled max, ai_generated | tiled max, FPR real | central window, ai_edited | central window, ai_generated | central window, FPR real |
|---|---|---|---|---|---|---|
| single_a | 0.593 | 0.566 | 0.047 | 0.6 | 0.583 | 0.025 |
| cfr_adaptive | 0.552 | 0.591 | 0.132 | 0.565 | 0.611 | 0.062 |
