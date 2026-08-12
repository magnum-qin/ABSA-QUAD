# Bootstrap multiplicity robustness

Post-hoc inference only; official-test choices remain frozen.

## teacher_fidelity_8x2

| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |
|---|---:|---:|---:|---:|---|---|
| eng_laptop | +0.0404 | 0.0883 | 0.4415 | 0.1766 | inconclusive | no |
| eng_restaurant | +0.1892 | 0.0001 | 0.0008 | 0.0008 | supported | yes |
| jpn_hotel | +0.0142 | 0.5557 | 1.0000 | 0.6124 | inconclusive | no |
| rus_restaurant | +0.0255 | 0.5116 | 1.0000 | 0.6124 | inconclusive | no |
| tat_restaurant | +0.0405 | 0.4647 | 1.0000 | 0.6124 | inconclusive | no |
| ukr_restaurant | +0.0188 | 0.6124 | 1.0000 | 0.6124 | inconclusive | no |
| zho_laptop | +0.0447 | 0.0051 | 0.0306 | 0.0136 | supported | yes |
| zho_restaurant | +0.0593 | 0.0013 | 0.0091 | 0.0052 | supported | yes |

## adopted_kd_vs_baseline_dev

| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |
|---|---:|---:|---:|---:|---|---|
| eng_laptop | +0.1058 | 0.0001 | 0.0005 | 0.0005 | supported | yes |
| rus_restaurant | +0.0895 | 0.0141 | 0.0564 | 0.0352 | supported | no |
| tat_restaurant | +0.0506 | 0.2416 | 0.7247 | 0.4026 | inconclusive | no |
| ukr_restaurant | +0.0472 | 0.3297 | 0.7247 | 0.4121 | inconclusive | no |
| zho_restaurant | +0.0069 | 0.6220 | 0.7247 | 0.6220 | inconclusive | no |

## exposure_control_dev

| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |
|---|---:|---:|---:|---:|---|---|
| eng_laptop | +0.0948 | 0.0002 | 0.0010 | 0.0010 | supported | yes |
| rus_restaurant | +0.0924 | 0.0075 | 0.0300 | 0.0187 | supported | yes |
| tat_restaurant | +0.0071 | 0.8388 | 0.8388 | 0.8388 | inconclusive | no |
| ukr_restaurant | +0.0736 | 0.0514 | 0.1542 | 0.0857 | supported | no |
| zho_restaurant | +0.0260 | 0.0770 | 0.1542 | 0.0962 | inconclusive | no |

## same_text_self_training_dev

| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |
|---|---:|---:|---:|---:|---|---|
| eng_laptop | +0.0933 | 0.0003 | 0.0015 | 0.0015 | supported | yes |
| rus_restaurant | +0.1183 | 0.0009 | 0.0036 | 0.0022 | supported | yes |
| tat_restaurant | +0.0569 | 0.1620 | 0.4860 | 0.2294 | inconclusive | no |
| ukr_restaurant | +0.0438 | 0.1835 | 0.4860 | 0.2294 | inconclusive | no |
| zho_restaurant | -0.0016 | 0.9082 | 0.9082 | 0.9082 | inconclusive | no |

## frozen_official_test_posthoc

| Dataset | ΔcF1 | Raw p | Holm p | BH p | Raw CI decision | Holm < .05 |
|---|---:|---:|---:|---:|---|---|
| eng_laptop | +0.0822 | 0.0001 | 0.0005 | 0.0005 | supported | yes |
| rus_restaurant | +0.0211 | 0.0515 | 0.1030 | 0.0644 | supported | no |
| tat_restaurant | +0.0274 | 0.0087 | 0.0261 | 0.0145 | supported | yes |
| ukr_restaurant | +0.0393 | 0.0016 | 0.0064 | 0.0040 | supported | yes |
| zho_restaurant | -0.0137 | 0.0744 | 0.1030 | 0.0744 | inconclusive | no |
