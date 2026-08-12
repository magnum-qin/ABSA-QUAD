# KD cost and efficiency inventory

Token usage and monetary billing were not retained historically; no cost is fabricated.

| Provider | Validated rows | Timed rows | Errors | Summed timed wall (s) | Timed rows/s | Token/cost status |
|---|---:|---:|---:|---:|---:|---|
| deepseek | 1344 | 1344 | 0 | 225.8 | 5.95 | unavailable |
| qwen | 1344 | 1264 | 0 | 1021.0 | 1.24 | unavailable |

| Dataset | Teacher | Silver rows | Dev ΔcF1 | Descriptive Δ per 1k silver |
|---|---|---:|---:|---:|
| eng_laptop | deepseek | 7547 | +0.1058 | +0.0140 |
| rus_restaurant | deepseek | 3000 | +0.0895 | +0.0298 |
| tat_restaurant | qwen | 3915 | +0.0506 | +0.0129 |
| ukr_restaurant | deepseek | 5534 | +0.0472 | +0.0085 |
| zho_restaurant | deepseek | 3882 | +0.0069 | +0.0018 |

| Dataset | System | Encoder | Parameters | 5-seed run-wall hours | Retained best GiB |
|---|---|---|---:|---:|---:|
| eng_laptop | gold_repeat | microsoft/deberta-v3-base | 202771798 | 13.41 | 14.91 |
| rus_restaurant | gold_repeat | xlm-roberta-large | 592918508 | 10.86 | 43.80 |
| tat_restaurant | gold_repeat | xlm-roberta-large | 592918508 | 15.28 | 43.80 |
| ukr_restaurant | gold_repeat | microsoft/mdeberta-v3-base | 297125356 | 10.97 | 21.94 |
| zho_restaurant | gold_repeat | hfl/chinese-roberta-wwm-ext-large | 358550508 | 13.53 | 26.33 |
| eng_laptop | self_training | microsoft/deberta-v3-base | 202771798 | 18.34 | 14.91 |
| rus_restaurant | self_training | xlm-roberta-large | 592918508 | 11.37 | 43.80 |
| zho_restaurant | self_training | hfl/chinese-roberta-wwm-ext-large | 358550508 | 16.43 | 26.33 |
| tat_restaurant | self_training | xlm-roberta-large | 592918508 | 13.93 | 43.80 |
| ukr_restaurant | self_training | microsoft/mdeberta-v3-base | 297125356 | 10.85 | 21.94 |
