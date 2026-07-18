# qwen standalone full-dev teacher fidelity

This table is unpaired evidence and must not be used as an 8x2 teacher comparison.

| Dataset | n | cF1 | Exact F1 | Aspect F1 | Opinion F1 | Category F1 | VA MAE | VA RMSE | Empty | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eng_laptop | 200 | 0.3170 | 0.3608 | 0.8631 | 0.7251 | 0.5506 | 0.8579 | 1.0702 | 2.00% | 0.00% |
| eng_restaurant | 200 | 0.3638 | 0.4026 | 0.8961 | 0.6839 | 0.5882 | 0.6756 | 0.9093 | 0.50% | 0.00% |
| jpn_hotel | 200 | 0.2004 | 0.2176 | 0.6204 | 0.4709 | 0.5497 | 0.5679 | 0.7208 | 5.00% | 0.00% |
| rus_restaurant | 48 | 0.3649 | 0.4080 | 0.7024 | 0.6129 | 0.8522 | 0.7813 | 1.0171 | 2.08% | 0.00% |
| tat_restaurant | 48 | 0.3954 | 0.4300 | 0.6463 | 0.6277 | 0.7788 | 0.6029 | 0.7172 | 8.33% | 0.00% |
| ukr_restaurant | 48 | 0.3598 | 0.3959 | 0.6951 | 0.6011 | 0.8000 | 0.6901 | 0.8111 | 6.25% | 0.00% |
| zho_laptop | 300 | 0.2374 | 0.2565 | 0.5538 | 0.4388 | 0.5862 | 0.5328 | 0.7224 | 5.00% | 0.00% |
| zho_restaurant | 300 | 0.4060 | 0.4347 | 0.6828 | 0.5799 | 0.7756 | 0.4746 | 0.6138 | 1.67% | 0.00% |

Full gold-dev labels were hidden from the teacher; few-shot examples came from train only.
