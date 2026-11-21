# Results summary

Docs: 121. Best method (relaxed F1, no verification): **zero_shot_json**.

## Main results

| method | n_pred | parse_ok | halluc_rate | P_strict | R_strict | F1_strict [95% CI] | P_relaxed | R_relaxed | F1_relaxed [95% CI] | P_value_unit | R_value_unit | F1_value_unit [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_text | 214 | 1.00 | 0.514 | 0.037 | 0.041 | 0.039 [0.010,0.074] | 0.079 | 0.088 | 0.083 [0.034,0.143] | 0.126 | 0.139 | 0.132 [0.074,0.202] |
| zero_shot_json | 194 | 0.96 | 0.613 | 0.964 | 0.964 | 0.964 [0.933,0.988] | 0.969 | 0.969 | 0.969 [0.939,0.992] | 0.974 | 0.974 | 0.974 [0.946,0.995] |
| few_shot_json | 162 | 0.94 | 0.722 | 0.148 | 0.124 | 0.135 [0.084,0.190] | 0.228 | 0.191 | 0.208 [0.142,0.272] | 0.259 | 0.216 | 0.236 [0.162,0.314] |
| rag_json | 149 | 0.96 | 0.698 | 0.134 | 0.103 | 0.117 [0.065,0.174] | 0.248 | 0.191 | 0.216 [0.143,0.296] | 0.302 | 0.232 | 0.262 [0.187,0.343] |
| lora_json | 12 | 0.99 | 1.000 | 0.417 | 0.026 | 0.049 [0.000,0.111] | 0.500 | 0.031 | 0.058 [0.009,0.122] | 0.583 | 0.036 | 0.068 [0.018,0.132] |
| regex_baseline | 364 | 1.00 | 0.005 | 0.000 | 0.000 | 0.000 [0.000,0.000] | 0.000 | 0.000 | 0.000 [0.000,0.000] | 0.041 | 0.077 | 0.054 [0.021,0.095] |
| zero_shot_json+verify | 48 | 0.96 | 0.000 | 1.000 | 0.247 | 0.397 [0.285,0.494] | 1.000 | 0.247 | 0.397 [0.285,0.494] | 1.000 | 0.247 | 0.397 [0.285,0.494] |
| few_shot_json+verify | 31 | 0.94 | 0.000 | 0.194 | 0.031 | 0.053 [0.010,0.100] | 0.355 | 0.057 | 0.098 [0.030,0.171] | 0.419 | 0.067 | 0.116 [0.038,0.203] |
| rag_json+verify | 36 | 0.96 | 0.000 | 0.222 | 0.041 | 0.070 [0.025,0.124] | 0.417 | 0.077 | 0.130 [0.059,0.214] | 0.444 | 0.082 | 0.139 [0.067,0.224] |
| lora_json+verify | 0 | 0.99 | 0.000 | 0.000 | 0.000 | 0.000 [0.000,0.000] | 0.000 | 0.000 | 0.000 [0.000,0.000] | 0.000 | 0.000 | 0.000 [0.000,0.000] |

## Calibration (verbalised confidence)

| method | n | ECE | Brier | mean_conf | accuracy |
|---|---|---|---|---|---|
| zero_shot_json | 194 | 0.007 | 0.030 | 0.98 | 0.97 |
| few_shot_json | 162 | 0.672 | 0.627 | 0.90 | 0.23 |
| rag_json | 149 | 0.653 | 0.612 | 0.90 | 0.25 |

## Error taxonomy (relaxed match)

| method | hallucinated_value | hallucinated_unit | wrong_unit | wrong_material | wrong_property | spurious_claim | missed_claim | field_error |
|---|---|---|---|---|---|---|---|---|
| zero_shot_text | 80 | 28 | 5 | 6 | 2 | 76 | 154 | 23 |
| zero_shot_json | 5 | 0 | 0 | 0 | 0 | 1 | 5 | 1 |
| few_shot_json | 97 | 0 | 5 | 0 | 2 | 21 | 135 | 22 |
| rag_json | 89 | 0 | 5 | 1 | 1 | 16 | 138 | 19 |
| lora_json | 6 | 0 | 0 | 0 | 0 | 0 | 187 | 1 |
| regex_baseline | 2 | 0 | 3 | 15 | 0 | 344 | 176 | 18 |
| zero_shot_json+verify | 0 | 0 | 0 | 0 | 0 | 0 | 145 | 1 |
| few_shot_json+verify | 0 | 0 | 4 | 0 | 2 | 14 | 177 | 6 |
| rag_json+verify | 0 | 0 | 5 | 1 | 0 | 15 | 172 | 7 |
| lora_json+verify | 0 | 0 | 0 | 0 | 0 | 0 | 194 | 0 |

## Paired bootstrap on relaxed F1

| A | B | F1(A)-F1(B) | 95% CI | p |
|---|---|---|---|---|
| zero_shot_json | zero_shot_text | +0.886 | [+0.821,+0.940] | 0.000 |
| zero_shot_json | zero_shot_json+verify | +0.572 | [+0.471,+0.684] | 0.000 |
| few_shot_json | zero_shot_text | +0.125 | [+0.029,+0.212] | 0.009 |
| few_shot_json | zero_shot_json | -0.761 | [-0.823,-0.694] | 0.000 |
| few_shot_json | rag_json | -0.008 | [-0.101,+0.084] | 0.888 |
| few_shot_json | lora_json | +0.150 | [+0.064,+0.236] | 0.000 |
| few_shot_json | regex_baseline | +0.208 | [+0.146,+0.273] | 0.000 |
| few_shot_json | zero_shot_json+verify | -0.189 | [-0.293,-0.074] | 0.002 |
| few_shot_json | few_shot_json+verify | +0.110 | [+0.051,+0.174] | 0.000 |
| few_shot_json | rag_json+verify | +0.077 | [-0.029,+0.172] | 0.136 |
| few_shot_json | lora_json+verify | +0.208 | [+0.146,+0.273] | 0.000 |
| rag_json | zero_shot_text | +0.132 | [+0.055,+0.210] | 0.000 |
| rag_json | zero_shot_json | -0.753 | [-0.829,-0.673] | 0.000 |
| rag_json | regex_baseline | +0.216 | [+0.144,+0.292] | 0.000 |
| rag_json | zero_shot_json+verify | -0.181 | [-0.282,-0.071] | 0.002 |
| rag_json | rag_json+verify | +0.085 | [+0.027,+0.149] | 0.002 |
| lora_json | zero_shot_text | -0.025 | [-0.107,+0.060] | 0.537 |
| lora_json | zero_shot_json | -0.911 | [-0.968,-0.843] | 0.000 |
| lora_json | rag_json | -0.157 | [-0.245,-0.066] | 0.001 |
| lora_json | regex_baseline | +0.058 | [+0.010,+0.122] | 0.038 |
| lora_json | zero_shot_json+verify | -0.338 | [-0.457,-0.205] | 0.000 |
| lora_json | rag_json+verify | -0.072 | [-0.173,+0.029] | 0.158 |
| lora_json | lora_json+verify | +0.058 | [+0.010,+0.122] | 0.038 |
| regex_baseline | zero_shot_text | -0.083 | [-0.144,-0.032] | 0.000 |
| regex_baseline | zero_shot_json | -0.969 | [-0.992,-0.939] | 0.000 |
| regex_baseline | zero_shot_json+verify | -0.397 | [-0.496,-0.283] | 0.000 |
| zero_shot_json+verify | zero_shot_text | +0.313 | [+0.210,+0.411] | 0.000 |
| few_shot_json+verify | zero_shot_text | +0.014 | [-0.076,+0.105] | 0.777 |
| few_shot_json+verify | zero_shot_json | -0.871 | [-0.941,-0.793] | 0.000 |
| few_shot_json+verify | rag_json | -0.118 | [-0.215,-0.016] | 0.018 |
| few_shot_json+verify | lora_json | +0.040 | [-0.053,+0.133] | 0.423 |
| few_shot_json+verify | regex_baseline | +0.098 | [+0.032,+0.172] | 0.001 |
| few_shot_json+verify | zero_shot_json+verify | -0.299 | [-0.390,-0.207] | 0.000 |
| few_shot_json+verify | rag_json+verify | -0.033 | [-0.134,+0.063] | 0.506 |
| few_shot_json+verify | lora_json+verify | +0.098 | [+0.032,+0.172] | 0.001 |
| rag_json+verify | zero_shot_text | +0.047 | [-0.025,+0.116] | 0.197 |
| rag_json+verify | zero_shot_json | -0.839 | [-0.914,-0.753] | 0.000 |
| rag_json+verify | regex_baseline | +0.130 | [+0.057,+0.213] | 0.000 |
| rag_json+verify | zero_shot_json+verify | -0.266 | [-0.355,-0.176] | 0.000 |
| lora_json+verify | zero_shot_text | -0.083 | [-0.144,-0.032] | 0.000 |
| lora_json+verify | zero_shot_json | -0.969 | [-0.992,-0.939] | 0.000 |
| lora_json+verify | rag_json | -0.216 | [-0.292,-0.144] | 0.000 |
| lora_json+verify | regex_baseline | +0.000 | [+0.000,+0.000] | 1.000 |
| lora_json+verify | zero_shot_json+verify | -0.397 | [-0.496,-0.283] | 0.000 |
| lora_json+verify | rag_json+verify | -0.130 | [-0.213,-0.057] | 0.000 |