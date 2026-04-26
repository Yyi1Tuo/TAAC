# Data Overview

## Raw Data

1000 samples, 120 columns, 1000 users, 837 items.

### Column Groups

| group | count |
| --- | --- |
| basic_feature | 3 |
| user_feature | 56 |
| item_feature | 14 |
| seq_feature | 45 |

---

## Feature Taxonomy

### Basic Features

| field | type_counter | unique_values | range | missing_prob | valid_prob |
| --- | --- | --- | --- | --- | --- |
| label_time | {'int_value': 1000} | 553 | [1772725027, 1772725910] | 0.000 | 1.000 |
| label_type | {'int_value': 1000} | 2 | [1, 2] | 0.000 | 1.000 |
| timestamp | {'int_value': 1000} | 501 | [1772725000, 1772725781] | 0.000 | 1.000 |

### Non-sequential Features

#### item_feature

| feat_id | field | type | unique_values | max_len | coverage | missing_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | item_int_feats_5 | int_value | 82 | 0 | 0.998 | 0.002 |  |
| 6 | item_int_feats_6 | int_value | 216 | 0 | 0.998 | 0.002 |  |
| 7 | item_int_feats_7 | int_value | 349 | 0 | 0.998 | 0.002 |  |
| 8 | item_int_feats_8 | int_value | 226 | 0 | 0.998 | 0.002 |  |
| 9 | item_int_feats_9 | int_value | 24 | 0 | 0.998 | 0.002 |  |
| 10 | item_int_feats_10 | int_value | 110 | 0 | 0.998 | 0.002 |  |
| 11 | item_int_feats_11 | int_array | 924 | 20 | 0.548 | 0.439 |  |
| 12 | item_int_feats_12 | int_value | 352 | 0 | 0.998 | 0.002 |  |
| 13 | item_int_feats_13 | int_value | 8 | 0 | 0.998 | 0.002 |  |
| 16 | item_int_feats_16 | int_value | 662 | 0 | 0.998 | 0.002 |  |
| 81 | item_int_feats_81 | int_value | 3 | 0 | 0.998 | 0.002 |  |
| 83 | item_int_feats_83 | int_value | 22 | 0 | 0.185 | 0.832 |  |
| 84 | item_int_feats_84 | int_value | 66 | 0 | 0.185 | 0.832 |  |
| 85 | item_int_feats_85 | int_value | 103 | 0 | 0.185 | 0.832 |  |

#### user_feature

**int_value 特征级 embedding**

| feat_id | field | unique_values | coverage | missing_prob | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | user_int_feats_1 | 3 | 1.000 | 0.000 |  |
| 3 | user_int_feats_3 | 341 | 0.970 | 0.030 |  |
| 4 | user_int_feats_4 | 268 | 0.970 | 0.030 |  |
| 48 | user_int_feats_48 | 52 | 0.998 | 0.002 |  |
| 49 | user_int_feats_49 | 2 | 0.993 | 0.007 |  |
| 50 | user_int_feats_50 | 2 | 0.996 | 0.004 |  |
| 51 | user_int_feats_51 | 5 | 0.999 | 0.001 |  |
| 52 | user_int_feats_52 | 36 | 0.999 | 0.001 |  |
| 53 | user_int_feats_53 | 264 | 0.999 | 0.001 |  |
| 54 | user_int_feats_54 | 462 | 0.632 | 0.368 |  |
| 55 | user_int_feats_55 | 13 | 0.981 | 0.019 |  |
| 56 | user_int_feats_56 | 405 | 0.981 | 0.019 |  |
| 57 | user_int_feats_57 | 105 | 0.969 | 0.031 |  |
| 58 | user_int_feats_58 | 2 | 0.850 | 0.150 |  |
| 59 | user_int_feats_59 | 8 | 0.850 | 0.150 |  |
| 82 | user_int_feats_82 | 23 | 0.796 | 0.204 |  |
| 86 | user_int_feats_86 | 61 | 0.308 | 0.692 |  |
| 92 | user_int_feats_92 | 2 | 0.506 | 0.494 |  |
| 93 | user_int_feats_93 | 36 | 0.829 | 0.171 |  |
| 94 | user_int_feats_94 | 6 | 0.479 | 0.521 |  |
| 95 | user_int_feats_95 | 3 | 0.682 | 0.318 |  |
| 96 | user_int_feats_96 | 3 | 0.322 | 0.678 |  |
| 97 | user_int_feats_97 | 3 | 0.708 | 0.292 |  |
| 98 | user_int_feats_98 | 3 | 0.897 | 0.103 |  |
| 99 | user_int_feats_99 | 2 | 0.188 | 0.812 |  |
| 100 | user_int_feats_100 | 2 | 0.155 | 0.845 |  |
| 101 | user_int_feats_101 | 2 | 0.090 | 0.910 |  |
| 102 | user_int_feats_102 | 2 | 0.123 | 0.877 |  |
| 103 | user_int_feats_103 | 3 | 0.138 | 0.862 |  |
| 104 | user_int_feats_104 | 3 | 0.628 | 0.372 |  |
| 105 | user_int_feats_105 | 3 | 0.691 | 0.309 |  |
| 106 | user_int_feats_106 | 3 | 0.840 | 0.160 |  |
| 107 | user_int_feats_107 | 2 | 0.700 | 0.300 |  |
| 108 | user_int_feats_108 | 6 | 0.484 | 0.516 |  |
| 109 | user_int_feats_109 | 7 | 0.146 | 0.854 |  |

**int_array 元素级 embedding**

| feat_id | field | unique_values | max_len | coverage | missing_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | user_int_feats_15 | 447 | 13 | 0.861 | 0.139 |  |
| 60 | user_int_feats_60 | 1 | 2 | 0.408 | 0.592 |  |
| 62 | user_int_feats_62 | 7 | 5 | 0.930 | 0.070 |  |
| 63 | user_int_feats_63 | 33 | 11 | 0.930 | 0.070 |  |
| 64 | user_int_feats_64 | 45 | 18 | 0.930 | 0.070 |  |
| 65 | user_int_feats_65 | 218 | 49 | 0.920 | 0.080 |  |
| 66 | user_int_feats_66 | 533 | 66 | 0.914 | 0.086 |  |
| 80 | user_int_feats_80 | 12 | 5 | 0.800 | 0.200 |  |
| 89 | user_int_feats_89 | 5 | 10 | 0.945 | 0.055 |  |
| 90 | user_int_feats_90 | 6 | 10 | 0.909 | 0.091 |  |
| 91 | user_int_feats_91 | 7 | 10 | 0.550 | 0.450 |  |

**float_array dense vector**

| feat_id | field | array_len | max_len | coverage | 值范围 | mean | missing_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 61 | user_dense_feats_61 | 256 | 256 | 0.998 | [-0.2510703206062317, 0.20129907131195068] | -0.001 | 0.002 |  |
| 62 | user_dense_feats_62 | None | 5 | 0.930 | [32.0, 4954893.0] | 136400.665 | 0.070 |  |
| 63 | user_dense_feats_63 | None | 11 | 0.930 | [6.0, 4860850.0] | 128705.326 | 0.070 |  |
| 64 | user_dense_feats_64 | None | 18 | 0.930 | [23.0, 12762341.0] | 121708.926 | 0.070 |  |
| 65 | user_dense_feats_65 | None | 49 | 0.920 | [5.0, 18401112.0] | 146881.663 | 0.080 |  |
| 66 | user_dense_feats_66 | None | 66 | 0.914 | [5.0, 18401112.0] | 198935.566 | 0.086 |  |
| 87 | user_dense_feats_87 | 320 | 320 | 0.985 | [-0.6841999888420105, 0.6844000220298767] | -0.004 | 0.015 |  |
| 89 | user_dense_feats_89 | 10 | 10 | 0.945 | [-0.7245000004768372, 0.9086999893188477] | -0.000 | 0.055 |  |
| 90 | user_dense_feats_90 | 10 | 10 | 0.909 | [-0.723800003528595, 0.8787999749183655] | -0.000 | 0.091 |  |
| 91 | user_dense_feats_91 | 10 | 10 | 0.550 | [-0.7804999947547913, 0.9075999855995178] | 0.000 | 0.450 |  |

### Sequential Features

| seq_name | feat数量 | feat_ids | 序列长度 | 备注 |
| --- | --- | --- | --- | --- |
| domain_a_seq | 9 | 38, 39, 40, 41, 42, 43, 44, 45, 46 | max=1888 |  |
| domain_b_seq | 14 | 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 88 | max=1952 |  |
| domain_c_seq | 12 | 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 47 | max=3894 |  |
| domain_d_seq | 10 | 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 | max=3951 |  |

时间戳特征 `timestamp=True` 标注 ⏱。


#### domain_a_seq

| feat_id | field | map_range | range | max_len | coverage | missing_prob | empty_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | domain_a_seq_38 | 17,776 | [-1, 1201292] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 39 | domain_a_seq_39 | 321,733 | [1712419199, 1772725487] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 40 | domain_a_seq_40 | 17 | [0, 18] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 41 | domain_a_seq_41 | 9 | [2, 11] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 42 | domain_a_seq_42 | 407 | [0, 1017] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 43 | domain_a_seq_43 | 1,263 | [0, 3449] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 44 | domain_a_seq_44 | 3,002 | [0, 15146] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 45 | domain_a_seq_45 | 2,389 | [0, 9212] | 1,888 | 0.995 | 0.005 | 0.000 |  |
| 46 | domain_a_seq_46 | 12 | [0, 17] | 1,888 | 0.995 | 0.005 | 0.000 |  |

#### domain_b_seq

| feat_id | field | map_range | range | max_len | coverage | missing_prob | empty_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 67 | domain_b_seq_67 | 421,843 | [1752645918, 1772725642] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 68 | domain_b_seq_68 | 23 | [0, 27] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 69 | domain_b_seq_69 | 191,931 | [0, 143233599] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 70 | domain_b_seq_70 | 429 | [0, 733] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 71 | domain_b_seq_71 | 1,344 | [0, 2690] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 72 | domain_b_seq_72 | 3,663 | [0, 10489] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 73 | domain_b_seq_73 | 2,580 | [0, 7238] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 74 | domain_b_seq_74 | 14,338 | [-1, 638923] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 75 | domain_b_seq_75 | 20 | [0, 29] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 76 | domain_b_seq_76 | 8,802 | [-1, 164031] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 77 | domain_b_seq_77 | 127 | [0, 167] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 78 | domain_b_seq_78 | 2,522 | [0, 4319] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 79 | domain_b_seq_79 | 6,368 | [0, 11780] | 1,952 | 0.988 | 0.012 | 0.000 |  |
| 88 | domain_b_seq_88 | 12,270 | [-1, 288958] | 1,952 | 0.988 | 0.012 | 0.000 |  |

#### domain_c_seq

| feat_id | field | map_range | range | max_len | coverage | missing_prob | empty_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | domain_c_seq_27 | 441,676 | [1717589186, 1772725680] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 28 | domain_c_seq_28 | 59 | [2, 74] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 29 | domain_c_seq_29 | 172,473 | [0, 8227607] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 30 | domain_c_seq_30 | 509 | [0, 857] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 31 | domain_c_seq_31 | 2,948 | [-1, 7085] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 32 | domain_c_seq_32 | 6 | [1, 6] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 33 | domain_c_seq_33 | 3 | [1, 3] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 34 | domain_c_seq_34 | 16,901 | [8, 1986299] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 35 | domain_c_seq_35 | 1,587 | [0, 3003] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 36 | domain_c_seq_36 | 55,121 | [0, 1511292] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 37 | domain_c_seq_37 | 4,092 | [-1, 9974] | 3,894 | 0.998 | 0.002 | 0.000 |  |
| 47 | domain_c_seq_47 | 287,308 | [0, 278677639] | 3,894 | 0.998 | 0.002 | 0.000 |  |

#### domain_d_seq

| feat_id | field | map_range | range | max_len | coverage | missing_prob | empty_prob | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | domain_d_seq_17 | 4 | [1, 4] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 18 | domain_d_seq_18 | 422 | [0, 974] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 19 | domain_d_seq_19 | 1,466 | [0, 3419] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 20 | domain_d_seq_20 | 3,842 | [0, 11297] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 21 | domain_d_seq_21 | 2,451 | [0, 5105] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 22 | domain_d_seq_22 | 17,886 | [0, 510190] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 23 | domain_d_seq_23 | 123,039 | [0, 674033] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 24 | domain_d_seq_24 | 24 | [0, 602] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 25 | domain_d_seq_25 | 11 | [0, 14] | 3,951 | 0.920 | 0.080 | 0.000 |  |
| 26 | domain_d_seq_26 | 50,828 | [1726916040, 1772725620] | 3,951 | 0.920 | 0.080 | 0.000 |  |

---

## Data Quality Notes

### Missing Values

| source | field | type | missing_prob | coverage |
| --- | --- | --- | --- | --- |
| user_feature | user_int_feats_3 | int_value | 0.030 | 0.970 |
| user_feature | user_int_feats_4 | int_value | 0.030 | 0.970 |
| user_feature | user_int_feats_15 | int_array | 0.139 | 0.861 |
| user_feature | user_int_feats_48 | int_value | 0.002 | 0.998 |
| user_feature | user_int_feats_49 | int_value | 0.007 | 0.993 |
| user_feature | user_int_feats_50 | int_value | 0.004 | 0.996 |
| user_feature | user_int_feats_51 | int_value | 0.001 | 0.999 |
| user_feature | user_int_feats_52 | int_value | 0.001 | 0.999 |
| user_feature | user_int_feats_53 | int_value | 0.001 | 0.999 |
| user_feature | user_int_feats_54 | int_value | 0.368 | 0.632 |
| user_feature | user_int_feats_55 | int_value | 0.019 | 0.981 |
| user_feature | user_int_feats_56 | int_value | 0.019 | 0.981 |
| user_feature | user_int_feats_57 | int_value | 0.031 | 0.969 |
| user_feature | user_int_feats_58 | int_value | 0.150 | 0.850 |
| user_feature | user_int_feats_59 | int_value | 0.150 | 0.850 |
| user_feature | user_int_feats_60 | int_array | 0.592 | 0.408 |
| user_feature | user_int_feats_62 | int_array | 0.070 | 0.930 |
| user_feature | user_int_feats_63 | int_array | 0.070 | 0.930 |
| user_feature | user_int_feats_64 | int_array | 0.070 | 0.930 |
| user_feature | user_int_feats_65 | int_array | 0.080 | 0.920 |
| user_feature | user_int_feats_66 | int_array | 0.086 | 0.914 |
| user_feature | user_int_feats_80 | int_array | 0.200 | 0.800 |
| user_feature | user_int_feats_82 | int_value | 0.204 | 0.796 |
| user_feature | user_int_feats_86 | int_value | 0.692 | 0.308 |
| user_feature | user_int_feats_89 | int_array | 0.055 | 0.945 |
| user_feature | user_int_feats_90 | int_array | 0.091 | 0.909 |
| user_feature | user_int_feats_91 | int_array | 0.450 | 0.550 |
| user_feature | user_int_feats_92 | int_value | 0.494 | 0.506 |
| user_feature | user_int_feats_93 | int_value | 0.171 | 0.829 |
| user_feature | user_int_feats_94 | int_value | 0.521 | 0.479 |
| user_feature | user_int_feats_95 | int_value | 0.318 | 0.682 |
| user_feature | user_int_feats_96 | int_value | 0.678 | 0.322 |
| user_feature | user_int_feats_97 | int_value | 0.292 | 0.708 |
| user_feature | user_int_feats_98 | int_value | 0.103 | 0.897 |
| user_feature | user_int_feats_99 | int_value | 0.812 | 0.188 |
| user_feature | user_int_feats_100 | int_value | 0.845 | 0.155 |
| user_feature | user_int_feats_101 | int_value | 0.910 | 0.090 |
| user_feature | user_int_feats_102 | int_value | 0.877 | 0.123 |
| user_feature | user_int_feats_103 | int_value | 0.862 | 0.138 |
| user_feature | user_int_feats_104 | int_value | 0.372 | 0.628 |
| user_feature | user_int_feats_105 | int_value | 0.309 | 0.691 |
| user_feature | user_int_feats_106 | int_value | 0.160 | 0.840 |
| user_feature | user_int_feats_107 | int_value | 0.300 | 0.700 |
| user_feature | user_int_feats_108 | int_value | 0.516 | 0.484 |
| user_feature | user_int_feats_109 | int_value | 0.854 | 0.146 |
| user_feature | user_dense_feats_61 | float_array | 0.002 | 0.998 |
| user_feature | user_dense_feats_62 | float_array | 0.070 | 0.930 |
| user_feature | user_dense_feats_63 | float_array | 0.070 | 0.930 |
| user_feature | user_dense_feats_64 | float_array | 0.070 | 0.930 |
| user_feature | user_dense_feats_65 | float_array | 0.080 | 0.920 |
| user_feature | user_dense_feats_66 | float_array | 0.086 | 0.914 |
| user_feature | user_dense_feats_87 | float_array | 0.015 | 0.985 |
| user_feature | user_dense_feats_89 | float_array | 0.055 | 0.945 |
| user_feature | user_dense_feats_90 | float_array | 0.091 | 0.909 |
| user_feature | user_dense_feats_91 | float_array | 0.450 | 0.550 |
| item_feature | item_int_feats_5 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_6 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_7 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_8 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_9 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_10 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_11 | int_array | 0.439 | 0.548 |
| item_feature | item_int_feats_12 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_13 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_16 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_81 | int_value | 0.002 | 0.998 |
| item_feature | item_int_feats_83 | int_value | 0.832 | 0.185 |
| item_feature | item_int_feats_84 | int_value | 0.832 | 0.185 |
| item_feature | item_int_feats_85 | int_value | 0.832 | 0.185 |
| seq_feature | domain_a_seq_38 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_39 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_40 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_41 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_42 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_43 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_44 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_45 | - | 0.005 | 0.995 |
| seq_feature | domain_a_seq_46 | - | 0.005 | 0.995 |
| seq_feature | domain_b_seq_67 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_68 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_69 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_70 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_71 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_72 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_73 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_74 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_75 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_76 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_77 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_78 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_79 | - | 0.012 | 0.988 |
| seq_feature | domain_b_seq_88 | - | 0.012 | 0.988 |
| seq_feature | domain_c_seq_27 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_28 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_29 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_30 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_31 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_32 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_33 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_34 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_35 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_36 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_37 | - | 0.002 | 0.998 |
| seq_feature | domain_c_seq_47 | - | 0.002 | 0.998 |
| seq_feature | domain_d_seq_17 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_18 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_19 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_20 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_21 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_22 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_23 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_24 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_25 | - | 0.080 | 0.920 |
| seq_feature | domain_d_seq_26 | - | 0.080 | 0.920 |

### Empty Sequence Fields

未发现空序列字段。

### High Cardinality Features

| source | field | feat_id | map_range | coverage | 备注 |
| --- | --- | --- | --- | --- | --- |
| seq_feature | domain_a_seq_38 | 38 | 17,776 | 0.995 | 高基数 |
| seq_feature | domain_a_seq_39 | 39 | 321,733 | 0.995 | 高基数 |
| seq_feature | domain_b_seq_67 | 67 | 421,843 | 0.988 | 高基数 |
| seq_feature | domain_b_seq_69 | 69 | 191,931 | 0.988 | 高基数 |
| seq_feature | domain_b_seq_74 | 74 | 14,338 | 0.988 | 高基数 |
| seq_feature | domain_b_seq_88 | 88 | 12,270 | 0.988 | 高基数 |
| seq_feature | domain_c_seq_27 | 27 | 441,676 | 0.998 | 高基数 |
| seq_feature | domain_c_seq_29 | 29 | 172,473 | 0.998 | 高基数 |
| seq_feature | domain_c_seq_34 | 34 | 16,901 | 0.998 | 高基数 |
| seq_feature | domain_c_seq_36 | 36 | 55,121 | 0.998 | 高基数 |
| seq_feature | domain_c_seq_47 | 47 | 287,308 | 0.998 | 高基数 |
| seq_feature | domain_d_seq_22 | 22 | 17,886 | 0.920 | 高基数 |
| seq_feature | domain_d_seq_23 | 23 | 123,039 | 0.920 | 高基数 |
| seq_feature | domain_d_seq_26 | 26 | 50,828 | 0.920 | 高基数 |

