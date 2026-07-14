from __future__ import annotations

GGUF_DENSE_TYPE_IDS = {
    "iq2_xxs": 0,
    "q2_k": 1,
    "iq1_m": 2,
    "q4_k": 3,
    "q5_k": 4,
    "iq2_xs": 5,
    "iq3_xxs": 6,
    "q6_k": 8,
}

GGUF_DENSE_TYPE_NAMES = {value: key for key, value in GGUF_DENSE_TYPE_IDS.items()}
