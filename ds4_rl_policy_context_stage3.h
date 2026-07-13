#pragma once
#include <stdint.h>

#define DS4_RL_POLICY_STAGE3_N_LAYER 43u
#define DS4_RL_POLICY_STAGE3_N_POS_BUCKET 6u
static inline uint32_t ds4_rl_policy_stage3_pos_bucket(uint64_t pos) {
    if (pos < 32u) return 0u;
    if (pos < 64u) return 1u;
    if (pos < 96u) return 2u;
    if (pos < 128u) return 3u;
    if (pos < 192u) return 4u;
    return 5u;
}

static const uint8_t ds4_rl_policy_stage3_preseed_k_table[43][6] = {
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {2, 2, 2, 4, 2, 2},
    {0, 1, 2, 1, 1, 1},
    {0, 2, 2, 2, 2, 2},
    {0, 0, 1, 0, 0, 0},
    {0, 1, 2, 1, 1, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {0, 0, 1, 2, 1, 1},
    {1, 1, 1, 1, 1, 1},
    {2, 2, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 2, 1, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {4, 4, 2, 4, 2, 2},
    {1, 0, 1, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {2, 2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {1, 1, 1, 1, 1, 1},
    {2, 4, 4, 2, 4, 4},
    {2, 2, 2, 2, 2, 2},
    {4, 4, 2, 2, 2, 2},
    {0, 1, 1, 0, 0, 0},
    {0, 2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {0, 2, 2, 4, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {2, 2, 2, 1, 2, 2},
    {4, 4, 4, 4, 4, 4},
    {0, 2, 2, 1, 1, 1},
    {2, 2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {4, 4, 4, 4, 4, 4},
    {2, 2, 2, 2, 2, 2},
    {4, 4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4, 4}
};

static inline uint32_t ds4_rl_policy_stage3_preseed_k(uint64_t pos, uint32_t layer) {
    if (layer >= DS4_RL_POLICY_STAGE3_N_LAYER) return 0u;
    const uint32_t bucket = ds4_rl_policy_stage3_pos_bucket(pos);
    return (uint32_t)ds4_rl_policy_stage3_preseed_k_table[layer][bucket];
}
