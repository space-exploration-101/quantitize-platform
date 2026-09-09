#include "nms_output21_zynq.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Zynq bare-metal friendly 21-class postprocess.
 *
 * This file intentionally avoids file I/O and dynamic malloc. The caller passes
 * the FPGA output buffer directly. The implementation uses static workspace in
 * BSS, so confirm the target linker script has enough memory for a few MB.
 *
 * Output layout expected from FPGA:
 *   [34000][32] fp16
 * Valid channels:
 *   ch0..3    bbox cx,cy,w,h
 *   ch4..24   21 class scores
 *   ch25..30  6 keypoint channels
 *   ch31      padding, ignored
 */

#define NMS21_DEFAULT_CONF_THRES 0.25f
#define NMS21_DEFAULT_IOU_THRES 0.70f
#define NMS21_DEFAULT_MAX_NMS 30000
#define NMS21_DEFAULT_MOON_CLASS_ID 20
#define NMS21_DEFAULT_PAD_X 140.0f
#define NMS21_DEFAULT_PAD_Y 140.0f
#define NMS21_DEFAULT_VALID_W 1000.0f
#define NMS21_DEFAULT_VALID_H 1000.0f

typedef struct {
    float box[4]; /* x1,y1,x2,y2 in padded 1280 coordinates until final output */
    float score;
    int class_id;
    float kpt[NMS21_KPT_DIM];
} Candidate21;

static Candidate21 g_candidates[NMS21_NUM_BOXES];
static int g_order[NMS21_NUM_BOXES];
static unsigned char g_suppressed[NMS21_NUM_BOXES];
static int g_keep_idx[NMS21_MAX_DET];

static const Candidate21 *g_sort_candidates = NULL;

void nms21_default_params(NMS21Params *params) {
    if (!params) return;
    params->conf_thres = NMS21_DEFAULT_CONF_THRES;
    params->iou_thres = NMS21_DEFAULT_IOU_THRES;
    params->max_det = NMS21_MAX_DET;
    params->max_nms = NMS21_DEFAULT_MAX_NMS;
    params->moon_class_id = NMS21_DEFAULT_MOON_CLASS_ID;
    params->pad_x = NMS21_DEFAULT_PAD_X;
    params->pad_y = NMS21_DEFAULT_PAD_Y;
    params->valid_w = NMS21_DEFAULT_VALID_W;
    params->valid_h = NMS21_DEFAULT_VALID_H;
}

static float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x1u;
    uint16_t exp = (h >> 10) & 0x1fu;
    uint16_t frac = h & 0x03ffu;

    uint32_t out_sign = ((uint32_t)sign) << 31;
    uint32_t out_exp;
    uint32_t out_frac;

    if (exp == 0) {
        if (frac == 0) {
            out_exp = 0;
            out_frac = 0;
        } else {
            int shift = 0;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                shift++;
            }
            frac &= 0x03ffu;
            out_exp = (uint32_t)(127 - 15 - shift) << 23;
            out_frac = ((uint32_t)frac) << 13;
        }
    } else if (exp == 31) {
        out_exp = 0xffu << 23;
        out_frac = ((uint32_t)frac) << 13;
    } else {
        out_exp = (uint32_t)(exp - 15 + 127) << 23;
        out_frac = ((uint32_t)frac) << 13;
    }

    uint32_t u = out_sign | out_exp | out_frac;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float clampf_local(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float iou_xyxy(const float *a, const float *b) {
    float xx1 = (a[0] > b[0]) ? a[0] : b[0];
    float yy1 = (a[1] > b[1]) ? a[1] : b[1];
    float xx2 = (a[2] < b[2]) ? a[2] : b[2];
    float yy2 = (a[3] < b[3]) ? a[3] : b[3];
    float w = xx2 - xx1;
    float h = yy2 - yy1;
    if (w < 0.0f) w = 0.0f;
    if (h < 0.0f) h = 0.0f;
    float inter = w * h;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float uni = area_a + area_b - inter;
    return (uni > 1e-7f) ? (inter / uni) : 0.0f;
}

static int cmp_candidate_score_desc(const void *pa, const void *pb) {
    int ia = *(const int *)pa;
    int ib = *(const int *)pb;
    if (g_sort_candidates[ia].score > g_sort_candidates[ib].score) return -1;
    if (g_sort_candidates[ia].score < g_sort_candidates[ib].score) return 1;
    return ia - ib;
}

static void candidate_to_output(
    const Candidate21 *src,
    const NMS21Params *params,
    NMS21Detection *dst
) {
    dst->x1 = clampf_local(src->box[0] - params->pad_x, 0.0f, params->valid_w);
    dst->y1 = clampf_local(src->box[1] - params->pad_y, 0.0f, params->valid_h);
    dst->x2 = clampf_local(src->box[2] - params->pad_x, 0.0f, params->valid_w);
    dst->y2 = clampf_local(src->box[3] - params->pad_y, 0.0f, params->valid_h);
    dst->score = src->score;
    dst->class_id = src->class_id;

    memcpy(dst->kpt, src->kpt, sizeof(dst->kpt));
    dst->kpt[0] = clampf_local(dst->kpt[0] - params->pad_x, 0.0f, params->valid_w);
    dst->kpt[1] = clampf_local(dst->kpt[1] - params->pad_y, 0.0f, params->valid_h);
    dst->kpt[3] = clampf_local(dst->kpt[3] - params->pad_x, 0.0f, params->valid_w);
    dst->kpt[4] = clampf_local(dst->kpt[4] - params->pad_y, 0.0f, params->valid_h);
}

static int build_candidates(const uint16_t *raw_fp16, const NMS21Params *params) {
    int num_valid = 0;

    for (int b = 0; b < NMS21_NUM_BOXES; b++) {
        float cx = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + 0]);
        float cy = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + 1]);
        float w = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + 2]);
        float h = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + 3]);

        float best_score = -1.0f;
        int best_class = 0;
        for (int c = 0; c < NMS21_NC; c++) {
            float score = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + 4 + c]);
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score <= params->conf_thres) {
            continue;
        }

        Candidate21 *d = &g_candidates[num_valid];
        d->box[0] = cx - w * 0.5f;
        d->box[1] = cy - h * 0.5f;
        d->box[2] = cx + w * 0.5f;
        d->box[3] = cy + h * 0.5f;
        d->score = best_score;
        d->class_id = best_class;
        for (int k = 0; k < NMS21_KPT_DIM; k++) {
            d->kpt[k] = fp16_to_float(raw_fp16[b * NMS21_RAW_CH + NMS21_KPT_START + k]);
        }

        g_order[num_valid] = num_valid;
        num_valid++;
    }

    return num_valid;
}

static int ordinary_class_aware_nms(int num_valid, const NMS21Params *params) {
    if (num_valid <= 0) return 0;

    g_sort_candidates = g_candidates;
    qsort(g_order, (size_t)num_valid, sizeof(int), cmp_candidate_score_desc);
    g_sort_candidates = NULL;

    if (num_valid > params->max_nms) {
        num_valid = params->max_nms;
    }

    memset(g_suppressed, 0, (size_t)num_valid);

    int num_kept = 0;
    for (int oi = 0; oi < num_valid && num_kept < params->max_det; oi++) {
        int i = g_order[oi];
        if (g_suppressed[oi]) continue;

        g_keep_idx[num_kept++] = i;

        for (int oj = oi + 1; oj < num_valid; oj++) {
            int j = g_order[oj];
            if (g_suppressed[oj]) continue;
            if (g_candidates[i].class_id != g_candidates[j].class_id) continue;
            if (iou_xyxy(g_candidates[i].box, g_candidates[j].box) > params->iou_thres) {
                g_suppressed[oj] = 1;
            }
        }
    }

    return num_kept;
}

static void business_filter(int num_kept, const NMS21Params *params, NMS21Output *output) {
    int best_landmark = -1;
    int best_moon = -1;

    for (int i = 0; i < num_kept; i++) {
        int idx = g_keep_idx[i];
        Candidate21 *d = &g_candidates[idx];

        if (d->class_id == params->moon_class_id) {
            if (best_moon < 0 || d->score > g_candidates[best_moon].score) {
                best_moon = idx;
            }
        } else {
            if (best_landmark < 0 || d->score > g_candidates[best_landmark].score) {
                best_landmark = idx;
            }
        }
    }

    output->num_detections = 0;
    if (best_landmark >= 0) {
        candidate_to_output(&g_candidates[best_landmark], params,
                            &output->detections[output->num_detections]);
        output->num_detections++;
    }
    if (best_moon >= 0) {
        candidate_to_output(&g_candidates[best_moon], params,
                            &output->detections[output->num_detections]);
        output->num_detections++;
    }
}

int nms_from_output21_zynq(
    const uint16_t *raw_fp16,
    const NMS21Params *params,
    NMS21Output *output
) {
    NMS21Params local_params;

    if (!raw_fp16 || !output) {
        return -1;
    }

    if (params) {
        local_params = *params;
    } else {
        nms21_default_params(&local_params);
    }

    if (local_params.max_det <= 0 || local_params.max_det > NMS21_MAX_DET) {
        return -2;
    }
    if (local_params.max_nms <= 0 || local_params.max_nms > NMS21_NUM_BOXES) {
        return -2;
    }
    if (local_params.moon_class_id < 0 || local_params.moon_class_id >= NMS21_NC) {
        return -2;
    }

    memset(output, 0, sizeof(*output));

    int num_valid = build_candidates(raw_fp16, &local_params);
    if (num_valid <= 0) {
        output->num_detections = 0;
        return 0;
    }

    int num_kept = ordinary_class_aware_nms(num_valid, &local_params);
    business_filter(num_kept, &local_params, output);
    return 0;
}
