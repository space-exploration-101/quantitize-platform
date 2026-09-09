#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * 21-class YOLOv8 Pose postprocess for FPGA OUTPUT.BIN.
 *
 * Expected FPGA output layout:
 *   raw[NUM_BOXES][RAW_CH] fp16
 *
 * Valid channels:
 *   ch0..3    bbox: cx, cy, w, h, in 1280x1280 padded input coordinates
 *   ch4..24   21 class scores
 *   ch25..30  2 keypoints x/y/score, 6 channels total
 *   ch31      padding, ignored
 *
 * Business rule after ordinary class-aware NMS:
 *   - class 0..19: earth landmarks, mutually exclusive, keep only best score
 *   - class 20: moon, may coexist with one landmark, keep only best moon
 */

#define NUM_BOXES 34000
#define RAW_CH 32
#define NC 21
#define NUM_FEATURES (4 + NC + 6)
#define KPT_START (4 + NC)
#define KPT_DIM 6

#define CONF_THRES 0.25f
#define IOU_THRES 0.70f
#define MAX_DET 30
#define MAX_NMS 30000

#define PAD_X 140.0f
#define PAD_Y 140.0f
#define VALID_W 1000.0f
#define VALID_H 1000.0f

#define MOON_CLASS_ID 20
#define MAX_FINAL_DET 2

typedef struct {
    float box[4];       /* x1,y1,x2,y2 before padding removal during NMS */
    float score;
    int class_id;
    float kpt[KPT_DIM]; /* [x0,y0,s0,x1,y1,s1] */
    float raw_cls[NC];
} Detection21;

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

static const Detection21 *g_sort_dets = NULL;

static int cmp_detection_score_desc(const void *pa, const void *pb) {
    const int ia = *(const int *)pa;
    const int ib = *(const int *)pb;
    if (g_sort_dets[ia].score > g_sort_dets[ib].score) return -1;
    if (g_sort_dets[ia].score < g_sort_dets[ib].score) return 1;
    return ia - ib;
}

static void remove_padding_and_clip(Detection21 *d) {
    d->box[0] = clampf_local(d->box[0] - PAD_X, 0.0f, VALID_W);
    d->box[1] = clampf_local(d->box[1] - PAD_Y, 0.0f, VALID_H);
    d->box[2] = clampf_local(d->box[2] - PAD_X, 0.0f, VALID_W);
    d->box[3] = clampf_local(d->box[3] - PAD_Y, 0.0f, VALID_H);

    /* kpt layout: [x0,y0,s0,x1,y1,s1]. Only x/y coordinates remove padding. */
    d->kpt[0] = clampf_local(d->kpt[0] - PAD_X, 0.0f, VALID_W);
    d->kpt[1] = clampf_local(d->kpt[1] - PAD_Y, 0.0f, VALID_H);
    d->kpt[3] = clampf_local(d->kpt[3] - PAD_X, 0.0f, VALID_W);
    d->kpt[4] = clampf_local(d->kpt[4] - PAD_Y, 0.0f, VALID_H);
}

static int build_candidates_from_raw(const uint16_t *raw, Detection21 *cands, int *order) {
    int num_valid = 0;

    for (int b = 0; b < NUM_BOXES; b++) {
        float cx = fp16_to_float(raw[b * RAW_CH + 0]);
        float cy = fp16_to_float(raw[b * RAW_CH + 1]);
        float w = fp16_to_float(raw[b * RAW_CH + 2]);
        float h = fp16_to_float(raw[b * RAW_CH + 3]);

        float best_score = -1.0f;
        int best_class = 0;
        float raw_cls[NC];
        for (int c = 0; c < NC; c++) {
            float score = fp16_to_float(raw[b * RAW_CH + 4 + c]);
            raw_cls[c] = score;
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score <= CONF_THRES) {
            continue;
        }

        Detection21 d;
        memset(&d, 0, sizeof(d));
        d.box[0] = cx - w * 0.5f;
        d.box[1] = cy - h * 0.5f;
        d.box[2] = cx + w * 0.5f;
        d.box[3] = cy + h * 0.5f;
        d.score = best_score;
        d.class_id = best_class;
        memcpy(d.raw_cls, raw_cls, sizeof(raw_cls));

        for (int k = 0; k < KPT_DIM; k++) {
            d.kpt[k] = fp16_to_float(raw[b * RAW_CH + KPT_START + k]);
        }

        cands[num_valid] = d;
        order[num_valid] = num_valid;
        num_valid++;
    }

    return num_valid;
}

static int ordinary_class_aware_nms(Detection21 *cands, int num_valid, int *order, int *keep_idx) {
    if (num_valid <= 0) return 0;

    g_sort_dets = cands;
    qsort(order, (size_t)num_valid, sizeof(int), cmp_detection_score_desc);
    g_sort_dets = NULL;

    if (num_valid > MAX_NMS) {
        num_valid = MAX_NMS;
    }

    bool *suppressed = (bool *)calloc((size_t)num_valid, sizeof(bool));
    if (!suppressed) {
        fprintf(stderr, "calloc suppressed failed\n");
        return -1;
    }

    int num_kept = 0;
    for (int oi = 0; oi < num_valid && num_kept < MAX_DET; oi++) {
        int i = order[oi];
        if (suppressed[oi]) continue;

        keep_idx[num_kept++] = i;

        for (int oj = oi + 1; oj < num_valid; oj++) {
            int j = order[oj];
            if (suppressed[oj]) continue;
            if (cands[i].class_id != cands[j].class_id) continue;

            float iou = iou_xyxy(cands[i].box, cands[j].box);
            if (iou > IOU_THRES) {
                suppressed[oj] = true;
            }
        }
    }

    free(suppressed);
    return num_kept;
}

static int business_filter_landmark_and_moon(
    Detection21 *cands,
    const int *keep_idx,
    int num_kept,
    Detection21 *final_dets
) {
    int best_landmark = -1;
    int best_moon = -1;

    for (int i = 0; i < num_kept; i++) {
        int idx = keep_idx[i];
        Detection21 *d = &cands[idx];
        if (d->class_id == MOON_CLASS_ID) {
            if (best_moon < 0 || d->score > cands[best_moon].score) {
                best_moon = idx;
            }
        } else {
            if (best_landmark < 0 || d->score > cands[best_landmark].score) {
                best_landmark = idx;
            }
        }
    }

    int n = 0;
    if (best_landmark >= 0) {
        final_dets[n] = cands[best_landmark];
        remove_padding_and_clip(&final_dets[n]);
        n++;
    }
    if (best_moon >= 0) {
        final_dets[n] = cands[best_moon];
        remove_padding_and_clip(&final_dets[n]);
        n++;
    }
    return n;
}

static void print_detection(const Detection21 *d, int idx) {
    printf("Detection %d:\n", idx);
    printf("  BBox(xyxy, 1000x1000 unpadded): x1=%.3f, y1=%.3f, x2=%.3f, y2=%.3f\n",
           d->box[0], d->box[1], d->box[2], d->box[3]);
    printf("  Conf=%.6f, ClassID=%d%s\n",
           d->score, d->class_id, d->class_id == MOON_CLASS_ID ? " (moon)" : " (landmark)");
    printf("  Keypoints: [%.3f, %.3f, %.6f, %.3f, %.3f, %.6f]\n",
           d->kpt[0], d->kpt[1], d->kpt[2], d->kpt[3], d->kpt[4], d->kpt[5]);
}

int main(int argc, char **argv) {
    const char *bin_path = (argc > 1) ? argv[1] : "OUTPUT.BIN";

    FILE *fp = fopen(bin_path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open: %s\n", bin_path);
        return 1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        fprintf(stderr, "fseek failed\n");
        return 1;
    }
    long file_bytes = ftell(fp);
    rewind(fp);

    const long expect_bytes = (long)NUM_BOXES * RAW_CH * (long)sizeof(uint16_t);
    if (file_bytes < expect_bytes) {
        fclose(fp);
        fprintf(stderr, "File too small: %ld, expected >= %ld\n", file_bytes, expect_bytes);
        return 1;
    }

    uint16_t *raw = (uint16_t *)malloc((size_t)expect_bytes);
    Detection21 *cands = (Detection21 *)malloc((size_t)NUM_BOXES * sizeof(Detection21));
    int *order = (int *)malloc((size_t)NUM_BOXES * sizeof(int));
    int *keep_idx = (int *)malloc((size_t)MAX_DET * sizeof(int));
    Detection21 final_dets[MAX_FINAL_DET];

    if (!raw || !cands || !order || !keep_idx) {
        fclose(fp);
        free(raw);
        free(cands);
        free(order);
        free(keep_idx);
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    size_t nread = fread(raw, 1, (size_t)expect_bytes, fp);
    fclose(fp);
    if (nread != (size_t)expect_bytes) {
        free(raw);
        free(cands);
        free(order);
        free(keep_idx);
        fprintf(stderr, "fread failed: got %zu bytes, expected %ld\n", nread, expect_bytes);
        return 1;
    }

    int num_valid = build_candidates_from_raw(raw, cands, order);
    free(raw);

    printf("NMS21 result from %s\n", bin_path);
    printf("params: boxes=%d raw_ch=%d valid_ch=%d nc=%d conf=%.2f iou=%.2f max_det=%d max_nms=%d pad=(%.0f,%.0f) moon_class=%d\n",
           NUM_BOXES, RAW_CH, NUM_FEATURES, NC, CONF_THRES, IOU_THRES, MAX_DET, MAX_NMS,
           PAD_X, PAD_Y, MOON_CLASS_ID);
    printf("candidates after score filter: %d\n", num_valid);

    if (num_valid == 0) {
        printf("final detections: 0\n");
        free(cands);
        free(order);
        free(keep_idx);
        return 0;
    }

    int num_kept = ordinary_class_aware_nms(cands, num_valid, order, keep_idx);
    if (num_kept < 0) {
        free(cands);
        free(order);
        free(keep_idx);
        return 1;
    }
    printf("detections after ordinary NMS: %d\n", num_kept);

    int num_final = business_filter_landmark_and_moon(cands, keep_idx, num_kept, final_dets);
    printf("final detections after business filter: %d\n", num_final);
    for (int i = 0; i < num_final; i++) {
        print_detection(&final_dets[i], i);
    }

    free(cands);
    free(order);
    free(keep_idx);
    return 0;
}
