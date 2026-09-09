#ifndef NMS_OUTPUT21_ZYNQ_H
#define NMS_OUTPUT21_ZYNQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NMS21_NUM_BOXES 34000
#define NMS21_RAW_CH 32
#define NMS21_NC 21
#define NMS21_KPT_DIM 6
#define NMS21_NUM_FEATURES (4 + NMS21_NC + NMS21_KPT_DIM)
#define NMS21_KPT_START (4 + NMS21_NC)
#define NMS21_MAX_DET 30
#define NMS21_MAX_FINAL_DET 2

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    float kpt[NMS21_KPT_DIM]; /* [x0,y0,s0,x1,y1,s1] */
} NMS21Detection;

typedef struct {
    float conf_thres;      /* default recommendation: 0.25 */
    float iou_thres;       /* default recommendation: 0.70 */
    int max_det;           /* ordinary NMS upper bound, <= NMS21_MAX_DET */
    int max_nms;           /* score-sorted candidate upper bound, <= 30000 */
    int moon_class_id;     /* default: 20, assuming class 0..19 are landmarks */
    float pad_x;           /* default: 140 */
    float pad_y;           /* default: 140 */
    float valid_w;         /* default: 1000 */
    float valid_h;         /* default: 1000 */
} NMS21Params;

typedef struct {
    int num_detections; /* final count: 0..2 */
    NMS21Detection detections[NMS21_MAX_FINAL_DET];
} NMS21Output;

void nms21_default_params(NMS21Params *params);

/*
 * raw_fp16 must point to FPGA output buffer:
 *   uint16_t raw_fp16[NMS21_NUM_BOXES * NMS21_RAW_CH]
 *
 * Return value:
 *   0  success
 *  -1  invalid argument
 *  -2  internal limit/config error
 */
int nms_from_output21_zynq(
    const uint16_t *raw_fp16,
    const NMS21Params *params,
    NMS21Output *output
);

#ifdef __cplusplus
}
#endif

#endif /* NMS_OUTPUT21_ZYNQ_H */
