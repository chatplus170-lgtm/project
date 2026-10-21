/*
 * LMS_no_UDP_NEON.c - Raw Ethernet (Layer 2) -> ARM NEON pack -> AXI DMA -> LMS
 * ----------------------------------------------------------------------------
 * Target : Zynq-7000 (ZC702 / Zynq-7020), Vitis bare-metal standalone
 * Flags  : -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -O3   (link with -lm)
 *          The BSP must be built with the same float ABI.
 *
 * Frame layout on the wire (USE_APP_HDR = 1):
 *   [0..13]   Ethernet header (DstMAC, SrcMAC, EtherType 0x88B5)
 *   [14..29]  App header, little endian: u32 seq | u32 N | u32 mu_q15 | u32 reserved
 *               mu_q15 = 0 means "keep the current mu"
 *   [30..]    complex64 x[N] (8*N bytes) then complex64 d[N] (8*N bytes)
 *
 * Changes versus the first NEON draft:
 *   - Run ends on TARGET_SAMPLES *or* after IDLE_TIMEOUT_SEC without frames, so a
 *     dropped frame can no longer hang the board forever.
 *   - Sequence gaps are counted from frame 0 (first lost frames are detected too);
 *     a restarted sender (seq going backwards) is a resync, not a huge gap count.
 *   - Timing is fully accounted: t_neon (pack + flush), t_dma (transfer + wait +
 *     result invalidate), t_total (whole per-frame handling), other = remainder.
 *   - RX buffer index is derived from the BD address (fixed BD<->buffer mapping).
 *   - Optional early RX-BD recycling (RECYCLE_BD_EARLY): the RX buffer is free as
 *     soon as NEON has converted it, so the GEM can receive the next frame while
 *     the DMA runs. Default 0 (same BD handling as the known-good raw version).
 *   - GEM RX status register is printed (buffer-not-available / overrun bits).
 *   - One-time self-test: NEON output of the first frame is compared with the
 *     scalar reference (outside all timers).
 */

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sleep.h"
#include "xaxidma.h"
#include "xemacps.h"
#include "xgpio.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_mmu.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xparameters.h"
#include "xtime_l.h"

/* =========================== USER CONFIG ================================== */

#define EMAC_DEV_ID XPAR_XEMACPS_0_DEVICE_ID
#define EMAC_BASEADDR XPAR_XEMACPS_0_BASEADDR
#define DMA_DEV_ID XPAR_AXI_DMA_0_DEVICE_ID
#define GPIO_DEV_ID XPAR_AXI_GPIO_0_DEVICE_ID
#define GPIO_CHANNEL 1

#define MAX_FRAME_SIZE 9216 /* 144 x 64 B RX buffer (jumbo)                   */
#define MAX_SAMPLES 576     /* >= (9216-30)/16 = 574 samples per frame         */
#define NUM_RX_BD 16
#define ETH_HDR_BYTES 14

#define USE_APP_HDR 1 /* 1: 16 B header (seq,N,mu,reserved); 0: raw payload   */
#define APP_HDR_BYTES (USE_APP_HDR ? 16U : 0U)

#define TARGET_SAMPLES 307200U /* 600 frames at N = 512                       */
#define Y_LATENCY 7            /* PIPELINE_LATENCY in the RTL                 */
#define METRIC_WIN 256

#define IDLE_TIMEOUT_SEC 2 /* end the run if no frame arrives for this long   */
#define IDLE_TIMEOUT_TICKS ((XTime)IDLE_TIMEOUT_SEC * COUNTS_PER_SECOND)

#define RECYCLE_BD_EARLY 0 /* 1: give the RX BD back right after NEON         */
#define SELFTEST_FIRST_FRAME 1

/* 1: after each frame is processed, also pack the SAME frame with the scalar
 * code and time it (cold cache, same flush), so the report prints the measured
 * NEON speedup. It adds run time, so throughput/script time are not valid in
 * this mode; use it for the t_neon vs t_scalar comparison only. */
#define BENCH_SCALAR_COMPARE 1
#if BENCH_SCALAR_COMPARE && RECYCLE_BD_EARLY
#error "BENCH_SCALAR_COMPARE needs the RX buffer to stay valid: set RECYCLE_BD_EARLY 0"
#endif

#define ETHERTYPE_FILTER 1
#define ACCEPT_ETHERTYPE 0x88B5

/* GEM writes the frame at buf+2, so the payload after the 14 B Ethernet header
 * starts at buf+16 (16-byte aligned); with the 16 B app header, x is at buf+32. */
#define RX_ALIGN_OFFSET 2

#define MU_FLOAT 0.005f
#define Q15_SCALE 32768.0f
#define DMA_TIMEOUT_SEC 1

static const u8 BOARD_MAC[6] = {0x00, 0x0a, 0x35, 0x00, 0x01, 0x03};

#if USE_APP_HDR
typedef struct {
  u32 seq;
  u32 n;
  u32 mu_q15;
  u32 reserved;
} __attribute__((packed)) app_hdr_t;
#endif

/* ============================= BUFFERS ==================================== */

static XEmacPs Emac;
static XAxiDma AxiDma;
static XGpio Gpio;

static u8 BdMem[0x100000] __attribute__((aligned(0x100000)));
static u8 RxBuffer[NUM_RX_BD][MAX_FRAME_SIZE] __attribute__((aligned(64)));

/* 64-bit Q1.15 words written by NEON, read by AXI DMA MM2S */
static u64 DmaTxBuf[MAX_SAMPLES] __attribute__((aligned(64)));
/* FPGA S2MM output words */
static u64 TxResultBuffer[MAX_SAMPLES] __attribute__((aligned(64)));
#if SELFTEST_FIRST_FRAME || BENCH_SCALAR_COMPARE
static u64 RefBuf[MAX_SAMPLES] __attribute__((aligned(64)));
#endif

/* ============================ Q1.15 & NEON ================================ */

/* Scalar reference: clamp, scale, round half away from zero. */
static inline int16_t f32_to_q15(float v) {
  const float max_val = 1.0f - 1.0f / Q15_SCALE;
  if (v > max_val)
    v = max_val;
  if (v < -1.0f)
    v = -1.0f;
  return (int16_t)lroundf(v * Q15_SCALE);
}

static inline u64 pack_scalar(float xr, float xi, float dr, float di) {
  return ((u64)(u16)f32_to_q15(xr) << 48) | ((u64)(u16)f32_to_q15(xi) << 32) |
         ((u64)(u16)f32_to_q15(dr) << 16) | (u64)(u16)f32_to_q15(di);
}

/* 4 floats -> 4 x int16 Q1.15. vcvtq truncates toward zero, so add
 * copysign(0.5, v) first (= round half away from zero, like lroundf).
 * vcvtq saturates to int32 and vqmovn saturates to int16, which replaces the
 * explicit clamp. */
static inline int16x4_t q15(float32x4_t v) {
  v = vmulq_n_f32(v, Q15_SCALE);
  v = vaddq_f32(v, vbslq_f32(vdupq_n_u32(0x80000000u), v, vdupq_n_f32(0.5f)));
  return vqmovn_s32(vcvtq_s32_f32(v));
}

static void neon_pack_xd(const float *x, const float *d, u64 *out, u32 n) {
  u32 i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4x2_t xv = vld2q_f32(x + 2 * i); /* val[0]=re, val[1]=im */
    float32x4x2_t dv = vld2q_f32(d + 2 * i);
    int16x4x4_t o;
    o.val[0] = q15(dv.val[1]); /* halfword 0 (bits 15:0)  : d_imag */
    o.val[1] = q15(dv.val[0]); /* halfword 1 (bits 31:16) : d_real */
    o.val[2] = q15(xv.val[1]); /* halfword 2 (bits 47:32) : x_imag */
    o.val[3] = q15(xv.val[0]); /* halfword 3 (bits 63:48) : x_real */
    vst4_s16((int16_t *)(out + i), o);
  }
  for (; i < n; i++) /* scalar tail */
    out[i] = pack_scalar(x[2 * i], x[2 * i + 1], d[2 * i], d[2 * i + 1]);
}

static void pf(double v, int dec) {
  u32 scale = 1, d, ip, fp, i;
  u64 x;

  for (i = 0; i < (u32)dec; i++)
    scale *= 10;
  if (v < 0.0) {
    xil_printf("-");
    v = -v;
  }

  x = (u64)(v * (double)scale + 0.5);
  ip = (u32)(x / scale);
  fp = (u32)(x % scale);

  xil_printf("%lu.", (unsigned long)ip);
  for (d = scale / 10; d > 1 && fp < d; d /= 10)
    xil_printf("0");
  xil_printf("%lu", (unsigned long)fp);
}

/* ============================ GPIO (mu) =================================== */

static int init_gpio_mu(void) {
  XGpio_Config *cfg = XGpio_LookupConfig(GPIO_DEV_ID);
  if (!cfg)
    return XST_FAILURE;
  if (XGpio_CfgInitialize(&Gpio, cfg, cfg->BaseAddress) != XST_SUCCESS)
    return XST_FAILURE;

  XGpio_SetDataDirection(&Gpio, GPIO_CHANNEL, 0x0);
  int16_t mu_q15 = (int16_t)(MU_FLOAT * Q15_SCALE); /* 163 */
  XGpio_DiscreteWrite(&Gpio, GPIO_CHANNEL, (u32)(u16)mu_q15);
  return XST_SUCCESS;
}

/* ============================ AXI DMA ===================================== */

static int init_axi_dma(void) {
  XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
  if (!cfg)
    return XST_FAILURE;
  if (XAxiDma_CfgInitialize(&AxiDma, cfg) != XST_SUCCESS)
    return XST_FAILURE;
  if (XAxiDma_HasSg(&AxiDma)) {
    xil_printf("ERROR: AXI DMA is in SG mode, need Simple mode\r\n");
    return XST_FAILURE;
  }
  XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
  XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
  return XST_SUCCESS;
}

static int dma_wait_done(void) {
  XTime t0, t;
  XTime_GetTime(&t0);

  while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
         XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
    XTime_GetTime(&t);
    if ((t - t0) > (XTime)DMA_TIMEOUT_SEC * COUNTS_PER_SECOND)
      return XST_FAILURE;
  }
  return XST_SUCCESS;
}

/*
 * Streams `len` bytes from src_addr through the FPGA into TxResultBuffer.
 * The caller has already flushed src_addr (that flush is timed as t_neon).
 * dma_ticks covers arming S2MM then MM2S, the wait, and the result invalidate.
 */
static int fpga_process(UINTPTR src_addr, u32 len, XTime *dma_ticks) {
  XTime t0, t1;
  int st = XST_SUCCESS;

  XTime_GetTime(&t0);

  if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)TxResultBuffer, len,
                             XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
    st = XST_FAILURE;
  else if (XAxiDma_SimpleTransfer(&AxiDma, src_addr, len,
                                  XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
    st = XST_FAILURE;
  else
    st = dma_wait_done();

  if (st != XST_SUCCESS) {
    XAxiDma_Reset(&AxiDma);
    while (!XAxiDma_ResetIsDone(&AxiDma)) {
    }
    XTime_GetTime(&t1);
    *dma_ticks = t1 - t0;
    return XST_FAILURE;
  }

  Xil_DCacheInvalidateRange((UINTPTR)TxResultBuffer, len);
  XTime_GetTime(&t1);
  *dma_ticks = t1 - t0;
  return XST_SUCCESS;
}

/* ===================== LMS METRICS (from S2MM results) ==================== */
static u8 m_start_captured = 0;
static u64 m_start_sumsq = 0;
static u32 m_start_n = 0;
static double m_end_e_rms = 0.0, m_end_d_rms = 0.0;
static double m_floor = 1.0e9;

static void metrics_update(const u64 *res, u32 words) {
  u32 valid_start = (words > Y_LATENCY) ? Y_LATENCY : 0;
  u32 avail = words - valid_start;
  if (avail == 0)
    return;

  u32 w = (avail < METRIC_WIN) ? avail : METRIC_WIN;
  u32 i;
  u64 se = 0, sd = 0;

  if (!m_start_captured) {
    u64 s = 0;
    for (i = valid_start; i < valid_start + w; i++) {
      int32_t er = (int16_t)(res[i] >> 16);
      int32_t ei = (int16_t)(res[i]);
      s += (u64)((int64_t)er * er + (int64_t)ei * ei);
    }
    m_start_sumsq = s;
    m_start_n = w;
    m_start_captured = 1;
  }

  for (i = words - w; i < words; i++) {
    u64 v = res[i];
    int32_t yr = (int16_t)(v >> 48), yi = (int16_t)(v >> 32);
    int32_t er = (int16_t)(v >> 16), ei = (int16_t)(v);
    int32_t dr = yr + er, di = yi + ei;
    se += (u64)((int64_t)er * er + (int64_t)ei * ei);
    sd += (u64)((int64_t)dr * dr + (int64_t)di * di);
  }

  m_end_e_rms = sqrt((double)se / (double)w) / 32768.0;
  m_end_d_rms = sqrt((double)sd / (double)w) / 32768.0;
  if (m_end_e_rms < m_floor)
    m_floor = m_end_e_rms;
}

/* ============================ GEM / PHY =================================== */

#define SLCR_LOCK_ADDR 0xF8000004U
#define SLCR_UNLOCK_ADDR 0xF8000008U
#define SLCR_GEM0_CLK_CTRL 0xF8000140U
#define SLCR_GEM1_CLK_CTRL 0xF8000144U
#define SLCR_GEM_DIV_MASK 0xFC0FC0FFU

#ifndef XEMACPS_NWCFG_RXOFFS_MASK
#define XEMACPS_NWCFG_RXOFFS_MASK 0x0000C000U
#endif
#define NWCFG_RXOFFS_SHIFT 14
#define RXBUF_LEN_MASK 0x00003FFFU

/* GEM receive status (sticky, write 1 to clear):
 * bit0 buffer not available, bit1 frame received, bit2 RX overrun, bit3 HRESP */
#define GEM_RXSR_CLEAR_ALL 0xFU

static void gem_set_clock(u32 speed_mbps) {
  u32 div0 = 8;
  u32 div1 = (speed_mbps == 1000) ? 1 : (speed_mbps == 100) ? 5 : 50;
  u32 addr =
      (EMAC_BASEADDR == 0xE000B000U) ? SLCR_GEM0_CLK_CTRL : SLCR_GEM1_CLK_CTRL;
  u32 v;

  Xil_Out32(SLCR_UNLOCK_ADDR, 0xDF0D);
  v = (Xil_In32(addr) & SLCR_GEM_DIV_MASK) | (div1 << 20) | (div0 << 8);
  Xil_Out32(addr, v);
  Xil_Out32(SLCR_LOCK_ADDR, 0x767B);
}

static u32 phy_autoneg(void) {
  u16 id, r;
  u32 phy = 32, a, t;

  for (a = 0; a < 32; a++) {
    XEmacPs_PhyRead(&Emac, a, 2, &id);
    if (id != 0xFFFF && id != 0x0000) {
      phy = a;
      break;
    }
  }
  if (phy == 32) {
    xil_printf("ERROR: no PHY found on MDIO\r\n");
    return 0;
  }

  XEmacPs_PhyWrite(&Emac, phy, 0, 0x8000);
  for (t = 0; t < 500; t++) {
    XEmacPs_PhyRead(&Emac, phy, 0, &r);
    if (!(r & 0x8000))
      break;
    usleep(1000);
  }
  XEmacPs_PhyWrite(&Emac, phy, 4, 0x01E1);
  XEmacPs_PhyWrite(&Emac, phy, 9, 0x0200);
  XEmacPs_PhyWrite(&Emac, phy, 0, 0x1200);

  for (t = 0; t < 5000; t++) {
    XEmacPs_PhyRead(&Emac, phy, 1, &r);
    if (r & 0x0020)
      break;
    usleep(1000);
  }
  XEmacPs_PhyRead(&Emac, phy, 1, &r);
  XEmacPs_PhyRead(&Emac, phy, 1, &r);
  if (!(r & 0x0004)) {
    xil_printf("ERROR: Ethernet link down\r\n");
    return 0;
  }

  XEmacPs_PhyRead(&Emac, phy, 10, &r);
  if (r & 0x0800)
    return 1000;
  XEmacPs_PhyRead(&Emac, phy, 5, &r);
  if (r & 0x0180)
    return 100;
  return 10;
}

static int init_emac(void) {
  XEmacPs_Config *cfg;
  XEmacPs_Bd tmpl, *bd;
  XEmacPs_BdRing *rx, *tx;
  u32 i, reg, speed;

  cfg = XEmacPs_LookupConfig(EMAC_DEV_ID);
  if (!cfg)
    return XST_FAILURE;
  if (XEmacPs_CfgInitialize(&Emac, cfg, cfg->BaseAddress) != XST_SUCCESS)
    return XST_FAILURE;

  XEmacPs_SetMdioDivisor(&Emac, MDC_DIV_224);
  XEmacPs_SetMacAddress(&Emac, (void *)BOARD_MAC, 1);
  XEmacPs_SetOptions(&Emac, XEMACPS_JUMBO_ENABLE_OPTION |
                                XEMACPS_FCS_STRIP_OPTION |
                                XEMACPS_BROADCAST_OPTION);

  Xil_SetTlbAttributes((UINTPTR)BdMem, 0xc02);
  rx = &XEmacPs_GetRxRing(&Emac);
  tx = &XEmacPs_GetTxRing(&Emac);

  XEmacPs_BdClear(&tmpl);
  if (XEmacPs_BdRingCreate(rx, (UINTPTR)BdMem, (UINTPTR)BdMem,
                           XEMACPS_BD_ALIGNMENT, NUM_RX_BD) != XST_SUCCESS)
    return XST_FAILURE;
  XEmacPs_BdRingClone(rx, &tmpl, XEMACPS_RECV);

  XEmacPs_BdClear(&tmpl);
  XEmacPs_BdSetStatus(&tmpl, XEMACPS_TXBUF_USED_MASK);
  if (XEmacPs_BdRingCreate(tx, (UINTPTR)BdMem + 0x1000, (UINTPTR)BdMem + 0x1000,
                           XEMACPS_BD_ALIGNMENT, 2) != XST_SUCCESS)
    return XST_FAILURE;
  XEmacPs_BdRingClone(tx, &tmpl, XEMACPS_SEND);

  if (XEmacPs_BdRingAlloc(rx, NUM_RX_BD, &bd) != XST_SUCCESS)
    return XST_FAILURE;
  {
    XEmacPs_Bd *b = bd;
    for (i = 0; i < NUM_RX_BD; i++) {
      XEmacPs_BdSetAddressRx(b, (UINTPTR)RxBuffer[i]);
      XEmacPs_BdClearRxNew(b);
      b = XEmacPs_BdRingNext(rx, b);
    }
  }
  if (XEmacPs_BdRingToHw(rx, NUM_RX_BD, bd) != XST_SUCCESS)
    return XST_FAILURE;

  XEmacPs_SetQueuePtr(&Emac, rx->BaseBdAddr, 0, XEMACPS_RECV);
  XEmacPs_SetQueuePtr(&Emac, tx->BaseBdAddr, 0, XEMACPS_SEND);

  reg = XEmacPs_ReadReg(EMAC_BASEADDR, XEMACPS_DMACR_OFFSET);
  reg &= ~XEMACPS_DMACR_RXBUF_MASK;
  reg |= ((MAX_FRAME_SIZE / 64) << XEMACPS_DMACR_RXBUF_SHIFT) &
         XEMACPS_DMACR_RXBUF_MASK;
  XEmacPs_WriteReg(EMAC_BASEADDR, XEMACPS_DMACR_OFFSET, reg);

  reg = XEmacPs_ReadReg(EMAC_BASEADDR, XEMACPS_NWCFG_OFFSET);
  reg &= ~XEMACPS_NWCFG_RXOFFS_MASK;
  reg |=
      ((u32)RX_ALIGN_OFFSET << NWCFG_RXOFFS_SHIFT) & XEMACPS_NWCFG_RXOFFS_MASK;
  XEmacPs_WriteReg(EMAC_BASEADDR, XEMACPS_NWCFG_OFFSET, reg);

  speed = phy_autoneg();
  if (!speed)
    return XST_FAILURE;
  XEmacPs_SetOperatingSpeed(&Emac, speed);
  gem_set_clock(speed);
  xil_printf("Ethernet link up @ %lu Mbps\r\n", (unsigned long)speed);

  XEmacPs_WriteReg(EMAC_BASEADDR, XEMACPS_RXSR_OFFSET, GEM_RXSR_CLEAR_ALL);
  XEmacPs_Start(&Emac);
  return XST_SUCCESS;
}

/* ===================== RX BD <-> buffer mapping =========================== */

/* BD i always owns RxBuffer[i]; derive i from the BD address. */
static inline u32 bd_index(XEmacPs_BdRing *r, XEmacPs_Bd *b) {
  return (u32)(((UINTPTR)b - (UINTPTR)r->BaseBdAddr) / sizeof(XEmacPs_Bd));
}

#if RECYCLE_BD_EARLY
/* Return one processed BD (must be the oldest posted one) to the hardware. */
static void rx_rearm_one(XEmacPs_BdRing *r, XEmacPs_Bd *old_bd) {
  XEmacPs_Bd *nb;
  XEmacPs_BdRingFree(r, 1, old_bd);
  if (XEmacPs_BdRingAlloc(r, 1, &nb) == XST_SUCCESS) {
    XEmacPs_BdSetAddressRx(nb, (UINTPTR)RxBuffer[bd_index(r, nb)]);
    XEmacPs_BdClearRxNew(nb);
    XEmacPs_BdRingToHw(r, 1, nb);
  }
}
#endif

/* ================================ MAIN ==================================== */

int main(void) {
  XEmacPs_BdRing *rx;
  XEmacPs_Bd *bd_set, *bd;

  u32 frames_done = 0;
  u32 frames_rejected = 0;
  u32 dma_failures = 0;
  u32 seq_gaps = 0;
  u32 seq_resyncs = 0;
  u32 expected_seq = 0;
  u32 total_samples = 0;
  u32 last_mu_q15 = (u32)(u16)(int16_t)(MU_FLOAT * Q15_SCALE);
  int idle_timeout = 0;
  int selftest_done = 0;
  u32 selftest_mismatch = 0;

  XTime neon_ticks_total = 0;  /* NEON pack + DmaTxBuf flush                 */
  XTime dma_ticks_total = 0;   /* DMA transfer + wait + result invalidate    */
  XTime frame_ticks_total = 0; /* whole per-frame handling (accepted frames) */
  XTime scalar_ticks_total = 0; /* scalar pack + flush, BENCH_SCALAR_COMPARE  */
  u32 bench_mismatch = 0;
  XTime t_first = 0, t_last = 0, t_activity = 0;
  int first_seen = 0;

  Xil_ICacheEnable();
  Xil_DCacheEnable();

  xil_printf("\r\n=== Raw-Ethernet + ARM NEON LMS (ZC702) ===\r\n");

  if (init_axi_dma() != XST_SUCCESS) {
    xil_printf("FATAL: DMA init\r\n");
    while (1) {
    }
  }
  if (init_gpio_mu() != XST_SUCCESS) {
    xil_printf("FATAL: GPIO init\r\n");
    while (1) {
    }
  }
  if (init_emac() != XST_SUCCESS) {
    xil_printf("FATAL: EMAC init\r\n");
    while (1) {
    }
  }

  rx = &XEmacPs_GetRxRing(&Emac);

  xil_printf("Waiting for %lu samples (EtherType 0x%04x, AppHdr=%u B, "
             "idle timeout %d s)...\r\n",
             (unsigned long)TARGET_SAMPLES, ACCEPT_ETHERTYPE,
             (unsigned)APP_HDR_BYTES, IDLE_TIMEOUT_SEC);

  while (total_samples < TARGET_SAMPLES) {

    u32 n_bd = XEmacPs_BdRingFromHwRx(rx, NUM_RX_BD, &bd_set);
    if (n_bd == 0) {
      if (first_seen) {
        XTime now;
        XTime_GetTime(&now);
        if ((now - t_activity) > IDLE_TIMEOUT_TICKS) {
          idle_timeout = 1;
          break;
        }
      }
      continue;
    }
    XTime_GetTime(&t_activity);

    bd = bd_set;

    for (u32 k = 0; k < n_bd && total_samples < TARGET_SAMPLES; k++) {

      XEmacPs_Bd *cur = bd;
      bd = XEmacPs_BdRingNext(rx, bd);

      u32 bi = bd_index(rx, cur);
      u8 *buf = RxBuffer[bi];
      u32 stat = XEmacPs_BdRead(cur, XEMACPS_BD_STAT_OFFSET);
      u32 rx_len = stat & RXBUF_LEN_MASK;
      int one_bd =
          (stat & XEMACPS_RXBUF_SOF_MASK) && (stat & XEMACPS_RXBUF_EOF_MASK);
      int accept = one_bd;
      int recycled = 0;
      XTime t_f0, t_f1;

      XTime_GetTime(&t_f0);

      if (accept && rx_len < (ETH_HDR_BYTES + APP_HDR_BYTES + 16U))
        accept = 0;

      /* Invalidate the whole received frame (rounded up to a 32 B cache line;
       * buf is 64 B aligned) BEFORE reading the EtherType, the header or NEON. */
      if (accept) {
        u32 inv_len = (rx_len + RX_ALIGN_OFFSET + 31U) & ~31U;
        if (inv_len > MAX_FRAME_SIZE)
          inv_len = MAX_FRAME_SIZE;
        Xil_DCacheInvalidateRange((UINTPTR)buf, inv_len);
      }

#if ETHERTYPE_FILTER
      if (accept) {
        u16 etype =
            ((u16)buf[RX_ALIGN_OFFSET + 12] << 8) | buf[RX_ALIGN_OFFSET + 13];
        if (etype != ACCEPT_ETHERTYPE)
          accept = 0;
      }
#endif

      u32 N = 0;
      UINTPTR l2_payload = (UINTPTR)buf + RX_ALIGN_OFFSET + ETH_HDR_BYTES;
      const float *x_ptr = NULL;
      const float *d_ptr = NULL;

      if (accept) {
#if USE_APP_HDR
        const app_hdr_t *hdr = (const app_hdr_t *)l2_payload;
        N = hdr->n;
        if (N == 0 || N > MAX_SAMPLES ||
            rx_len < (ETH_HDR_BYTES + APP_HDR_BYTES + N * 16U)) {
          accept = 0;
        } else {
          if (hdr->seq >= expected_seq)
            seq_gaps += hdr->seq - expected_seq; /* frames lost in between */
          else
            seq_resyncs++; /* sender restarted or reordered */
          expected_seq = hdr->seq + 1;

          u32 mu_raw = hdr->mu_q15 & 0xFFFFU;
          if (mu_raw != 0 && mu_raw != last_mu_q15) {
            XGpio_DiscreteWrite(&Gpio, GPIO_CHANNEL, mu_raw);
            last_mu_q15 = mu_raw;
          }
          UINTPTR float_base = l2_payload + APP_HDR_BYTES; /* buf + 32 */
          x_ptr = (const float *)float_base;
          d_ptr = (const float *)(float_base + N * 8U);
        }
#else
        u32 payload_len = rx_len - ETH_HDR_BYTES;
        N = payload_len / 16U;
        if (N == 0 || N > MAX_SAMPLES) {
          accept = 0;
        } else {
          x_ptr = (const float *)l2_payload;
          d_ptr = (const float *)(l2_payload + N * 8U);
        }
#endif
      }

      if (accept) {
        XTime t_n0, t_n1, dma_ticks;
        u32 dma_bytes = N * 8U;

        if (!first_seen) {
          XTime_GetTime(&t_first);
          first_seen = 1;
        }

        /* 1. NEON conversion + DmaTxBuf flush */
        XTime_GetTime(&t_n0);
        neon_pack_xd(x_ptr, d_ptr, DmaTxBuf, N);
        Xil_DCacheFlushRange((UINTPTR)DmaTxBuf, dma_bytes);
        XTime_GetTime(&t_n1);

#if SELFTEST_FIRST_FRAME
        /* Untimed: compare NEON against the scalar reference once. */
        if (!selftest_done) {
          for (u32 i = 0; i < N; i++)
            RefBuf[i] = pack_scalar(x_ptr[2 * i], x_ptr[2 * i + 1],
                                    d_ptr[2 * i], d_ptr[2 * i + 1]);
          for (u32 i = 0; i < N; i++)
            if (RefBuf[i] != DmaTxBuf[i])
              selftest_mismatch++;
          selftest_done = 1;
        }
#endif

#if RECYCLE_BD_EARLY
        /* The RX buffer is no longer needed: let the GEM refill it now. */
        rx_rearm_one(rx, cur);
        recycled = 1;
#endif

        /* 2. FPGA round trip */
        if (fpga_process((UINTPTR)DmaTxBuf, dma_bytes, &dma_ticks) ==
            XST_SUCCESS) {
          XTime_GetTime(&t_f1);
          neon_ticks_total += (t_n1 - t_n0);
          dma_ticks_total += dma_ticks;
          frame_ticks_total += (t_f1 - t_f0);
          total_samples += N;
          frames_done++;
          metrics_update(TxResultBuffer, N); /* not inside any timer */
          XTime_GetTime(&t_last);

#if BENCH_SCALAR_COMPARE
          {
            /* Same frame, scalar path. Invalidate the RX frame again (untimed)
             * so the scalar loop starts from the same cold-cache state NEON had,
             * and include the same output flush in its time. */
            XTime t_s0, t_s1;
            u32 inv_len = (rx_len + RX_ALIGN_OFFSET + 31U) & ~31U;
            if (inv_len > MAX_FRAME_SIZE)
              inv_len = MAX_FRAME_SIZE;
            Xil_DCacheInvalidateRange((UINTPTR)buf, inv_len);

            XTime_GetTime(&t_s0);
            for (u32 i = 0; i < N; i++)
              RefBuf[i] = pack_scalar(x_ptr[2 * i], x_ptr[2 * i + 1],
                                      d_ptr[2 * i], d_ptr[2 * i + 1]);
            Xil_DCacheFlushRange((UINTPTR)RefBuf, dma_bytes);
            XTime_GetTime(&t_s1);
            scalar_ticks_total += (t_s1 - t_s0);

            for (u32 i = 0; i < N; i++)
              if (RefBuf[i] != DmaTxBuf[i])
                bench_mismatch++;
          }
#endif
        } else {
          xil_printf("WARN: DMA failed/timeout on frame, skipped\r\n");
          dma_failures++;
        }
      } else {
        frames_rejected++;
      }

#if RECYCLE_BD_EARLY
      if (!recycled)
        rx_rearm_one(rx, cur);
#else
      (void)recycled;
#endif
    }

    if (total_samples >= TARGET_SAMPLES)
      break;

#if !RECYCLE_BD_EARLY
    /* Batch re-arm: give all n_bd BDs back after the whole batch. */
    XEmacPs_BdRingFree(rx, n_bd, bd_set);
    if (XEmacPs_BdRingAlloc(rx, n_bd, &bd_set) == XST_SUCCESS) {
      bd = bd_set;
      for (u32 k = 0; k < n_bd; k++) {
        XEmacPs_BdSetAddressRx(bd, (UINTPTR)RxBuffer[bd_index(rx, bd)]);
        XEmacPs_BdClearRxNew(bd);
        bd = XEmacPs_BdRingNext(rx, bd);
      }
      XEmacPs_BdRingToHw(rx, n_bd, bd_set);
    }
#endif
  }

  u32 gem_rxsr = XEmacPs_ReadReg(EMAC_BASEADDR, XEMACPS_RXSR_OFFSET);
  XEmacPs_Stop(&Emac);

  /* ---------------------------- RESULTS ------------------------------- */
  {
    double cps = (double)COUNTS_PER_SECOND;
    double nf = (frames_done > 0) ? (double)frames_done : 1.0;

    double t_script = (frames_done > 0) ? (double)(t_last - t_first) / cps : 0.0;
    double thr_fps = (t_script > 0.0) ? (double)frames_done / t_script : 0.0;
    double thr_msps =
        (t_script > 0.0) ? (double)total_samples / t_script / 1.0e6 : 0.0;

    double avg_neon_us = (double)neon_ticks_total / cps * 1.0e6 / nf;
    double avg_dma_us = (double)dma_ticks_total / cps * 1.0e6 / nf;
    double avg_tot_us = (double)frame_ticks_total / cps * 1.0e6 / nf;
    double avg_oth_us = avg_tot_us - avg_neon_us - avg_dma_us;
    double busy_s = (double)frame_ticks_total / cps;
    double board_msps =
        (busy_s > 0.0) ? (double)total_samples / busy_s / 1.0e6 : 0.0;

    double e_start =
        (m_start_n > 0)
            ? sqrt((double)m_start_sumsq / (double)m_start_n) / 32768.0
            : 0.0;
    double e_end = m_end_e_rms;
    double norm_pct = (m_end_d_rms > 0.0) ? 100.0 * e_end / m_end_d_rms : 0.0;
    double red_pct = (e_start > 0.0) ? 100.0 * (1.0 - e_end / e_start) : 0.0;
    double snr_db = (e_end > 0.0 && m_end_d_rms > 0.0)
                        ? 20.0 * log10(m_end_d_rms / e_end)
                        : 0.0;

    xil_printf(
        "\r\n============================================================\r\n");
    xil_printf("FPGA RESULTS - 4-Tap Complex LMS (Raw L2 + ARM NEON)\r\n");
    xil_printf(
        "============================================================\r\n");
    xil_printf("Run ended by           : %s\r\n",
               idle_timeout ? "IDLE TIMEOUT (frames missing)" : "target reached");
    xil_printf("Total frames           : %lu\r\n", (unsigned long)frames_done);
    xil_printf("Total samples          : %lu of %lu\r\n",
               (unsigned long)total_samples, (unsigned long)TARGET_SAMPLES);
    xil_printf("Rejected / DMA fail    : %lu / %lu\r\n",
               (unsigned long)frames_rejected, (unsigned long)dma_failures);
    xil_printf("Seq gaps (lost frames) : %lu   resyncs: %lu\r\n",
               (unsigned long)seq_gaps, (unsigned long)seq_resyncs);
    xil_printf("GEM RX status (sticky) : 0x%lx  [b0 buf-not-avail, b2 overrun]\r\n",
               (unsigned long)gem_rxsr);
#if SELFTEST_FIRST_FRAME
    xil_printf("NEON vs scalar (frame 1): %lu mismatching words\r\n",
               (unsigned long)selftest_mismatch);
#endif
    xil_printf("Total script time      : ");
    pf(t_script, 3);
    xil_printf(" s (first frame -> last frame, includes metrics)\r\n");
    xil_printf("Throughput total       : ");
    pf(thr_fps, 1);
    xil_printf(" frames/s  (");
    pf(thr_msps, 4);
    xil_printf(" MSamples/s)\r\n");
    xil_printf("\r\nTiming (average per frame, microseconds):\r\n");
    xil_printf("t_total  (all handling): ");
    pf(avg_tot_us, 2);
    xil_printf(" us\r\n");
    xil_printf("  |-- t_neon (pack+fl) : ");
    pf(avg_neon_us, 2);
    xil_printf(" us\r\n");
    xil_printf("  |-- t_dma  (PL+inv)  : ");
    pf(avg_dma_us, 2);
    xil_printf(" us\r\n");
    xil_printf("  |-- other (rx inv, hdr, mu): ");
    pf(avg_oth_us, 2);
    xil_printf(" us\r\n");
#if BENCH_SCALAR_COMPARE
    {
      double avg_scalar_us = (double)scalar_ticks_total / cps * 1.0e6 / nf;
      double speedup = (avg_neon_us > 0.0) ? avg_scalar_us / avg_neon_us : 0.0;
      xil_printf("\r\nConversion comparison (same frames, cold cache):\r\n");
      xil_printf("  scalar pack+flush    : ");
      pf(avg_scalar_us, 2);
      xil_printf(" us/frame\r\n");
      xil_printf("  NEON   pack+flush    : ");
      pf(avg_neon_us, 2);
      xil_printf(" us/frame\r\n");
      xil_printf("  NEON speedup         : ");
      pf(speedup, 2);
      xil_printf(" x   (mismatching words over all frames: %lu)\r\n",
                 (unsigned long)bench_mismatch);
      xil_printf("  (throughput/script time above include the extra scalar "
                 "pass; ignore them in this mode)\r\n\r\n");
    }
#endif
    xil_printf("Board-side rate        : ");
    pf(board_msps, 3);
    xil_printf(" MSamples/s (samples / sum of t_total)\r\n");
    xil_printf("\r\nSignal / LMS metrics:\r\n");
    xil_printf("Error RMS start        : ");
    pf(e_start, 6);
    xil_printf("\r\n");
    xil_printf("Error RMS end          : ");
    pf(e_end, 6);
    xil_printf("\r\n");
    xil_printf("Normalized error       : ");
    pf(norm_pct, 2);
    xil_printf("%%\r\n");
    xil_printf("Noise floor approx     : ");
    pf(m_floor, 6);
    xil_printf(" (measured)\r\n");
    xil_printf("Error reduction        : ");
    pf(red_pct, 2);
    xil_printf("%%\r\n");
    xil_printf("Final SNR              : ");
    pf(snr_db, 2);
    xil_printf(" dB\r\n");
    xil_printf(
        "============================================================\r\n");
  }

  Xil_DCacheDisable();
  Xil_ICacheDisable();
  while (1) {
  }
  return 0;
}