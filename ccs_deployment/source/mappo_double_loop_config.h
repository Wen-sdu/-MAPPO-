#ifndef MAPPO_DOUBLE_LOOP_CONFIG_H
#define MAPPO_DOUBLE_LOOP_CONFIG_H

#include "driverlib.h"
#include "device.h"

/*
 * HIL404 analog output scaling currently used in Typhoon:
 *   AO1~AO4: Vdc / 50  -> F28P55 ADC voltage * 50 = Vdc
 *   AO5~AO8: IL  / 10  -> F28P55 ADC voltage * 10 = IL
 *   AO9~AO12: Vo / 50  -> F28P55 ADC voltage * 50 = converter-side Vo
 */
/*
 * LAUNCHXL-F28P55X ADC readings in this HIL setup match a 2.5 V ADC reference:
 *   0.50 V input -> raw ~= 820 counts
 *   0.25 V input -> raw ~= 410 counts
 * If your board is configured for a 3.3 V ADC reference, change this to 3.3f.
 */
#define HIL_ADC_FULL_SCALE_V      2.5f
#define HIL_ADC_MAX_COUNT         4095.0f
#define HIL_VDC_SCALE              50.0f
#define HIL_IL_SCALE               10.0f
#define HIL_VO_SCALE               50.0f
#define HIL_ADC_INPUT_COUNT       12U

/*
 * Per-channel calibration measured through the complete
 * Typhoon-AO -> wiring -> F28P55 ADC chain.
 *
 * The zero and span values below are the centres estimated from the CCS
 * captures supplied during commissioning.  Keep them channel-specific so a
 * later CSV/automatic-statistics measurement can refine one channel without
 * changing the conversion code.
 *
 *   raw == 0: engineering_value = 0
 *   raw >  0: engineering_value = max(raw - offset_raw, 0) * gain_per_count
 *
 * Vdc1~Vdc4 were recalibrated on 2026-10-19 from the supplied CCS captures:
 *
 *                 24 V point       48 V point       fitted raw offset
 *   AO1/Vdc1      raw ~= 783       raw ~= 1572      -6 counts
 *   AO2/Vdc2      raw ~= 782       raw ~= 1570      -6 counts
 *   AO3/Vdc3      raw ~= 783       raw ~= 1573      -7 counts
 *   AO4/Vdc4      raw ~= 783       raw ~= 1573      -7 counts
 *
 * The explicit raw-zero handling is required because the unipolar ADC clips
 * the fitted negative intercept at zero.  Replacing the fitted offsets with
 * zero would leave roughly 0.1 V of midpoint error.
 *
 * Voltage span: 48 V engineering value (0.96 V at the HIL AO).
 * Current span:  2 A engineering value (0.20 V at the HIL AO).
 * IL1~IL4 were rechecked on 2026-10-19 at 0 A, 2 A, and 4 A.  The
 * zero-input raw counts were approximately zero, so the old 10-count offsets
 * were removed and the gains below were fitted to the observed working range.
 * Revalidate at 0/1/2/3/4 A after rebuilding and flashing the firmware.
 *
 * Vo1 and Vo2 were recalibrated on 2026-10-19 from the supplied CCS captures:
 *
 *                 24 V point       48 V point       fitted raw offset
 *   AO9/Vo1       raw ~= 782.91    raw ~= 1572.23   -6.42 counts
 *   AO10/Vo2      raw ~= 784.00    raw ~= 1572.41   -4.41 counts
 *
 * Vo3 and Vo4 temporarily use the mean of the measured Vo1/Vo2 calibration.
 * Replace their provisional values after AO11 and AO12 are measured at
 * 0/24/48 V.  The raw-zero special case in calibratedRawToEngineering()
 * preserves an exact zero reading despite the fitted negative intercepts.
 */
#define HIL_ADC_OFFSET_RAW_VDC1    -6.0f
#define HIL_ADC_OFFSET_RAW_VDC2    -6.0f
#define HIL_ADC_OFFSET_RAW_VDC3    -7.0f
#define HIL_ADC_OFFSET_RAW_VDC4    -7.0f
#define HIL_ADC_OFFSET_RAW_IL1      0.0f
#define HIL_ADC_OFFSET_RAW_IL2      0.0f
#define HIL_ADC_OFFSET_RAW_IL3      0.0f
#define HIL_ADC_OFFSET_RAW_IL4      0.0f
#define HIL_ADC_OFFSET_RAW_VO1     -6.42f
#define HIL_ADC_OFFSET_RAW_VO2     -4.41f
#define HIL_ADC_OFFSET_RAW_VO3     -5.42f
#define HIL_ADC_OFFSET_RAW_VO4     -5.42f

#define HIL_ADC_GAIN_VDC1_V_PER_COUNT (48.0f / (1572.0f - HIL_ADC_OFFSET_RAW_VDC1))
#define HIL_ADC_GAIN_VDC2_V_PER_COUNT (48.0f / (1570.0f - HIL_ADC_OFFSET_RAW_VDC2))
#define HIL_ADC_GAIN_VDC3_V_PER_COUNT (48.0f / (1573.0f - HIL_ADC_OFFSET_RAW_VDC3))
#define HIL_ADC_GAIN_VDC4_V_PER_COUNT (48.0f / (1573.0f - HIL_ADC_OFFSET_RAW_VDC4))
#define HIL_ADC_GAIN_IL1_A_PER_COUNT   0.00615f
#define HIL_ADC_GAIN_IL2_A_PER_COUNT   0.00613f
#define HIL_ADC_GAIN_IL3_A_PER_COUNT   0.00613f
#define HIL_ADC_GAIN_IL4_A_PER_COUNT   0.00614f
#define HIL_ADC_GAIN_VO1_V_PER_COUNT   0.03040574f
#define HIL_ADC_GAIN_VO2_V_PER_COUNT   0.03044087f
#define HIL_ADC_GAIN_VO3_V_PER_COUNT   0.03042330f
#define HIL_ADC_GAIN_VO4_V_PER_COUNT   0.03042330f

/*
 * ADC anti-glitch settings.
 *
 * Each input owns one fixed SOC.  In the double-loop candidate, ePWM1 SOCA
 * triggers one complete ADC scan at a fixed PWM phase.  Vdc uses a rolling
 * five-scan median followed by a light IIR; IL goes directly to the 10 kHz
 * current loop after calibration; Vo is averaged over each 1 ms voltage-loop
 * interval.  No input performs multiple blocking ADC scans per interrupt.
 */
#define HIL_ADC_ACQ_WINDOW        64U
#define HIL_ADC_MEDIAN_SAMPLES     5U
#define HIL_ADC_VOLTAGE_FILTER_ALPHA 0.20f
#define HIL_ADC_CURRENT_FILTER_ALPHA 0.50f
#define HIL_ADC_RAW_SPREAD_WARN   200U

/*
 * Edit these ADC channel macros to match the F28P55 board pins you wire to.
 * Defaults use ADCA0~ADCA7 as a convenient placeholder.
 */
#define HIL_ADC_BASE_VDC1         ADCA_BASE
#define HIL_ADC_RESULT_VDC1       ADCARESULT_BASE
#define HIL_ADC_CH_VDC1           ADC_CH_ADCIN0

#define HIL_ADC_BASE_VDC2         ADCA_BASE
#define HIL_ADC_RESULT_VDC2       ADCARESULT_BASE
#define HIL_ADC_CH_VDC2           ADC_CH_ADCIN1

/*
 * Avoid ADCINA2/B6/PGA1_P for Vdc3 on the current bench board: ADCA2 and ADCB6
 * both read incorrectly. Wire HIL AO3 to the board pin labelled
 * ADCINA14/B14/C4 and sample it through ADCA14.
 */
#define HIL_ADC_BASE_VDC3         ADCA_BASE
#define HIL_ADC_RESULT_VDC3       ADCARESULT_BASE
#define HIL_ADC_CH_VDC3           ADC_CH_ADCIN14

#define HIL_ADC_BASE_VDC4         ADCA_BASE
#define HIL_ADC_RESULT_VDC4       ADCARESULT_BASE
#define HIL_ADC_CH_VDC4           ADC_CH_ADCIN3

#define HIL_ADC_BASE_IL1          ADCA_BASE
#define HIL_ADC_RESULT_IL1        ADCARESULT_BASE
#define HIL_ADC_CH_IL1            ADC_CH_ADCIN4

#define HIL_ADC_BASE_IL2          ADCA_BASE
#define HIL_ADC_RESULT_IL2        ADCARESULT_BASE
#define HIL_ADC_CH_IL2            ADC_CH_ADCIN5

#define HIL_ADC_BASE_IL3          ADCA_BASE
#define HIL_ADC_RESULT_IL3        ADCARESULT_BASE
/* HIL AO7 is wired to ADCINA11/J3-25; ADCINA6 gave low/dropout readings. */
#define HIL_ADC_CH_IL3            ADC_CH_ADCIN11

#define HIL_ADC_BASE_IL4          ADCA_BASE
#define HIL_ADC_RESULT_IL4        ADCARESULT_BASE
#define HIL_ADC_CH_IL4            ADC_CH_ADCIN7

/*
 * AO9~AO12 carry Vo1~Vo4.  The selections below are all exposed on the
 * LAUNCHXL-F28P55X headers and do not overlap the eight existing Vdc/IL inputs.
 *   AO9  -> Vo1 -> J5-46 -> ADCINA9
 *   AO10 -> Vo2 -> J7-63 -> ADCINB0
 *   AO11 -> Vo3 -> J7-64 -> ADCINA10
 *   AO12 -> Vo4 -> J7-69 -> ADCINA12
 */
#define HIL_ADC_BASE_VO1          ADCA_BASE
#define HIL_ADC_RESULT_VO1        ADCARESULT_BASE
#define HIL_ADC_CH_VO1            ADC_CH_ADCIN9
#define HIL_ADC_GPIO_VO1          227U

#define HIL_ADC_BASE_VO2          ADCB_BASE
#define HIL_ADC_RESULT_VO2        ADCBRESULT_BASE
#define HIL_ADC_CH_VO2            ADC_CH_ADCIN0
#define HIL_ADC_GPIO_VO2          253U

#define HIL_ADC_BASE_VO3          ADCA_BASE
#define HIL_ADC_RESULT_VO3        ADCARESULT_BASE
#define HIL_ADC_CH_VO3            ADC_CH_ADCIN10
#define HIL_ADC_GPIO_VO3          230U

#define HIL_ADC_BASE_VO4          ADCA_BASE
#define HIL_ADC_RESULT_VO4        ADCARESULT_BASE
#define HIL_ADC_CH_VO4            ADC_CH_ADCIN12

/*
 * PWM output defaults. Change GPIO mux constants if your board routes ePWM
 * outputs to different pins.
 */
#define HIL_PWM_FREQ_HZ           10000UL
/* EPWM TBCLK prescalers are both /1, so TBCLK follows the 150 MHz SYSCLK. */
#define HIL_EPWM_TBCLK_HZ         DEVICE_SYSCLK_FREQ
#define HIL_FIXED_DUTY            0.48f

#define HIL_PWM1_BASE             EPWM1_BASE
#define HIL_PWM2_BASE             EPWM2_BASE
#define HIL_PWM3_BASE             EPWM3_BASE
#define HIL_PWM4_BASE             EPWM4_BASE

#define HIL_PWM1_GPIO_MUX         GPIO_0_EPWM1_A
#define HIL_PWM2_GPIO_MUX         GPIO_2_EPWM2_A
#define HIL_PWM3_GPIO_MUX         GPIO_4_EPWM3_A
#define HIL_PWM4_GPIO_MUX         GPIO_6_EPWM4_A

#define HIL_PWM1_GPIO             0U
#define HIL_PWM2_GPIO             2U
#define HIL_PWM3_GPIO             4U
#define HIL_PWM4_GPIO             6U

/*
 * Deterministic multi-rate schedule, all derived from the 10 kHz ePWM/ADC
 * interrupt.  No software delay is used as a control-period generator.
 */
#define HIL_CURRENT_LOOP_TS_SEC       0.0001f
#define HIL_VOLTAGE_LOOP_DIVIDER      10U
#define HIL_VOLTAGE_LOOP_TS_SEC       0.001f
#define HIL_VO_AVERAGE_SAMPLES        HIL_VOLTAGE_LOOP_DIVIDER
#define HIL_SECONDARY_LOOP_DIVIDER    100U
#define HIL_SECONDARY_LOOP_TS_SEC     0.010f

/*
 * Actor deployment conditioning.  The current PI uses the latest calibrated
 * IL sample.  At 1 kHz, the droop equation uses a coherent 1 ms IL average
 * and the voltage PI uses the matching calibrated 1 ms Vo average.  At each
 * 100 Hz boundary the actor observes the latest 1 ms IL average and current
 * filtered Vdc, matching the training model's point observation without an
 * extra moving-average delay.  Its four corrections are published atomically.
 * The actor differential action and the common-voltage PI use independent
 * filters.  MAPPO remains at 100 Hz, while the inexpensive common-voltage PI
 * runs with the 1 kHz voltage loop so a load event does not wait for the next
 * neural-network inference boundary.
 */
#define HIL_MAPPO_OBSERVATION_PERIOD_SAMPLES  HIL_SECONDARY_LOOP_DIVIDER
#define HIL_MAPPO_ACTION_FILTER_ALPHA          0.10f
#define HIL_COMMON_ACTION_FILTER_ALPHA          1.00f
#define HIL_COMMON_LOOP_TS_SEC                  HIL_VOLTAGE_LOOP_TS_SEC
#define HIL_COMMON_VOLTAGE_KP                   0.80f
#define HIL_COMMON_VOLTAGE_KI                   8.00f
#define HIL_COMMON_VOLTAGE_MIN_V               -2.00f
#define HIL_COMMON_VOLTAGE_MAX_V                8.00f
#define HIL_COMMON_CURRENT_FEEDFORWARD_GAIN      0.80f
#define HIL_COMMON_FEEDFORWARD_FILTER_ALPHA      1.00f

/*
 * Cascaded controller. The voltage PI outputs an inductor-current reference;
 * the current PI directly generates PWM duty. No duty feed-forward is used.
 * These are conservative first-pass HIL gains and must be commissioned in
 * mode 1 before enabling the MAPPO secondary controller.
 */
#define HIL_PI_VREF               48.0f
#define HIL_VOLTAGE_PI_KP_A_PER_V     0.30f
#define HIL_VOLTAGE_PI_KI_A_PER_VS   25.0f
#define HIL_CURRENT_PI_KP_DUTY_PER_A  0.12f
#define HIL_CURRENT_PI_KI_DUTY_PER_AS 0.25f
#define HIL_DUTY_MIN              0.00f
#define HIL_DUTY_MAX              1.00f
#define HIL_DROOP_BASE_V_PER_A     1.00f
#define HIL_CURRENT_BASE          10.0f

/*
 * CCS control-panel modes used by this independent experiment:
 *   0: fixed PWM (safe fallback)
 *   1: cascaded voltage/current PI only (commissioning)
 *   2: MAPPO u_i + primary droop + cascaded PI
 */
#define HIL_CONTROL_MODE_FIXED       0U
#define HIL_CONTROL_MODE_DROOP       1U
#define HIL_CONTROL_MODE_MAPPO_OPT   2U

/*
 * Real trained MAPPO actor deployment.
 * The actor is evaluated as a slower secondary controller. Its output is a raw
 * continuous action in volts, mapped exactly like the Python environment:
 *   du_i = raw_action_i - mean(raw_action)
 *   u_ff = 0.8 * mean(droop_i * IL_i)
 *   u_common = PI(48 - mean(Vdc)) + u_ff
 *   u_i = u_common + du_i
 *   Vcmd_i = 48 - (droop_base/capacity_i)*IL_i + u_i
 * The Actor therefore handles differential current sharing and the explicit
 * common PI restores average Vdc without changing the six-element observation.
 */
#define HIL_MAPPO_OBS_VOLTAGE_DELTA_V   9.6f
#define HIL_MAPPO_LAYER_NORM_EPS        0.00001f

/* Runtime clamps for the CCS tuning fields. */
#define HIL_VOLTAGE_PI_KP_MAX                 2.0f
#define HIL_VOLTAGE_PI_KI_MAX               100.0f
#define HIL_CURRENT_PI_KP_MAX                 0.50f
#define HIL_CURRENT_PI_KI_MAX                 2.00f

#endif
