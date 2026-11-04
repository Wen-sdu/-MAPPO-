#ifndef VVD_CONTROL_CONFIG_H
#define VVD_CONTROL_CONFIG_H

#include "driverlib.h"
#include "device.h"

/*
 * Fair-comparison contract
 * ------------------------
 * These hardware, calibration, PWM and primary-loop settings intentionally
 * match ccs_deployment/source/mappo_double_loop_config.h. The comparison
 * changes only the secondary controller: MAPPO versus paper equation (24).
 *
 * Typhoon HIL analog-output mapping:
 *   AO1~AO4  : Vdc1~Vdc4 / 50
 *   AO5~AO8  : IL1~IL4   / 10
 *   AO9~AO12 : Vo1~Vo4   / 50
 */
#define VVD_ADC_FULL_SCALE_V             2.5f
#define VVD_ADC_MAX_COUNT                4095.0f
#define VVD_ADC_INPUT_COUNT              12U

/*
 * Bench calibration synchronized with the validated MAPPO signal chain.
 * The fitted raw offsets may be negative; raw code zero is handled
 * explicitly in calibratedRawToEngineering().
 *
 * Vdc two-point measurements (Typhoon command 24 V / 48 V):
 *   AO1: raw 783 / 1572, AO2: raw 782 / 1570
 *   AO3: raw 783 / 1573, AO4: raw 783 / 1573
 * Current gains come from the 0 A / 2 A / 4 A recheck.
 * Vo1/Vo2 use their individual fits; Vo3/Vo4 retain the same provisional
 * averaged fit currently used by MAPPO until individual measurements replace it.
 */
#define VVD_ADC_OFFSET_RAW_VDC1          -6.0f
#define VVD_ADC_OFFSET_RAW_VDC2          -6.0f
#define VVD_ADC_OFFSET_RAW_VDC3          -7.0f
#define VVD_ADC_OFFSET_RAW_VDC4          -7.0f
#define VVD_ADC_OFFSET_RAW_IL1            0.0f
#define VVD_ADC_OFFSET_RAW_IL2            0.0f
#define VVD_ADC_OFFSET_RAW_IL3            0.0f
#define VVD_ADC_OFFSET_RAW_IL4            0.0f
#define VVD_ADC_OFFSET_RAW_VO1           -6.42f
#define VVD_ADC_OFFSET_RAW_VO2           -4.41f
#define VVD_ADC_OFFSET_RAW_VO3           -5.42f
#define VVD_ADC_OFFSET_RAW_VO4           -5.42f

#define VVD_ADC_GAIN_VDC1_V_PER_COUNT \
    (48.0f / (1572.0f - VVD_ADC_OFFSET_RAW_VDC1))
#define VVD_ADC_GAIN_VDC2_V_PER_COUNT \
    (48.0f / (1570.0f - VVD_ADC_OFFSET_RAW_VDC2))
#define VVD_ADC_GAIN_VDC3_V_PER_COUNT \
    (48.0f / (1573.0f - VVD_ADC_OFFSET_RAW_VDC3))
#define VVD_ADC_GAIN_VDC4_V_PER_COUNT \
    (48.0f / (1573.0f - VVD_ADC_OFFSET_RAW_VDC4))
#define VVD_ADC_GAIN_IL1_A_PER_COUNT      0.00615f
#define VVD_ADC_GAIN_IL2_A_PER_COUNT      0.00613f
#define VVD_ADC_GAIN_IL3_A_PER_COUNT      0.00613f
#define VVD_ADC_GAIN_IL4_A_PER_COUNT      0.00614f
#define VVD_ADC_GAIN_VO1_V_PER_COUNT      0.03040574f
#define VVD_ADC_GAIN_VO2_V_PER_COUNT      0.03044087f
#define VVD_ADC_GAIN_VO3_V_PER_COUNT      0.03042330f
#define VVD_ADC_GAIN_VO4_V_PER_COUNT      0.03042330f

#define VVD_ADC_ACQ_WINDOW               64U
#define VVD_ADC_MEDIAN_SAMPLES            5U
#define VVD_ADC_VOLTAGE_FILTER_ALPHA      0.20f

/* Vdc1~Vdc4. */
#define VVD_ADC_BASE_VDC1                ADCA_BASE
#define VVD_ADC_RESULT_VDC1              ADCARESULT_BASE
#define VVD_ADC_CH_VDC1                  ADC_CH_ADCIN0
#define VVD_ADC_BASE_VDC2                ADCA_BASE
#define VVD_ADC_RESULT_VDC2              ADCARESULT_BASE
#define VVD_ADC_CH_VDC2                  ADC_CH_ADCIN1
#define VVD_ADC_BASE_VDC3                ADCA_BASE
#define VVD_ADC_RESULT_VDC3              ADCARESULT_BASE
#define VVD_ADC_CH_VDC3                  ADC_CH_ADCIN14
#define VVD_ADC_BASE_VDC4                ADCA_BASE
#define VVD_ADC_RESULT_VDC4              ADCARESULT_BASE
#define VVD_ADC_CH_VDC4                  ADC_CH_ADCIN3

/* IL1~IL4. */
#define VVD_ADC_BASE_IL1                 ADCA_BASE
#define VVD_ADC_RESULT_IL1               ADCARESULT_BASE
#define VVD_ADC_CH_IL1                   ADC_CH_ADCIN4
#define VVD_ADC_BASE_IL2                 ADCA_BASE
#define VVD_ADC_RESULT_IL2               ADCARESULT_BASE
#define VVD_ADC_CH_IL2                   ADC_CH_ADCIN5
#define VVD_ADC_BASE_IL3                 ADCA_BASE
#define VVD_ADC_RESULT_IL3               ADCARESULT_BASE
#define VVD_ADC_CH_IL3                   ADC_CH_ADCIN11
#define VVD_ADC_BASE_IL4                 ADCA_BASE
#define VVD_ADC_RESULT_IL4               ADCARESULT_BASE
#define VVD_ADC_CH_IL4                   ADC_CH_ADCIN7

/* Vo1~Vo4: converter-side output voltages used by the primary voltage loop. */
#define VVD_ADC_BASE_VO1                 ADCA_BASE
#define VVD_ADC_RESULT_VO1               ADCARESULT_BASE
#define VVD_ADC_CH_VO1                   ADC_CH_ADCIN9
#define VVD_ADC_GPIO_VO1                 227U
#define VVD_ADC_BASE_VO2                 ADCB_BASE
#define VVD_ADC_RESULT_VO2               ADCBRESULT_BASE
#define VVD_ADC_CH_VO2                   ADC_CH_ADCIN0
#define VVD_ADC_GPIO_VO2                 253U
#define VVD_ADC_BASE_VO3                 ADCA_BASE
#define VVD_ADC_RESULT_VO3               ADCARESULT_BASE
#define VVD_ADC_CH_VO3                   ADC_CH_ADCIN10
#define VVD_ADC_GPIO_VO3                 230U
#define VVD_ADC_BASE_VO4                 ADCA_BASE
#define VVD_ADC_RESULT_VO4               ADCARESULT_BASE
#define VVD_ADC_CH_VO4                   ADC_CH_ADCIN12

/* PWM routing and switching frequency, identical to MAPPO. */
#define VVD_PWM_FREQ_HZ                  10000UL
#define VVD_EPWM_TBCLK_HZ                DEVICE_SYSCLK_FREQ
#define VVD_FIXED_DUTY                   0.48f
#define VVD_PWM1_BASE                    EPWM1_BASE
#define VVD_PWM2_BASE                    EPWM2_BASE
#define VVD_PWM3_BASE                    EPWM3_BASE
#define VVD_PWM4_BASE                    EPWM4_BASE
#define VVD_PWM1_GPIO_MUX                GPIO_0_EPWM1_A
#define VVD_PWM2_GPIO_MUX                GPIO_2_EPWM2_A
#define VVD_PWM3_GPIO_MUX                GPIO_4_EPWM3_A
#define VVD_PWM4_GPIO_MUX                GPIO_6_EPWM4_A
#define VVD_PWM1_GPIO                    0U
#define VVD_PWM2_GPIO                    2U
#define VVD_PWM3_GPIO                    4U
#define VVD_PWM4_GPIO                    6U

/* Deterministic multirate schedule derived from the 10 kHz PWM/ADC ISR. */
#define VVD_CURRENT_LOOP_TS_SEC          0.0001f
#define VVD_VOLTAGE_LOOP_DIVIDER         10U
#define VVD_VOLTAGE_LOOP_TS_SEC          0.001f
#define VVD_PRIMARY_AVERAGE_SAMPLES      VVD_VOLTAGE_LOOP_DIVIDER
#define VVD_SECONDARY_LOOP_TS_SEC        VVD_VOLTAGE_LOOP_TS_SEC

/* Primary cascaded voltage/current PI, identical to MAPPO. */
#define VVD_NOMINAL_VOLTAGE_V            48.0f
#define VVD_VOLTAGE_PI_KP_A_PER_V         0.30f
#define VVD_VOLTAGE_PI_KI_A_PER_VS       25.0f
#define VVD_CURRENT_PI_KP_DUTY_PER_A      0.12f
#define VVD_CURRENT_PI_KI_DUTY_PER_AS     0.25f
#define VVD_DUTY_MIN                      0.00f
#define VVD_DUTY_MAX                      1.00f

/*
 * Modes exposed through g_panel.mode:
 *   0: fixed PWM safety baseline
 *   1: primary droop + cascaded PI, without secondary control
 *   2: paper VVD distributed secondary controller + same primary control
 */
#define VVD_CONTROL_MODE_FIXED           0U
#define VVD_CONTROL_MODE_DROOP           1U
#define VVD_CONTROL_MODE_SECONDARY       2U

/* Same 1:2:1:1 capacity target and same primary droop as MAPPO. */
#define VVD_CAPACITY_RATIO_1             1.0f
#define VVD_CAPACITY_RATIO_2             2.0f
#define VVD_CAPACITY_RATIO_3             1.0f
#define VVD_CAPACITY_RATIO_4             1.0f
#define VVD_DROOP_BASE_OHM               1.00f

/* Converter-to-bus resistances in the common Typhoon controlled plant. */
#define VVD_LOCAL_R1_OHM                 0.010f
#define VVD_LOCAL_R2_OHM                 0.005f
#define VVD_LOCAL_R3_OHM                 0.008f
#define VVD_LOCAL_R4_OHM                 0.008f

/* Paper four-bus equation-(24) gains; tune only after fair baseline runs. */
#define VVD_SECONDARY_KI_PER_SEC         15.0f
#define VVD_DAC_ALPHA_PER_SEC            50.0f
#define VVD_DAC_BETA_PER_SEC            100.0f

/* Runtime tuning bounds. */
#define VVD_VOLTAGE_PI_KP_MAX             2.0f
#define VVD_VOLTAGE_PI_KI_MAX           100.0f
#define VVD_CURRENT_PI_KP_MAX             0.50f
#define VVD_CURRENT_PI_KI_MAX             2.00f
#define VVD_DROOP_BASE_MIN_OHM            0.05f
#define VVD_DROOP_BASE_MAX_OHM            2.00f

/*
 * Numerical guards. These are wider than the normal operating range and
 * must remain inactive in a valid comparison. Duty saturation is the common
 * physical actuator limit for both controllers.
 */
#define VVD_SECONDARY_MIN_V             -24.0f
#define VVD_SECONDARY_MAX_V              24.0f
#define VVD_X_STATE_LIMIT_V              48.0f
#define VVD_ERROR_INTEGRAL_LIMIT_VS       4.0f

/* Online monitoring only; it does not alter control actions. */
#define VVD_CURRENT_BASE_A               10.0f
#define VVD_VOLTAGE_SAFE_FRACTION         0.05f
#define VVD_AVG_VOLTAGE_ERROR_TARGET      0.02f
#define VVD_CURRENT_ERROR_TARGET          0.08f
#define VVD_PASS_HOLD_SAMPLES          1000U

#endif
