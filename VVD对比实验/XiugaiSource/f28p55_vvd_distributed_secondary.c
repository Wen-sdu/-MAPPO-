#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "driverlib.h"
#include "device.h"
#include "vvd_control_config.h"

/*
 * Four-bus VVD comparison firmware.
 *
 * Hardware acquisition, PWM generation and primary cascaded PI control are
 * deliberately identical to the MAPPO deployment.  Only the secondary
 * controller differs and implements equation (24) of Liu et al.:
 *
 *   Vbar_i = (k_i + R_i) I_i
 *   dx_i/dt = -alpha*x_i - beta*sum_j a_ij(h_i-h_j)
 *   h_i = x_i + Vbar_i
 *   u_i = h_i - K_I*integral(Vbar_i-h_i)
 *   Vref_i = 48 - k_i*I_i + u_i
 *
 * Vdc is used only for experiment metrics.  The controller uses IL for the
 * VVD/droop calculation and Vo for the primary voltage feedback, preserving
 * the paper's no-bus-voltage-feedback property.
 */

typedef struct {
    uint32_t adcBase;
    uint32_t resultBase;
    ADC_Channel channel;
    ADC_SOCNumber socNumber;
    float offsetRaw;
    float gainEngineeringPerCount;
} VvdAdcInput;

typedef struct {
    float vdc[4];
    float il[4];
    float vo[4];
    float adcVoltage[VVD_ADC_INPUT_COUNT];
    float adcCalibrated[VVD_ADC_INPUT_COUNT];
    uint16_t adcRaw[VVD_ADC_INPUT_COUNT];
} VvdMeasurements;

typedef struct {
    float averageVoltageV;
    float averageVoltageErrorPu;
    float maxVoltageDeviationV;
    float minBusVoltageV;
    float maxBusVoltageV;
    float shareLambdaA;
    float currentPerCapacityA[4];
    float targetCurrentA[4];
    float currentErrorPu;
    uint16_t voltagePass;
    uint16_t averageVoltagePass;
    uint16_t currentPass;
    uint16_t overallPass;
} VvdMetrics;

/* Variables intended for CCS Expressions/Graph monitoring. */
typedef struct {
    uint16_t mode;
    uint16_t reset;

    float nominalVoltageV;
    float droopBaseOhm;
    float secondaryKi;
    float dacAlpha;
    float dacBeta;
    float voltagePiKp;
    float voltagePiKi;
    float currentPiKp;
    float currentPiKi;

    float averageVoltageV;
    float maxVoltageDeviationV;
    float minBusVoltageV;
    float maxBusVoltageV;
    float currentErrorPu;
    float shareLambdaA;

    float droopGainOhm[4];
    float localResistanceOhm[4];
    float converterVoltageV[4];
    float virtualVoltageDropV[4];
    float averageVvdEstimateV[4];
    float secondaryVoltageV[4];
    float voltageReferenceOutV[4];
    float currentReferenceA[4];
    float voltageLoopErrorV[4];
    float currentLoopErrorA[4];
    float currentPerCapacityA[4];
    float targetCurrentA[4];

    uint16_t voltagePass;
    uint16_t averageVoltagePass;
    uint16_t currentPass;
    uint16_t secondarySaturated;
    uint16_t dutySaturated;
    uint16_t overallPass;
    uint16_t passHoldCount;
    uint16_t qualifiedPass;
    uint32_t adcOverflowCount;
    uint32_t currentLoopCount;
    uint32_t voltageLoopCount;
} VvdControlPanel;

typedef struct {
    float x[4];
    float h[4];
    float shareErrorIntegral[4];
    float u[4];
} VvdControllerState;

volatile VvdMeasurements g_hil;
volatile VvdMetrics g_metrics;
volatile VvdControlPanel g_panel = {
    .mode = VVD_CONTROL_MODE_FIXED,
    .reset = 0U,
    .nominalVoltageV = VVD_NOMINAL_VOLTAGE_V,
    .droopBaseOhm = VVD_DROOP_BASE_OHM,
    .secondaryKi = VVD_SECONDARY_KI_PER_SEC,
    .dacAlpha = VVD_DAC_ALPHA_PER_SEC,
    .dacBeta = VVD_DAC_BETA_PER_SEC,
    .voltagePiKp = VVD_VOLTAGE_PI_KP_A_PER_V,
    .voltagePiKi = VVD_VOLTAGE_PI_KI_A_PER_VS,
    .currentPiKp = VVD_CURRENT_PI_KP_DUTY_PER_A,
    .currentPiKi = VVD_CURRENT_PI_KI_DUTY_PER_AS
};

volatile float g_pwmDuty[4] = {
    VVD_FIXED_DUTY, VVD_FIXED_DUTY, VVD_FIXED_DUTY, VVD_FIXED_DUTY
};
volatile float g_voltageReferenceV[4] = {
    VVD_NOMINAL_VOLTAGE_V, VVD_NOMINAL_VOLTAGE_V,
    VVD_NOMINAL_VOLTAGE_V, VVD_NOMINAL_VOLTAGE_V
};
volatile float g_currentReferenceA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_voAveragedV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_ilVoltageLoopAveragedA[4] = {0.0f, 0.0f, 0.0f, 0.0f};

static VvdControllerState g_vvd;
static float g_voltagePiIntegralA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static float g_currentPiIntegralDuty[4] = {
    VVD_FIXED_DUTY, VVD_FIXED_DUTY, VVD_FIXED_DUTY, VVD_FIXED_DUTY
};
static uint16_t g_lastControlMode = VVD_CONTROL_MODE_FIXED;
volatile uint16_t g_activeControlMode = VVD_CONTROL_MODE_FIXED;
static uint16_t g_voltageLoopDivider = 0U;
static volatile uint16_t g_metricsPending = 0U;

static uint16_t g_adcRawHistory[4][VVD_ADC_MEDIAN_SAMPLES];
static float g_adcFilteredEngineering[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint16_t g_adcHistoryIndex = 0U;
static bool g_adcFilterPrimed = false;
static float g_voSampleSumV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static float g_ilVoltageLoopSampleSumA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint16_t g_primaryAverageCount = 0U;

static const float kCapacityRatio[4] = {
    VVD_CAPACITY_RATIO_1,
    VVD_CAPACITY_RATIO_2,
    VVD_CAPACITY_RATIO_3,
    VVD_CAPACITY_RATIO_4
};

static const float kLocalResistanceOhm[4] = {
    VVD_LOCAL_R1_OHM,
    VVD_LOCAL_R2_OHM,
    VVD_LOCAL_R3_OHM,
    VVD_LOCAL_R4_OHM
};

static const VvdAdcInput kInputs[VVD_ADC_INPUT_COUNT] = {
    {VVD_ADC_BASE_VDC1, VVD_ADC_RESULT_VDC1, VVD_ADC_CH_VDC1,
     ADC_SOC_NUMBER0, VVD_ADC_OFFSET_RAW_VDC1,
     VVD_ADC_GAIN_VDC1_V_PER_COUNT},
    {VVD_ADC_BASE_VDC2, VVD_ADC_RESULT_VDC2, VVD_ADC_CH_VDC2,
     ADC_SOC_NUMBER1, VVD_ADC_OFFSET_RAW_VDC2,
     VVD_ADC_GAIN_VDC2_V_PER_COUNT},
    {VVD_ADC_BASE_VDC3, VVD_ADC_RESULT_VDC3, VVD_ADC_CH_VDC3,
     ADC_SOC_NUMBER2, VVD_ADC_OFFSET_RAW_VDC3,
     VVD_ADC_GAIN_VDC3_V_PER_COUNT},
    {VVD_ADC_BASE_VDC4, VVD_ADC_RESULT_VDC4, VVD_ADC_CH_VDC4,
     ADC_SOC_NUMBER3, VVD_ADC_OFFSET_RAW_VDC4,
     VVD_ADC_GAIN_VDC4_V_PER_COUNT},
    {VVD_ADC_BASE_IL1, VVD_ADC_RESULT_IL1, VVD_ADC_CH_IL1,
     ADC_SOC_NUMBER4, VVD_ADC_OFFSET_RAW_IL1,
     VVD_ADC_GAIN_IL1_A_PER_COUNT},
    {VVD_ADC_BASE_IL2, VVD_ADC_RESULT_IL2, VVD_ADC_CH_IL2,
     ADC_SOC_NUMBER5, VVD_ADC_OFFSET_RAW_IL2,
     VVD_ADC_GAIN_IL2_A_PER_COUNT},
    {VVD_ADC_BASE_IL3, VVD_ADC_RESULT_IL3, VVD_ADC_CH_IL3,
     ADC_SOC_NUMBER6, VVD_ADC_OFFSET_RAW_IL3,
     VVD_ADC_GAIN_IL3_A_PER_COUNT},
    {VVD_ADC_BASE_IL4, VVD_ADC_RESULT_IL4, VVD_ADC_CH_IL4,
     ADC_SOC_NUMBER7, VVD_ADC_OFFSET_RAW_IL4,
     VVD_ADC_GAIN_IL4_A_PER_COUNT},
    {VVD_ADC_BASE_VO1, VVD_ADC_RESULT_VO1, VVD_ADC_CH_VO1,
     ADC_SOC_NUMBER8, VVD_ADC_OFFSET_RAW_VO1,
     VVD_ADC_GAIN_VO1_V_PER_COUNT},
    {VVD_ADC_BASE_VO2, VVD_ADC_RESULT_VO2, VVD_ADC_CH_VO2,
     ADC_SOC_NUMBER9, VVD_ADC_OFFSET_RAW_VO2,
     VVD_ADC_GAIN_VO2_V_PER_COUNT},
    {VVD_ADC_BASE_VO3, VVD_ADC_RESULT_VO3, VVD_ADC_CH_VO3,
     ADC_SOC_NUMBER10, VVD_ADC_OFFSET_RAW_VO3,
     VVD_ADC_GAIN_VO3_V_PER_COUNT},
    {VVD_ADC_BASE_VO4, VVD_ADC_RESULT_VO4, VVD_ADC_CH_VO4,
     ADC_SOC_NUMBER11, VVD_ADC_OFFSET_RAW_VO4,
     VVD_ADC_GAIN_VO4_V_PER_COUNT}
};

__interrupt void adca1ISR(void);

static float clampFloat(float value, float lower, float upper)
{
    if(value < lower) {
        return lower;
    }
    if(value > upper) {
        return upper;
    }
    return value;
}

static float rawToAdcVoltage(uint16_t raw)
{
    return ((float)raw * VVD_ADC_FULL_SCALE_V) / VVD_ADC_MAX_COUNT;
}

static float calibratedRawToEngineering(uint16_t inputIndex, uint16_t raw)
{
    float correctedCounts;

    /* Preserve the physical zero despite a negative fitted intercept. */
    if(raw == 0U) {
        return 0.0f;
    }

    correctedCounts = (float)raw - kInputs[inputIndex].offsetRaw;

    if(correctedCounts < 0.0f) {
        correctedCounts = 0.0f;
    }
    return correctedCounts * kInputs[inputIndex].gainEngineeringPerCount;
}

static uint16_t medianAdcRaw(const uint16_t samples[VVD_ADC_MEDIAN_SAMPLES])
{
    uint16_t sorted[VVD_ADC_MEDIAN_SAMPLES];
    uint16_t value;
    uint16_t i;
    uint16_t j;

    for(i = 0U; i < VVD_ADC_MEDIAN_SAMPLES; i++) {
        sorted[i] = samples[i];
    }
    for(i = 1U; i < VVD_ADC_MEDIAN_SAMPLES; i++) {
        value = sorted[i];
        j = i;
        while((j > 0U) && (sorted[j - 1U] > value)) {
            sorted[j] = sorted[j - 1U];
            j--;
        }
        sorted[j] = value;
    }
    return sorted[VVD_ADC_MEDIAN_SAMPLES / 2U];
}

static void initHilAdcPins(void)
{
    GPIO_setAnalogMode(VVD_ADC_GPIO_VO1, GPIO_ANALOG_ENABLED);
    GPIO_setAnalogMode(VVD_ADC_GPIO_VO2, GPIO_ANALOG_ENABLED);
    GPIO_setAnalogMode(VVD_ADC_GPIO_VO3, GPIO_ANALOG_ENABLED);
}

static void initAdcModule(uint32_t adcBase, bool configureInterrupt)
{
    uint16_t i;

    ADC_setVREF(adcBase, ADC_REFERENCE_INTERNAL, ADC_REFERENCE_2_5V);
    ADC_setPrescaler(adcBase, ADC_CLK_DIV_4_0);
    ADC_setInterruptPulseMode(adcBase, ADC_PULSE_END_OF_CONV);
    ADC_enableConverter(adcBase);
    DEVICE_DELAY_US(1000U);

    for(i = 0U; i < VVD_ADC_INPUT_COUNT; i++) {
        if(kInputs[i].adcBase == adcBase) {
            ADC_setupSOC(adcBase,
                         kInputs[i].socNumber,
                         ADC_TRIGGER_EPWM1_SOCA,
                         kInputs[i].channel,
                         VVD_ADC_ACQ_WINDOW);
        }
    }

    if(configureInterrupt) {
        /* ADCA SOC11 completes after all ADCA channels; ADCB samples Vo2. */
        ADC_setInterruptSource(adcBase, ADC_INT_NUMBER1, ADC_SOC_NUMBER11);
        ADC_enableInterrupt(adcBase, ADC_INT_NUMBER1);
        ADC_clearInterruptStatus(adcBase, ADC_INT_NUMBER1);
    }
}

static void initAdc(void)
{
    initAdcModule(ADCA_BASE, true);
    initAdcModule(ADCB_BASE, false);
    Interrupt_register(INT_ADCA1, &adca1ISR);
    Interrupt_enable(INT_ADCA1);
}

static void resetPrimaryAverageAccumulator(void)
{
    uint16_t i;

    g_primaryAverageCount = 0U;
    for(i = 0U; i < 4U; i++) {
        g_voSampleSumV[i] = 0.0f;
        g_ilVoltageLoopSampleSumA[i] = 0.0f;
        g_voAveragedV[i] = 0.0f;
        g_ilVoltageLoopAveragedA[i] = 0.0f;
    }
}

static void sampleHilOutputsFromAdc(void)
{
    uint16_t raw;
    float adcVoltage;
    float calibratedEngineering;
    float medianEngineering;
    uint16_t historySample;
    uint16_t i;

    for(i = 0U; i < VVD_ADC_INPUT_COUNT; i++) {
        raw = ADC_readResult(kInputs[i].resultBase, kInputs[i].socNumber);
        adcVoltage = rawToAdcVoltage(raw);
        calibratedEngineering = calibratedRawToEngineering(i, raw);

        g_hil.adcRaw[i] = raw;
        g_hil.adcVoltage[i] = adcVoltage;
        g_hil.adcCalibrated[i] = calibratedEngineering;

        if(i < 4U) {
            if(!g_adcFilterPrimed) {
                for(historySample = 0U;
                    historySample < VVD_ADC_MEDIAN_SAMPLES;
                    historySample++) {
                    g_adcRawHistory[i][historySample] = raw;
                }
                g_adcFilteredEngineering[i] = calibratedEngineering;
            } else {
                g_adcRawHistory[i][g_adcHistoryIndex] = raw;
                medianEngineering = calibratedRawToEngineering(
                    i, medianAdcRaw(g_adcRawHistory[i]));
                g_adcFilteredEngineering[i] +=
                    VVD_ADC_VOLTAGE_FILTER_ALPHA *
                    (medianEngineering - g_adcFilteredEngineering[i]);
            }
            g_hil.vdc[i] = g_adcFilteredEngineering[i];
        } else if(i < 8U) {
            g_hil.il[i - 4U] = calibratedEngineering;
            g_ilVoltageLoopSampleSumA[i - 4U] += calibratedEngineering;
        } else {
            g_voSampleSumV[i - 8U] += calibratedEngineering;
        }
    }

    g_primaryAverageCount++;
    if(g_primaryAverageCount >= VVD_PRIMARY_AVERAGE_SAMPLES) {
        const float averageScale =
            1.0f / (float)VVD_PRIMARY_AVERAGE_SAMPLES;
        g_primaryAverageCount = 0U;
        for(i = 0U; i < 4U; i++) {
            g_voAveragedV[i] = g_voSampleSumV[i] * averageScale;
            g_ilVoltageLoopAveragedA[i] =
                g_ilVoltageLoopSampleSumA[i] * averageScale;
            g_voSampleSumV[i] = 0.0f;
            g_ilVoltageLoopSampleSumA[i] = 0.0f;
            g_hil.vo[i] = g_voAveragedV[i];
            g_panel.converterVoltageV[i] = g_hil.vo[i];
        }
    }

    if(!g_adcFilterPrimed) {
        g_adcFilterPrimed = true;
    }
    g_adcHistoryIndex++;
    if(g_adcHistoryIndex >= VVD_ADC_MEDIAN_SAMPLES) {
        g_adcHistoryIndex = 0U;
    }
}

static void resetPrimaryPiState(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_voltagePiIntegralA[i] = g_hil.il[i];
        g_currentReferenceA[i] = g_hil.il[i];
        g_currentPiIntegralDuty[i] = clampFloat(
            g_pwmDuty[i], VVD_DUTY_MIN, VVD_DUTY_MAX);
        g_panel.currentReferenceA[i] = g_currentReferenceA[i];
        g_panel.voltageLoopErrorV[i] = 0.0f;
        g_panel.currentLoopErrorA[i] = 0.0f;
    }
}

static void resetVvdState(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_vvd.x[i] = 0.0f;
        g_vvd.h[i] = 0.0f;
        g_vvd.shareErrorIntegral[i] = 0.0f;
        g_vvd.u[i] = 0.0f;
        g_panel.averageVvdEstimateV[i] = 0.0f;
        g_panel.secondaryVoltageV[i] = 0.0f;
        g_voltageReferenceV[i] = g_panel.nominalVoltageV;
    }
    g_panel.secondarySaturated = 0U;
}

static void serviceControlPanel(void)
{
    uint16_t mode = g_panel.mode;

    if(mode > VVD_CONTROL_MODE_SECONDARY) {
        mode = VVD_CONTROL_MODE_SECONDARY;
        g_panel.mode = mode;
    }

    g_panel.nominalVoltageV = clampFloat(
        g_panel.nominalVoltageV, 46.0f, 50.0f);
    g_panel.droopBaseOhm = clampFloat(
        g_panel.droopBaseOhm,
        VVD_DROOP_BASE_MIN_OHM,
        VVD_DROOP_BASE_MAX_OHM);
    g_panel.secondaryKi = clampFloat(g_panel.secondaryKi, 0.0f, 30.0f);
    g_panel.dacAlpha = clampFloat(g_panel.dacAlpha, 0.0f, 100.0f);
    g_panel.dacBeta = clampFloat(g_panel.dacBeta, 0.0f, 200.0f);
    g_panel.voltagePiKp = clampFloat(
        g_panel.voltagePiKp, 0.0f, VVD_VOLTAGE_PI_KP_MAX);
    g_panel.voltagePiKi = clampFloat(
        g_panel.voltagePiKi, 0.0f, VVD_VOLTAGE_PI_KI_MAX);
    g_panel.currentPiKp = clampFloat(
        g_panel.currentPiKp, 0.0f, VVD_CURRENT_PI_KP_MAX);
    g_panel.currentPiKi = clampFloat(
        g_panel.currentPiKi, 0.0f, VVD_CURRENT_PI_KI_MAX);

    if((g_panel.reset != 0U) || (mode != g_lastControlMode)) {
        /* Stop the real-time controller while the shared states are reset. */
        DINT;
        g_activeControlMode = VVD_CONTROL_MODE_FIXED;
        EINT;

        resetVvdState();

        /* Preserve primary PI state for the important mode-1 -> mode-2 step. */
        if((g_panel.reset != 0U) ||
           (mode <= VVD_CONTROL_MODE_DROOP) ||
           (g_lastControlMode == VVD_CONTROL_MODE_FIXED)) {
            resetPrimaryPiState();
        }

        DINT;
        g_voltageLoopDivider = 0U;
        g_metricsPending = 0U;
        resetPrimaryAverageAccumulator();
        g_panel.passHoldCount = 0U;
        g_panel.qualifiedPass = 0U;
        g_panel.secondarySaturated = 0U;
        g_panel.dutySaturated = 0U;
        g_lastControlMode = mode;
        g_activeControlMode = mode;
        g_panel.reset = 0U;
        EINT;
    }
}

static void updateMetrics(void)
{
    float voltageSum = 0.0f;
    float maxDeviation = 0.0f;
    float minVoltage = g_hil.vdc[0];
    float maxVoltage = g_hil.vdc[0];
    float currentSum = 0.0f;
    float capacitySum = 0.0f;
    float perCapacityErrorSum = 0.0f;
    float deviation;
    float denominator;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        voltageSum += g_hil.vdc[i];
        deviation = fabsf(g_hil.vdc[i] - g_panel.nominalVoltageV);
        if(deviation > maxDeviation) {
            maxDeviation = deviation;
        }
        if(g_hil.vdc[i] < minVoltage) {
            minVoltage = g_hil.vdc[i];
        }
        if(g_hil.vdc[i] > maxVoltage) {
            maxVoltage = g_hil.vdc[i];
        }
        currentSum += g_ilVoltageLoopAveragedA[i];
        capacitySum += kCapacityRatio[i];
        g_metrics.currentPerCapacityA[i] =
            g_ilVoltageLoopAveragedA[i] / kCapacityRatio[i];
    }

    g_metrics.averageVoltageV = 0.25f * voltageSum;
    g_metrics.averageVoltageErrorPu =
        fabsf(g_metrics.averageVoltageV - g_panel.nominalVoltageV) /
        g_panel.nominalVoltageV;
    g_metrics.maxVoltageDeviationV = maxDeviation;
    g_metrics.minBusVoltageV = minVoltage;
    g_metrics.maxBusVoltageV = maxVoltage;
    g_metrics.shareLambdaA = currentSum / capacitySum;

    for(i = 0U; i < 4U; i++) {
        g_metrics.targetCurrentA[i] =
            g_metrics.shareLambdaA * kCapacityRatio[i];
        perCapacityErrorSum += fabsf(
            g_metrics.currentPerCapacityA[i] - g_metrics.shareLambdaA);
    }
    denominator = fmaxf(
        fabsf(g_metrics.shareLambdaA), 0.1f * VVD_CURRENT_BASE_A);
    g_metrics.currentErrorPu =
        (0.25f * perCapacityErrorSum) / denominator;
    g_metrics.voltagePass =
        (maxDeviation <=
         g_panel.nominalVoltageV * VVD_VOLTAGE_SAFE_FRACTION) ? 1U : 0U;
    g_metrics.averageVoltagePass =
        (g_metrics.averageVoltageErrorPu <=
         VVD_AVG_VOLTAGE_ERROR_TARGET) ? 1U : 0U;
    g_metrics.currentPass =
        (g_metrics.currentErrorPu <= VVD_CURRENT_ERROR_TARGET) ? 1U : 0U;
    g_metrics.overallPass =
        ((g_metrics.voltagePass != 0U) &&
         (g_metrics.averageVoltagePass != 0U) &&
         (g_metrics.currentPass != 0U)) ? 1U : 0U;

    g_panel.averageVoltageV = g_metrics.averageVoltageV;
    g_panel.maxVoltageDeviationV = g_metrics.maxVoltageDeviationV;
    g_panel.minBusVoltageV = g_metrics.minBusVoltageV;
    g_panel.maxBusVoltageV = g_metrics.maxBusVoltageV;
    g_panel.currentErrorPu = g_metrics.currentErrorPu;
    g_panel.shareLambdaA = g_metrics.shareLambdaA;
    g_panel.voltagePass = g_metrics.voltagePass;
    g_panel.averageVoltagePass = g_metrics.averageVoltagePass;
    g_panel.currentPass = g_metrics.currentPass;
    g_panel.overallPass = g_metrics.overallPass;

    if(g_panel.overallPass != 0U) {
        if(g_panel.passHoldCount < VVD_PASS_HOLD_SAMPLES) {
            g_panel.passHoldCount++;
        }
    } else {
        g_panel.passHoldCount = 0U;
    }
    g_panel.qualifiedPass =
        (g_panel.passHoldCount >= VVD_PASS_HOLD_SAMPLES) ? 1U : 0U;

    for(i = 0U; i < 4U; i++) {
        g_panel.currentPerCapacityA[i] =
            g_metrics.currentPerCapacityA[i];
        g_panel.targetCurrentA[i] = g_metrics.targetCurrentA[i];
    }
}

static void updateDroopAndVirtualDrop(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_panel.droopGainOhm[i] =
            g_panel.droopBaseOhm / kCapacityRatio[i];
        g_panel.localResistanceOhm[i] = kLocalResistanceOhm[i];
        g_panel.converterVoltageV[i] = g_hil.vo[i];
        g_panel.virtualVoltageDropV[i] =
            (g_panel.droopGainOhm[i] + kLocalResistanceOhm[i]) *
            g_ilVoltageLoopAveragedA[i];
    }
}

static void clearSecondaryVoltage(void)
{
    uint16_t i;

    g_panel.secondarySaturated = 0U;
    for(i = 0U; i < 4U; i++) {
        g_vvd.h[i] = 0.0f;
        g_vvd.u[i] = 0.0f;
        g_panel.averageVvdEstimateV[i] = 0.0f;
        g_panel.secondaryVoltageV[i] = 0.0f;
    }
}

static void updateDistributedVvdSecondary(void)
{
    float oldH[4];
    float nextX[4];
    float consensus;
    float proposedIntegral;
    float unsaturatedSecondary;
    uint16_t previous;
    uint16_t next;
    uint16_t i;

    g_panel.secondarySaturated = 0U;

    /* Weight-balanced, strongly connected four-node ring. */
    for(i = 0U; i < 4U; i++) {
        oldH[i] = g_vvd.x[i] + g_panel.virtualVoltageDropV[i];
    }
    for(i = 0U; i < 4U; i++) {
        previous = (i == 0U) ? 3U : (uint16_t)(i - 1U);
        next = (i == 3U) ? 0U : (uint16_t)(i + 1U);
        consensus = (2.0f * oldH[i]) - oldH[previous] - oldH[next];
        nextX[i] = g_vvd.x[i] + VVD_SECONDARY_LOOP_TS_SEC *
            ((-g_panel.dacAlpha * g_vvd.x[i]) -
             (g_panel.dacBeta * consensus));
        nextX[i] = clampFloat(
            nextX[i], -VVD_X_STATE_LIMIT_V, VVD_X_STATE_LIMIT_V);
    }

    for(i = 0U; i < 4U; i++) {
        g_vvd.x[i] = nextX[i];
        g_vvd.h[i] = g_vvd.x[i] + g_panel.virtualVoltageDropV[i];
        proposedIntegral = g_vvd.shareErrorIntegral[i] +
            (g_panel.virtualVoltageDropV[i] - g_vvd.h[i]) *
            VVD_SECONDARY_LOOP_TS_SEC;
        g_vvd.shareErrorIntegral[i] = clampFloat(
            proposedIntegral,
            -VVD_ERROR_INTEGRAL_LIMIT_VS,
            VVD_ERROR_INTEGRAL_LIMIT_VS);

        unsaturatedSecondary = g_vvd.h[i] -
            g_panel.secondaryKi * g_vvd.shareErrorIntegral[i];
        g_vvd.u[i] = clampFloat(
            unsaturatedSecondary,
            VVD_SECONDARY_MIN_V,
            VVD_SECONDARY_MAX_V);
        if(fabsf(unsaturatedSecondary - g_vvd.u[i]) > 0.0001f) {
            g_panel.secondarySaturated = 1U;
        }

        g_panel.averageVvdEstimateV[i] = g_vvd.h[i];
        g_panel.secondaryVoltageV[i] = g_vvd.u[i];
    }
}

static void composeVoltageReference(void)
{
    float secondary;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        secondary = (g_activeControlMode == VVD_CONTROL_MODE_SECONDARY) ?
            g_vvd.u[i] : 0.0f;
        g_voltageReferenceV[i] = g_panel.nominalVoltageV -
            g_panel.droopGainOhm[i] * g_ilVoltageLoopAveragedA[i] +
            secondary;
        g_panel.voltageReferenceOutV[i] = g_voltageReferenceV[i];
    }
}

static void updateVoltageOuterLoop(void)
{
    float error;
    float referenceA;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        error = g_voltageReferenceV[i] - g_hil.vo[i];
        g_voltagePiIntegralA[i] +=
            g_panel.voltagePiKi * error * VVD_VOLTAGE_LOOP_TS_SEC;
        referenceA =
            g_panel.voltagePiKp * error + g_voltagePiIntegralA[i];
        g_currentReferenceA[i] = referenceA;
        g_panel.voltageLoopErrorV[i] = error;
        g_panel.currentReferenceA[i] = referenceA;
    }
}

static void updateCurrentInnerLoop(void)
{
    float error;
    float proposedIntegralDuty;
    float unsaturatedDuty;
    float duty;
    uint16_t i;

    g_panel.dutySaturated = 0U;
    for(i = 0U; i < 4U; i++) {
        error = g_currentReferenceA[i] - g_hil.il[i];
        proposedIntegralDuty = g_currentPiIntegralDuty[i] +
            g_panel.currentPiKi * error * VVD_CURRENT_LOOP_TS_SEC;
        proposedIntegralDuty = clampFloat(
            proposedIntegralDuty, VVD_DUTY_MIN, VVD_DUTY_MAX);
        unsaturatedDuty =
            g_panel.currentPiKp * error + proposedIntegralDuty;
        duty = clampFloat(unsaturatedDuty, VVD_DUTY_MIN, VVD_DUTY_MAX);

        if((unsaturatedDuty == duty) ||
           ((duty <= VVD_DUTY_MIN) && (error > 0.0f)) ||
           ((duty >= VVD_DUTY_MAX) && (error < 0.0f))) {
            g_currentPiIntegralDuty[i] = proposedIntegralDuty;
        }
        if(fabsf(unsaturatedDuty - duty) > 0.0001f) {
            g_panel.dutySaturated = 1U;
        }

        g_panel.currentLoopErrorA[i] = error;
        g_pwmDuty[i] = duty;
    }
}

static void updateFixedDuty(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_pwmDuty[i] = VVD_FIXED_DUTY;
    }
}

static void initGpioForPwm(void)
{
    GPIO_setPinConfig(VVD_PWM1_GPIO_MUX);
    GPIO_setPinConfig(VVD_PWM2_GPIO_MUX);
    GPIO_setPinConfig(VVD_PWM3_GPIO_MUX);
    GPIO_setPinConfig(VVD_PWM4_GPIO_MUX);
    GPIO_setPadConfig(VVD_PWM1_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(VVD_PWM2_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(VVD_PWM3_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(VVD_PWM4_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setQualificationMode(VVD_PWM1_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(VVD_PWM2_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(VVD_PWM3_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(VVD_PWM4_GPIO, GPIO_QUAL_SYNC);
}

static uint16_t pwmPeriodTicks(void)
{
    return (uint16_t)(VVD_EPWM_TBCLK_HZ / (2UL * VVD_PWM_FREQ_HZ));
}

static uint16_t dutyToCmp(float duty, uint16_t period)
{
    duty = clampFloat(duty, 0.0f, 1.0f);
    return (uint16_t)((float)period * duty);
}

static void configureOnePwm(uint32_t epwmBase, float duty)
{
    uint16_t period = pwmPeriodTicks();

    EPWM_setTimeBasePeriod(epwmBase, period);
    EPWM_setPhaseShift(epwmBase, 0U);
    EPWM_setTimeBaseCounter(epwmBase, 0U);
    EPWM_setTimeBaseCounterMode(epwmBase, EPWM_COUNTER_MODE_UP_DOWN);
    EPWM_setClockPrescaler(epwmBase,
                           EPWM_CLOCK_DIVIDER_1,
                           EPWM_HSCLOCK_DIVIDER_1);
    EPWM_setCounterCompareValue(epwmBase,
                                EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(duty, period));
    EPWM_setCounterCompareShadowLoadMode(epwmBase,
                                         EPWM_COUNTER_COMPARE_A,
                                         EPWM_COMP_LOAD_ON_CNTR_ZERO);
    EPWM_setActionQualifierAction(epwmBase,
                                  EPWM_AQ_OUTPUT_A,
                                  EPWM_AQ_OUTPUT_HIGH,
                                  EPWM_AQ_OUTPUT_ON_TIMEBASE_DOWN_CMPA);
    EPWM_setActionQualifierAction(epwmBase,
                                  EPWM_AQ_OUTPUT_A,
                                  EPWM_AQ_OUTPUT_LOW,
                                  EPWM_AQ_OUTPUT_ON_TIMEBASE_UP_CMPA);
}

static void initPwm(void)
{
    SysCtl_disablePeripheral(SYSCTL_PERIPH_CLK_TBCLKSYNC);
    initGpioForPwm();
    configureOnePwm(VVD_PWM1_BASE, g_pwmDuty[0]);
    configureOnePwm(VVD_PWM2_BASE, g_pwmDuty[1]);
    configureOnePwm(VVD_PWM3_BASE, g_pwmDuty[2]);
    configureOnePwm(VVD_PWM4_BASE, g_pwmDuty[3]);

    EPWM_setADCTriggerSource(
        VVD_PWM1_BASE, EPWM_SOC_A, EPWM_SOC_TBCTR_ZERO);
    EPWM_setADCTriggerEventPrescale(VVD_PWM1_BASE, EPWM_SOC_A, 1U);
    EPWM_enableADCTrigger(VVD_PWM1_BASE, EPWM_SOC_A);
    SysCtl_enablePeripheral(SYSCTL_PERIPH_CLK_TBCLKSYNC);
}

static void updatePwmDuty(void)
{
    uint16_t period = pwmPeriodTicks();

    EPWM_setCounterCompareValue(VVD_PWM1_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[0], period));
    EPWM_setCounterCompareValue(VVD_PWM2_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[1], period));
    EPWM_setCounterCompareValue(VVD_PWM3_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[2], period));
    EPWM_setCounterCompareValue(VVD_PWM4_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[3], period));
}

__interrupt void adca1ISR(void)
{
    sampleHilOutputsFromAdc();
    g_panel.currentLoopCount++;

    g_voltageLoopDivider++;
    if(g_voltageLoopDivider >= VVD_VOLTAGE_LOOP_DIVIDER) {
        g_voltageLoopDivider = 0U;
        g_panel.voltageLoopCount++;
        /* Keep reporting outside the hard real-time ISR, as in MAPPO. */
        g_metricsPending = 1U;

        if(g_activeControlMode >= VVD_CONTROL_MODE_DROOP) {
            updateDroopAndVirtualDrop();
            if(g_activeControlMode == VVD_CONTROL_MODE_SECONDARY) {
                updateDistributedVvdSecondary();
            } else {
                clearSecondaryVoltage();
            }
            composeVoltageReference();
            updateVoltageOuterLoop();
        }
    }

    if(g_activeControlMode >= VVD_CONTROL_MODE_DROOP) {
        updateCurrentInnerLoop();
    } else {
        updateFixedDuty();
    }
    updatePwmDuty();

    ADC_clearInterruptStatus(ADCA_BASE, ADC_INT_NUMBER1);
    if(ADC_getInterruptOverflowStatus(ADCA_BASE, ADC_INT_NUMBER1)) {
        g_panel.adcOverflowCount++;
        ADC_clearInterruptOverflowStatus(ADCA_BASE, ADC_INT_NUMBER1);
        ADC_clearInterruptStatus(ADCA_BASE, ADC_INT_NUMBER1);
    }
    Interrupt_clearACKGroup(INTERRUPT_ACK_GROUP1);
}

void main(void)
{
    uint16_t updateReporting;

    Device_init();
    Device_initGPIO();
    Interrupt_initModule();
    Interrupt_initVectorTable();

    initHilAdcPins();
    initAdc();
    initPwm();

    EINT;
    ERTM;

    for(;;) {
        serviceControlPanel();

        updateReporting = 0U;
        DINT;
        if(g_metricsPending != 0U) {
            g_metricsPending = 0U;
            updateReporting = 1U;
        }
        EINT;
        if(updateReporting != 0U) {
            updateMetrics();
        }
    }
}
