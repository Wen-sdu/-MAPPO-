#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "driverlib.h"
#include "device.h"
#include "mappo_double_loop_config.h"
#include "mappo_actor_weights.h"

#if !defined(MAPPO_USE_GRU) || (MAPPO_USE_GRU != 0U)
#error "This application requires a feed-forward MAPPO actor without GRU parameters."
#endif

#if MAPPO_TRAIN_CAPACITY_RATIO_COUNT != 4U
#error "This F28P55 application requires four actor capacity ratios."
#endif

#if MAPPO_OBS_DIM != 6U
#error "This F28P55 application requires the six-element MAPPO observation."
#endif

typedef struct {
    uint32_t adcBase;
    uint32_t resultBase;
    ADC_Channel channel;
    ADC_SOCNumber socNumber;
    float offsetRaw;
    float gainEngineeringPerCount;
} HilAdcInput;

typedef struct {
    float vdc[4];
    float il[4];
    float vo[4];
    float adcVoltage[HIL_ADC_INPUT_COUNT];
    uint16_t adcRaw[HIL_ADC_INPUT_COUNT];
    float adcCalibrated[HIL_ADC_INPUT_COUNT];
} HilMeasurements;

typedef struct {
    uint32_t firmwareBuildId;
    uint16_t mode;
    uint16_t reset;
    float voltagePiKp;
    float voltagePiKi;
    float currentPiKp;
    float currentPiKi;
    float converterVoltageV[4];
    float secondaryVoltageV[4];
    float voltageReferenceOutV[4];
    float currentReferenceA[4];
    float voltageLoopErrorV[4];
    float currentLoopErrorA[4];
} HilControlPanel;

volatile HilMeasurements g_hil;
volatile HilControlPanel g_panel = {
    2026101502UL,
    HIL_CONTROL_MODE_FIXED,
    0U,
    HIL_VOLTAGE_PI_KP_A_PER_V,
    HIL_VOLTAGE_PI_KI_A_PER_VS,
    HIL_CURRENT_PI_KP_DUTY_PER_A,
    HIL_CURRENT_PI_KI_DUTY_PER_AS
};
volatile float g_pwmDuty[4] = {
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY
};
/* Internal states remain visible for deep debugging; normal CCS use needs g_panel only. */
volatile float g_voltagePiIntegralA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_voltageLoopErrorV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_currentReferenceA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_currentPiIntegralDuty[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_currentLoopErrorA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_currentPiUnsatDuty[4] = {
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY,
    HIL_FIXED_DUTY
};
volatile float g_mappoVref[4] = {
    HIL_PI_VREF,
    HIL_PI_VREF,
    HIL_PI_VREF,
    HIL_PI_VREF
};
volatile float g_mappoRawAction[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_mappoDifferentialActionV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_mappoDifferentialFilteredV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_mappoActorMeanV = 0.0f;
volatile float g_commonVoltageErrorV = 0.0f;
volatile float g_commonVoltageIntegralV = 0.0f;
volatile float g_commonVoltageActionV = 0.0f;
volatile float g_commonVoltageFilteredV = 0.0f;
volatile float g_commonFeedforwardTargetV = 0.0f;
volatile float g_commonFeedforwardFilteredV = 0.0f;
volatile float g_secondaryVoltageV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_voltageCommandV[4] = {
    HIL_PI_VREF, HIL_PI_VREF, HIL_PI_VREF, HIL_PI_VREF
};
static uint16_t g_lastControlMode = HIL_CONTROL_MODE_FIXED;
volatile uint16_t g_activeControlMode = HIL_CONTROL_MODE_FIXED;
static volatile uint16_t g_voltageLoopDivider = 0U;
static volatile uint16_t g_secondaryLoopDivider = 0U;
static volatile uint16_t g_secondaryLoopPending = 0U;
volatile uint32_t g_currentLoopCount = 0UL;
volatile uint32_t g_secondaryDeadlineMissCount = 0UL;

/*
 * At each 100 Hz boundary the ISR captures one coherent, current-state
 * observation for the actor.  IL comes from the latest completed 1 ms average
 * and Vdc from the current conditioned sample.  This matches the training
 * model's point observation without adding a 10 ms moving-average delay.
 */
static volatile float g_actorPendingCurrentA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static volatile float g_actorPendingVdcV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_mappoSnapshotCurrentA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_mappoSnapshotVdcV[4] = {0.0f, 0.0f, 0.0f, 0.0f};

/*
 * ADC conditioning state.  Keep g_hil.adcRaw[] and g_hil.adcVoltage[] as
 * unfiltered diagnostic values and expose each instantaneous calibrated
 * engineering value through g_hil.adcCalibrated[].  Vdc keeps the rolling
 * median plus IIR, IL stays unfiltered for the 10 kHz inner loop, and matching
 * IL/Vo windows provide one non-overlapping 10-sample/1 ms average to the
 * droop equation and voltage PI.
 */
static uint16_t g_adcRawHistory[HIL_ADC_INPUT_COUNT][HIL_ADC_MEDIAN_SAMPLES];
static float g_adcFilteredEngineering[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint16_t g_adcHistoryIndex = 0U;
static bool g_adcFilterPrimed = false;
static float g_voSampleSumV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static float g_ilVoltageLoopSampleSumA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint16_t g_voAverageCount = 0U;
volatile float g_voAveragedV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
volatile float g_ilVoltageLoopAveragedA[4] = {0.0f, 0.0f, 0.0f, 0.0f};

__interrupt void adca1ISR(void);

static const HilAdcInput kInputs[HIL_ADC_INPUT_COUNT] = {
    {HIL_ADC_BASE_VDC1, HIL_ADC_RESULT_VDC1, HIL_ADC_CH_VDC1, ADC_SOC_NUMBER0, HIL_ADC_OFFSET_RAW_VDC1, HIL_ADC_GAIN_VDC1_V_PER_COUNT},
    {HIL_ADC_BASE_VDC2, HIL_ADC_RESULT_VDC2, HIL_ADC_CH_VDC2, ADC_SOC_NUMBER1, HIL_ADC_OFFSET_RAW_VDC2, HIL_ADC_GAIN_VDC2_V_PER_COUNT},
    {HIL_ADC_BASE_VDC3, HIL_ADC_RESULT_VDC3, HIL_ADC_CH_VDC3, ADC_SOC_NUMBER2, HIL_ADC_OFFSET_RAW_VDC3, HIL_ADC_GAIN_VDC3_V_PER_COUNT},
    {HIL_ADC_BASE_VDC4, HIL_ADC_RESULT_VDC4, HIL_ADC_CH_VDC4, ADC_SOC_NUMBER3, HIL_ADC_OFFSET_RAW_VDC4, HIL_ADC_GAIN_VDC4_V_PER_COUNT},
    {HIL_ADC_BASE_IL1,  HIL_ADC_RESULT_IL1,  HIL_ADC_CH_IL1,  ADC_SOC_NUMBER4, HIL_ADC_OFFSET_RAW_IL1, HIL_ADC_GAIN_IL1_A_PER_COUNT},
    {HIL_ADC_BASE_IL2,  HIL_ADC_RESULT_IL2,  HIL_ADC_CH_IL2,  ADC_SOC_NUMBER5, HIL_ADC_OFFSET_RAW_IL2, HIL_ADC_GAIN_IL2_A_PER_COUNT},
    {HIL_ADC_BASE_IL3,  HIL_ADC_RESULT_IL3,  HIL_ADC_CH_IL3,  ADC_SOC_NUMBER6, HIL_ADC_OFFSET_RAW_IL3, HIL_ADC_GAIN_IL3_A_PER_COUNT},
    {HIL_ADC_BASE_IL4,  HIL_ADC_RESULT_IL4,  HIL_ADC_CH_IL4,  ADC_SOC_NUMBER7, HIL_ADC_OFFSET_RAW_IL4, HIL_ADC_GAIN_IL4_A_PER_COUNT},
    {HIL_ADC_BASE_VO1,  HIL_ADC_RESULT_VO1,  HIL_ADC_CH_VO1,  ADC_SOC_NUMBER8, HIL_ADC_OFFSET_RAW_VO1, HIL_ADC_GAIN_VO1_V_PER_COUNT},
    {HIL_ADC_BASE_VO2,  HIL_ADC_RESULT_VO2,  HIL_ADC_CH_VO2,  ADC_SOC_NUMBER9, HIL_ADC_OFFSET_RAW_VO2, HIL_ADC_GAIN_VO2_V_PER_COUNT},
    {HIL_ADC_BASE_VO3,  HIL_ADC_RESULT_VO3,  HIL_ADC_CH_VO3,  ADC_SOC_NUMBER10, HIL_ADC_OFFSET_RAW_VO3, HIL_ADC_GAIN_VO3_V_PER_COUNT},
    {HIL_ADC_BASE_VO4,  HIL_ADC_RESULT_VO4,  HIL_ADC_CH_VO4,  ADC_SOC_NUMBER11, HIL_ADC_OFFSET_RAW_VO4, HIL_ADC_GAIN_VO4_V_PER_COUNT},
};

static void initHilAdcPins(void)
{
    /*
     * These LaunchPad header inputs share their package pins with GPIOs.
     * Select the analog path before configuring the ADC SOCs; otherwise the
     * header voltage can be correct while the ADC sees a disconnected input.
     * Vo4/ADCINA12 is not GPIO-multiplexed and needs no matching call.
     */
    GPIO_setAnalogMode(HIL_ADC_GPIO_VO1, GPIO_ANALOG_ENABLED);
    GPIO_setAnalogMode(HIL_ADC_GPIO_VO2, GPIO_ANALOG_ENABLED);
    GPIO_setAnalogMode(HIL_ADC_GPIO_VO3, GPIO_ANALOG_ENABLED);
}

static void initAdcModule(uint32_t adcBase, bool configureInterrupt)
{
    uint16_t i;

    ADC_setVREF(adcBase, ADC_REFERENCE_INTERNAL, ADC_REFERENCE_2_5V);
    ADC_setPrescaler(adcBase, ADC_CLK_DIV_4_0);
    ADC_setInterruptPulseMode(adcBase, ADC_PULSE_END_OF_CONV);
    ADC_enableConverter(adcBase);
    DEVICE_DELAY_US(1000);

    for(i = 0U; i < HIL_ADC_INPUT_COUNT; i++) {
        if(kInputs[i].adcBase == adcBase) {
            ADC_setupSOC(adcBase,
                         kInputs[i].socNumber,
                         ADC_TRIGGER_EPWM1_SOCA,
                         kInputs[i].channel,
                         HIL_ADC_ACQ_WINDOW);
        }
    }

    if(configureInterrupt) {
        ADC_setInterruptSource(adcBase, ADC_INT_NUMBER1, ADC_SOC_NUMBER11);
        ADC_enableInterrupt(adcBase, ADC_INT_NUMBER1);
        ADC_clearInterruptStatus(adcBase, ADC_INT_NUMBER1);
    }
}

static void initAdc(void)
{
    /* ADCA SOC11 is the end-of-scan interrupt. ADCB only samples Vo2. */
    initAdcModule(ADCA_BASE, true);
    initAdcModule(ADCB_BASE, false);

    Interrupt_register(INT_ADCA1, &adca1ISR);
    Interrupt_enable(INT_ADCA1);
}

static float rawToAdcVoltage(uint16_t raw)
{
    return ((float)raw * HIL_ADC_FULL_SCALE_V) / HIL_ADC_MAX_COUNT;
}

static float calibratedRawToEngineering(uint16_t inputIndex, uint16_t raw)
{
    float correctedCounts;

    /*
     * A true zero input is clipped to raw zero by the unipolar ADC.  Some
     * channels have a small negative fitted raw intercept at normal operating
     * levels, so preserve the measured zero explicitly before applying it.
     */
    if(raw == 0U) {
        return 0.0f;
    }

    correctedCounts = (float)raw - kInputs[inputIndex].offsetRaw;

    /* All twelve deployed signals are non-negative DC quantities. */
    if(correctedCounts < 0.0f) {
        correctedCounts = 0.0f;
    }
    return correctedCounts * kInputs[inputIndex].gainEngineeringPerCount;
}

static void resetPrimaryLoopAverageAccumulator(void)
{
    uint16_t i;

    g_voAverageCount = 0U;
    for(i = 0U; i < 4U; i++) {
        g_voSampleSumV[i] = 0.0f;
        g_ilVoltageLoopSampleSumA[i] = 0.0f;
        g_ilVoltageLoopAveragedA[i] = 0.0f;
    }
}

static uint16_t medianAdcRaw(const uint16_t samples[HIL_ADC_MEDIAN_SAMPLES])
{
    uint16_t sorted[HIL_ADC_MEDIAN_SAMPLES];
    uint16_t value;
    uint16_t i;
    uint16_t j;

    for(i = 0U; i < HIL_ADC_MEDIAN_SAMPLES; i++) {
        sorted[i] = samples[i];
    }

    for(i = 1U; i < HIL_ADC_MEDIAN_SAMPLES; i++) {
        value = sorted[i];
        j = i;
        while((j > 0U) && (sorted[j - 1U] > value)) {
            sorted[j] = sorted[j - 1U];
            j--;
        }
        sorted[j] = value;
    }

    return sorted[HIL_ADC_MEDIAN_SAMPLES / 2U];
}

static float clampFloat(float value, float minValue, float maxValue)
{
    if(value < minValue) {
        return minValue;
    }
    if(value > maxValue) {
        return maxValue;
    }
    return value;
}

static float reluFloat(float value)
{
    return (value > 0.0f) ? value : 0.0f;
}

static void layerNorm(const float *input,
                      const float *weight,
                      const float *bias,
                      uint16_t length,
                      float *output)
{
    float mean = 0.0f;
    float variance = 0.0f;
    float centered;
    float invStd;
    uint16_t i;

    for(i = 0U; i < length; i++) {
        mean += input[i];
    }
    mean /= (float)length;

    for(i = 0U; i < length; i++) {
        centered = input[i] - mean;
        variance += centered * centered;
    }
    variance /= (float)length;
    invStd = 1.0f / sqrtf(variance + HIL_MAPPO_LAYER_NORM_EPS);

    for(i = 0U; i < length; i++) {
        output[i] = ((input[i] - mean) * invStd * weight[i]) + bias[i];
    }
}

static void linearLayer(const float *input,
                        const float *weight,
                        const float *bias,
                        uint16_t outDim,
                        uint16_t inDim,
                        float *output)
{
    float acc;
    uint16_t row;
    uint16_t col;

    for(row = 0U; row < outDim; row++) {
        acc = bias[row];
        for(col = 0U; col < inDim; col++) {
            acc += weight[(row * inDim) + col] * input[col];
        }
        output[row] = acc;
    }
}

static void reluVector(float *values, uint16_t length)
{
    uint16_t i;

    for(i = 0U; i < length; i++) {
        values[i] = reluFloat(values[i]);
    }
}

static uint16_t firstRingNeighbor(uint16_t agent)
{
    if(agent == 0U) {
        return 3U;
    }
    return agent - 1U;
}

static uint16_t secondRingNeighbor(uint16_t agent)
{
    if(agent == 3U) {
        return 0U;
    }
    return agent + 1U;
}

static float normalizeCurrent(float current)
{
    return current / (2.0f * HIL_CURRENT_BASE);
}

static float normalizeVoltage(float voltage)
{
    return (voltage - HIL_PI_VREF) / HIL_MAPPO_OBS_VOLTAGE_DELTA_V;
}

static void buildMappoObservation(
    uint16_t agent,
    const volatile float actorCurrentA[4],
    const volatile float actorVdcV[4],
    float obs[MAPPO_OBS_DIM])
{
    uint16_t n1 = firstRingNeighbor(agent);
    uint16_t n2 = secondRingNeighbor(agent);

    obs[0] = normalizeCurrent(
        actorCurrentA[agent] / MAPPO_TRAIN_CAPACITY_RATIOS[agent]);
    obs[1] = normalizeCurrent(
        actorCurrentA[n1] / MAPPO_TRAIN_CAPACITY_RATIOS[n1]);
    obs[2] = normalizeCurrent(
        actorCurrentA[n2] / MAPPO_TRAIN_CAPACITY_RATIOS[n2]);
    obs[3] = normalizeVoltage(actorVdcV[agent]);
    obs[4] = normalizeVoltage(actorVdcV[n1]);
    obs[5] = normalizeVoltage(actorVdcV[n2]);
}

static void resetMappoActorState(void)
{
    float initialCommonV = 0.0f;
    float initialFeedforwardV;
    uint16_t i;

    /* Bumpless mode-2 entry: cancel the existing primary-droop drop instead
     * of rebuilding the common secondary correction from zero. */
    for(i = 0U; i < 4U; i++) {
        initialCommonV +=
            (HIL_DROOP_BASE_V_PER_A / MAPPO_TRAIN_CAPACITY_RATIOS[i]) *
            g_hil.il[i];
    }
    initialCommonV = clampFloat(
        initialCommonV * 0.25f,
        HIL_COMMON_VOLTAGE_MIN_V,
        HIL_COMMON_VOLTAGE_MAX_V);
    initialFeedforwardV = clampFloat(
        HIL_COMMON_CURRENT_FEEDFORWARD_GAIN * initialCommonV,
        0.0f,
        HIL_COMMON_VOLTAGE_MAX_V);

    for(i = 0U; i < 4U; i++) {
        g_mappoRawAction[i] = 0.0f;
        g_mappoDifferentialActionV[i] = 0.0f;
        g_mappoDifferentialFilteredV[i] = 0.0f;
        g_secondaryVoltageV[i] = initialCommonV;
        g_voltageCommandV[i] = HIL_PI_VREF;
        g_mappoVref[i] = HIL_PI_VREF;
        g_panel.secondaryVoltageV[i] = initialCommonV;
        g_panel.voltageReferenceOutV[i] = HIL_PI_VREF;
    }
    g_mappoActorMeanV = 0.0f;
    g_commonVoltageErrorV = 0.0f;
    g_commonFeedforwardTargetV = initialFeedforwardV;
    g_commonFeedforwardFilteredV = initialFeedforwardV;
    g_commonVoltageIntegralV = clampFloat(
        initialCommonV - initialFeedforwardV,
        HIL_COMMON_VOLTAGE_MIN_V,
        HIL_COMMON_VOLTAGE_MAX_V);
    g_commonVoltageActionV = initialCommonV;
    g_commonVoltageFilteredV = initialCommonV;
}

static void resetMappoObservationAccumulator(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_actorPendingCurrentA[i] = 0.0f;
        g_actorPendingVdcV[i] = 0.0f;
        g_mappoSnapshotCurrentA[i] = 0.0f;
        g_mappoSnapshotVdcV[i] = 0.0f;
    }
}

static float runMappoActor(
    uint16_t agent,
    const volatile float actorCurrentA[4],
    const volatile float actorVdcV[4])
{
    float obs[MAPPO_OBS_DIM];
    float normObs[MAPPO_OBS_DIM];
    float hidden1[MAPPO_HIDDEN_DIM];
    float hidden2[MAPPO_HIDDEN_DIM];
    float rawAction;
    uint16_t i;

    buildMappoObservation(agent, actorCurrentA, actorVdcV, obs);
    layerNorm(obs,
              MAPPO_FEATURE_NORM_WEIGHT,
              MAPPO_FEATURE_NORM_BIAS,
              MAPPO_OBS_DIM,
              normObs);

    linearLayer(normObs,
                MAPPO_FC1_WEIGHT,
                MAPPO_FC1_BIAS,
                MAPPO_HIDDEN_DIM,
                MAPPO_OBS_DIM,
                hidden1);
    reluVector(hidden1, MAPPO_HIDDEN_DIM);
    layerNorm(hidden1,
              MAPPO_FC1_NORM_WEIGHT,
              MAPPO_FC1_NORM_BIAS,
              MAPPO_HIDDEN_DIM,
              hidden1);

    linearLayer(hidden1,
                MAPPO_FC2_WEIGHT,
                MAPPO_FC2_BIAS,
                MAPPO_HIDDEN_DIM,
                MAPPO_HIDDEN_DIM,
                hidden2);
    reluVector(hidden2, MAPPO_HIDDEN_DIM);
    layerNorm(hidden2,
              MAPPO_FC2_NORM_WEIGHT,
              MAPPO_FC2_NORM_BIAS,
              MAPPO_HIDDEN_DIM,
              hidden2);

    rawAction = MAPPO_ACTION_BIAS[0];
    for(i = 0U; i < MAPPO_HIDDEN_DIM; i++) {
        rawAction += MAPPO_ACTION_WEIGHT[i] * hidden2[i];
    }
    return rawAction;
}

static void updateMappoSecondaryVoltage(
    const volatile float actorCurrentA[4],
    const volatile float actorVdcV[4])
{
    float nextRawAction[4];
    float nextDifferentialActionV[4];
    float nextDifferentialFilteredV[4];
    float rawActionSum = 0.0f;
    float actorMeanV;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        nextRawAction[i] = runMappoActor(i, actorCurrentA, actorVdcV);
        rawActionSum += nextRawAction[i];
    }
    actorMeanV = rawActionSum * 0.25f;

    for(i = 0U; i < 4U; i++) {
        nextDifferentialActionV[i] = nextRawAction[i] - actorMeanV;
        nextDifferentialFilteredV[i] = g_mappoDifferentialFilteredV[i] +
            HIL_MAPPO_ACTION_FILTER_ALPHA *
            (nextDifferentialActionV[i] - g_mappoDifferentialFilteredV[i]);
    }

    /* Publish one coherent four-agent differential vector to the 1 kHz ISR. */
    DINT;
    g_mappoActorMeanV = actorMeanV;
    for(i = 0U; i < 4U; i++) {
        g_mappoRawAction[i] = nextRawAction[i];
        g_mappoDifferentialActionV[i] = nextDifferentialActionV[i];
        g_mappoDifferentialFilteredV[i] = nextDifferentialFilteredV[i];
    }
    EINT;
}

static void updateFastCommonVoltage(void)
{
    float averageVdcV = 0.0f;
    float estimatedDroopV = 0.0f;
    float errorV;
    float candidateIntegralV;
    float unsaturatedActionV;
    float actionV;
    bool integrate;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        averageVdcV += g_hil.vdc[i];
        estimatedDroopV +=
            (HIL_DROOP_BASE_V_PER_A / MAPPO_TRAIN_CAPACITY_RATIOS[i]) *
            g_ilVoltageLoopAveragedA[i];
    }
    averageVdcV *= 0.25f;
    estimatedDroopV *= 0.25f;
    g_commonFeedforwardTargetV = clampFloat(
        HIL_COMMON_CURRENT_FEEDFORWARD_GAIN * estimatedDroopV,
        0.0f,
        HIL_COMMON_VOLTAGE_MAX_V);
    g_commonFeedforwardFilteredV +=
        HIL_COMMON_FEEDFORWARD_FILTER_ALPHA *
        (g_commonFeedforwardTargetV - g_commonFeedforwardFilteredV);
    errorV = HIL_PI_VREF - averageVdcV;
    candidateIntegralV = clampFloat(
        g_commonVoltageIntegralV +
        HIL_COMMON_VOLTAGE_KI * errorV * HIL_COMMON_LOOP_TS_SEC,
        HIL_COMMON_VOLTAGE_MIN_V,
        HIL_COMMON_VOLTAGE_MAX_V);
    unsaturatedActionV =
        HIL_COMMON_VOLTAGE_KP * errorV + candidateIntegralV +
        g_commonFeedforwardFilteredV;
    actionV = clampFloat(
        unsaturatedActionV,
        HIL_COMMON_VOLTAGE_MIN_V,
        HIL_COMMON_VOLTAGE_MAX_V);
    integrate =
        ((unsaturatedActionV >= HIL_COMMON_VOLTAGE_MIN_V) &&
         (unsaturatedActionV <= HIL_COMMON_VOLTAGE_MAX_V)) ||
        ((actionV >= HIL_COMMON_VOLTAGE_MAX_V) && (errorV < 0.0f)) ||
        ((actionV <= HIL_COMMON_VOLTAGE_MIN_V) && (errorV > 0.0f));
    if(integrate) {
        g_commonVoltageIntegralV = candidateIntegralV;
    }

    g_commonVoltageErrorV = errorV;
    g_commonVoltageActionV = actionV;
    g_commonVoltageFilteredV += HIL_COMMON_ACTION_FILTER_ALPHA *
        (actionV - g_commonVoltageFilteredV);
    for(i = 0U; i < 4U; i++) {
        g_secondaryVoltageV[i] = g_commonVoltageFilteredV +
                                 g_mappoDifferentialFilteredV[i];
        g_panel.secondaryVoltageV[i] = g_secondaryVoltageV[i];
    }
}

static void captureMappoObservation(void)
{
    uint16_t i;

    g_secondaryLoopDivider++;
    if(g_secondaryLoopDivider >= HIL_MAPPO_OBSERVATION_PERIOD_SAMPLES) {
        g_secondaryLoopDivider = 0U;
        for(i = 0U; i < 4U; i++) {
            g_actorPendingCurrentA[i] = g_ilVoltageLoopAveragedA[i];
            g_actorPendingVdcV[i] = g_hil.vdc[i];
        }
        if(g_secondaryLoopPending != 0U) {
            g_secondaryDeadlineMissCount++;
        }
        g_secondaryLoopPending = 1U;
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

    for(i = 0U; i < HIL_ADC_INPUT_COUNT; i++) {
        raw = ADC_readResult(kInputs[i].resultBase, kInputs[i].socNumber);
        adcVoltage = rawToAdcVoltage(raw);
        calibratedEngineering = calibratedRawToEngineering(i, raw);

        /* Preserve instantaneous values so ADC wiring/noise remains visible. */
        g_hil.adcRaw[i] = raw;
        g_hil.adcVoltage[i] = adcVoltage;
        g_hil.adcCalibrated[i] = calibratedEngineering;

        if(i < 4U) {
            if(!g_adcFilterPrimed) {
                for(historySample = 0U;
                    historySample < HIL_ADC_MEDIAN_SAMPLES;
                    historySample++) {
                    g_adcRawHistory[i][historySample] = raw;
                }
                g_adcFilteredEngineering[i] = calibratedEngineering;
            } else {
                g_adcRawHistory[i][g_adcHistoryIndex] = raw;
                medianEngineering = calibratedRawToEngineering(
                    i, medianAdcRaw(g_adcRawHistory[i]));
                g_adcFilteredEngineering[i] +=
                    HIL_ADC_VOLTAGE_FILTER_ALPHA *
                    (medianEngineering - g_adcFilteredEngineering[i]);
            }
            g_hil.vdc[i] = g_adcFilteredEngineering[i];
        } else if(i < 8U) {
            /* Calibrated fast feedback deliberately bypasses median/IIR. */
            g_hil.il[i - 4U] = calibratedEngineering;
            /* Keep a coherent 1 ms current average for the droop equation. */
            g_ilVoltageLoopSampleSumA[i - 4U] += calibratedEngineering;
        } else {
            /* Accumulate one calibrated Vo window for the 1 kHz voltage PI. */
            g_voSampleSumV[i - 8U] += calibratedEngineering;
        }
    }

    g_voAverageCount++;
    if(g_voAverageCount >= HIL_VO_AVERAGE_SAMPLES) {
        const float averageScale = 1.0f / (float)HIL_VO_AVERAGE_SAMPLES;
        g_voAverageCount = 0U;
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
    if(g_adcHistoryIndex >= HIL_ADC_MEDIAN_SAMPLES) {
        g_adcHistoryIndex = 0U;
    }
}

#if 0
/* Legacy online experiment scoring is intentionally excluded from this firmware. */
static void updateControlMetrics(void)
{
    float voltageSum = 0.0f;
    float voltageAbsErrorSum = 0.0f;
    float maxVoltageDeviation = 0.0f;
    float minVoltage = g_hil.vdc[0];
    float maxVoltage = g_hil.vdc[0];
    float currentSum = 0.0f;
    float capacitySum = 0.0f;
    float shareErrorSum = 0.0f;
    float perCapacityErrorSum = 0.0f;
    float minCurrentPerCapacity;
    float maxCurrentPerCapacity;
    float deviation;
    float denominator;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        voltageSum += g_hil.vdc[i];
        deviation = fabsf(g_hil.vdc[i] - HIL_PI_VREF);
        voltageAbsErrorSum += deviation;
        if(deviation > maxVoltageDeviation) {
            maxVoltageDeviation = deviation;
        }
        if(g_hil.vdc[i] < minVoltage) {
            minVoltage = g_hil.vdc[i];
        }
        if(g_hil.vdc[i] > maxVoltage) {
            maxVoltage = g_hil.vdc[i];
        }

        currentSum += g_hil.il[i];
        capacitySum += MAPPO_TRAIN_CAPACITY_RATIOS[i];
        g_metrics.currentPerCapacityA[i] =
            g_hil.il[i] / MAPPO_TRAIN_CAPACITY_RATIOS[i];
    }

    g_metrics.averageVoltageV = voltageSum * 0.25f;
    g_metrics.meanVoltageErrorPu =
        voltageAbsErrorSum / (4.0f * HIL_PI_VREF);
    g_metrics.averageVoltageErrorPu =
        fabsf(g_metrics.averageVoltageV - HIL_PI_VREF) / HIL_PI_VREF;
    g_metrics.maxVoltageDeviationV = maxVoltageDeviation;
    g_metrics.maxVoltageErrorPu = maxVoltageDeviation / HIL_PI_VREF;
    g_metrics.minBusVoltageV = minVoltage;
    g_metrics.maxBusVoltageV = maxVoltage;
    g_metrics.voltageWithinSafeBand =
        (g_metrics.maxVoltageErrorPu <= HIL_VOLTAGE_SAFE_FRACTION) ? 1U : 0U;
    g_metrics.averageVoltageWithinTarget =
        (fabsf(g_metrics.averageVoltageV - HIL_PI_VREF) <=
         HIL_AVERAGE_VOLTAGE_BAND_V) ? 1U : 0U;

    g_metrics.shareLambdaA = currentSum / capacitySum;
    minCurrentPerCapacity = g_metrics.currentPerCapacityA[0];
    maxCurrentPerCapacity = g_metrics.currentPerCapacityA[0];
    for(i = 0U; i < 4U; i++) {
        g_metrics.targetCurrentA[i] =
            g_metrics.shareLambdaA * MAPPO_TRAIN_CAPACITY_RATIOS[i];
        shareErrorSum += fabsf(g_hil.il[i] - g_metrics.targetCurrentA[i]);
        perCapacityErrorSum += fabsf(
            g_metrics.currentPerCapacityA[i] - g_metrics.shareLambdaA);
        if(g_metrics.currentPerCapacityA[i] < minCurrentPerCapacity) {
            minCurrentPerCapacity = g_metrics.currentPerCapacityA[i];
        }
        if(g_metrics.currentPerCapacityA[i] > maxCurrentPerCapacity) {
            maxCurrentPerCapacity = g_metrics.currentPerCapacityA[i];
        }
    }

    g_metrics.meanCurrentShareErrorA = shareErrorSum * 0.25f;
    g_metrics.meanCurrentPerCapacityErrorA = perCapacityErrorSum * 0.25f;
    g_metrics.currentPerCapacitySpreadA =
        maxCurrentPerCapacity - minCurrentPerCapacity;
    denominator = fmaxf(fabsf(g_metrics.shareLambdaA),
                        0.1f * HIL_CURRENT_BASE);
    g_metrics.currentErrorPu =
        g_metrics.meanCurrentPerCapacityErrorA / denominator;
    g_metrics.currentWithinTarget =
        (g_metrics.currentErrorPu <= HIL_CURRENT_ERROR_TARGET) ? 1U : 0U;
    g_metrics.overallPass =
        ((g_metrics.voltageWithinSafeBand != 0U) &&
         (g_metrics.averageVoltageWithinTarget != 0U) &&
         (g_metrics.currentWithinTarget != 0U)) ? 1U : 0U;

    g_panel.averageVoltageV = g_metrics.averageVoltageV;
    g_panel.maxVoltageDeviationV = g_metrics.maxVoltageDeviationV;
    g_panel.minBusVoltageV = g_metrics.minBusVoltageV;
    g_panel.maxBusVoltageV = g_metrics.maxBusVoltageV;
    g_panel.shareLambdaA = g_metrics.shareLambdaA;
    g_panel.currentErrorPu = g_metrics.currentErrorPu;
    g_panel.voltagePass = g_metrics.voltageWithinSafeBand;
    g_panel.averageVoltagePass = g_metrics.averageVoltageWithinTarget;
    g_panel.currentPass = g_metrics.currentWithinTarget;
    g_panel.overallPass = g_metrics.overallPass;
    if(g_panel.overallPass != 0U) {
        if(g_panel.passHoldCount < HIL_PASS_HOLD_SAMPLES) {
            g_panel.passHoldCount++;
        }
    } else {
        g_panel.passHoldCount = 0U;
    }
    g_panel.qualifiedPass =
        (g_panel.passHoldCount >= HIL_PASS_HOLD_SAMPLES) ? 1U : 0U;
    for(i = 0U; i < 4U; i++) {
        g_panel.currentPerCapacityA[i] =
            g_metrics.currentPerCapacityA[i];
        g_panel.targetCurrentA[i] = g_metrics.targetCurrentA[i];
        g_panel.converterVoltageV[i] = g_hil.vo[i];
        g_panel.secondaryVoltageV[i] = g_secondaryVoltageV[i];
        g_panel.voltageReferenceOutV[i] = g_mappoVref[i];
    }
}
#endif

static void composeVoltageCommand(void)
{
    float droopCoefficient;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        droopCoefficient = HIL_DROOP_BASE_V_PER_A /
                           MAPPO_TRAIN_CAPACITY_RATIOS[i];
        g_voltageCommandV[i] = HIL_PI_VREF -
                               (droopCoefficient *
                                g_ilVoltageLoopAveragedA[i]) +
                               g_secondaryVoltageV[i];
    }
}

static void resetPiState(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_voltagePiIntegralA[i] = g_hil.il[i];
        g_voltageLoopErrorV[i] = 0.0f;
        g_currentReferenceA[i] = g_voltagePiIntegralA[i];
        g_currentPiIntegralDuty[i] = clampFloat(
            g_pwmDuty[i], HIL_DUTY_MIN, HIL_DUTY_MAX);
        g_currentLoopErrorA[i] = 0.0f;
        g_currentPiUnsatDuty[i] = HIL_FIXED_DUTY;
        g_panel.currentReferenceA[i] = g_currentReferenceA[i];
        g_panel.voltageLoopErrorV[i] = 0.0f;
        g_panel.currentLoopErrorA[i] = 0.0f;
    }
}

static void serviceControlPanel(void)
{
    uint16_t mode = g_panel.mode;

    if(mode > HIL_CONTROL_MODE_MAPPO_OPT) {
        mode = HIL_CONTROL_MODE_MAPPO_OPT;
        g_panel.mode = mode;
    }
    g_panel.voltagePiKp = clampFloat(
        g_panel.voltagePiKp, 0.0f, HIL_VOLTAGE_PI_KP_MAX);
    g_panel.voltagePiKi = clampFloat(
        g_panel.voltagePiKi, 0.0f, HIL_VOLTAGE_PI_KI_MAX);
    g_panel.currentPiKp = clampFloat(
        g_panel.currentPiKp, 0.0f, HIL_CURRENT_PI_KP_MAX);
    g_panel.currentPiKi = clampFloat(
        g_panel.currentPiKi, 0.0f, HIL_CURRENT_PI_KI_MAX);
    if((g_panel.reset != 0U) || (mode != g_lastControlMode)) {
        /* Keep the 10 kHz ISR on a known-safe output while slow states reset. */
        DINT;
        g_activeControlMode = HIL_CONTROL_MODE_FIXED;
        EINT;
        resetMappoActorState();
        if((g_panel.reset != 0U) ||
           (mode <= HIL_CONTROL_MODE_DROOP) ||
           (g_lastControlMode == HIL_CONTROL_MODE_FIXED)) {
            resetPiState();
        }
        DINT;
        g_voltageLoopDivider = 0U;
        g_secondaryLoopDivider = 0U;
        g_secondaryLoopPending = 0U;
        g_secondaryDeadlineMissCount = 0UL;
        resetPrimaryLoopAverageAccumulator();
        resetMappoObservationAccumulator();
        g_lastControlMode = mode;
        g_activeControlMode = mode;
        g_panel.reset = 0U;
        EINT;
    }
}

static void updateBaseVoltageReference(void)
{
    uint16_t i;

    DINT;
    for(i = 0U; i < 4U; i++) {
        g_secondaryVoltageV[i] = 0.0f;
        g_panel.secondaryVoltageV[i] = 0.0f;
    }
    EINT;
}

static void updateFixedDuty(void)
{
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_pwmDuty[i] = HIL_FIXED_DUTY;
    }
}

static void updateVoltageOuterLoop(void)
{
    float error;
    float referenceA;
    uint16_t i;

    for(i = 0U; i < 4U; i++) {
        g_mappoVref[i] = g_voltageCommandV[i];
        g_panel.voltageReferenceOutV[i] = g_mappoVref[i];

        /* The primary voltage loop regulates Vo, measured before line Rs. */
        error = g_mappoVref[i] - g_hil.vo[i];
        g_voltagePiIntegralA[i] +=
            (g_panel.voltagePiKi * error * HIL_VOLTAGE_LOOP_TS_SEC);
        referenceA = (g_panel.voltagePiKp * error) +
                     g_voltagePiIntegralA[i];

        g_voltageLoopErrorV[i] = error;
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

    for(i = 0U; i < 4U; i++) {
        error = g_currentReferenceA[i] - g_hil.il[i];
        proposedIntegralDuty = g_currentPiIntegralDuty[i] +
            (g_panel.currentPiKi * error * HIL_CURRENT_LOOP_TS_SEC);
        proposedIntegralDuty = clampFloat(
            proposedIntegralDuty,
            HIL_DUTY_MIN,
            HIL_DUTY_MAX);
        unsaturatedDuty =
            (g_panel.currentPiKp * error) + proposedIntegralDuty;
        duty = clampFloat(unsaturatedDuty, HIL_DUTY_MIN, HIL_DUTY_MAX);

        if((unsaturatedDuty == duty) ||
           ((duty <= HIL_DUTY_MIN) && (error > 0.0f)) ||
           ((duty >= HIL_DUTY_MAX) && (error < 0.0f))) {
            g_currentPiIntegralDuty[i] = proposedIntegralDuty;
        }

        g_currentLoopErrorA[i] = error;
        g_currentPiUnsatDuty[i] = unsaturatedDuty;
        g_panel.currentLoopErrorA[i] = error;
        g_pwmDuty[i] = duty;
    }
}

static void initGpioForPwm(void)
{
    GPIO_setPinConfig(HIL_PWM1_GPIO_MUX);
    GPIO_setPinConfig(HIL_PWM2_GPIO_MUX);
    GPIO_setPinConfig(HIL_PWM3_GPIO_MUX);
    GPIO_setPinConfig(HIL_PWM4_GPIO_MUX);

    GPIO_setPadConfig(HIL_PWM1_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(HIL_PWM2_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(HIL_PWM3_GPIO, GPIO_PIN_TYPE_STD);
    GPIO_setPadConfig(HIL_PWM4_GPIO, GPIO_PIN_TYPE_STD);

    GPIO_setQualificationMode(HIL_PWM1_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(HIL_PWM2_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(HIL_PWM3_GPIO, GPIO_QUAL_SYNC);
    GPIO_setQualificationMode(HIL_PWM4_GPIO, GPIO_QUAL_SYNC);
}

static uint16_t pwmPeriodTicks(void)
{
    return (uint16_t)(HIL_EPWM_TBCLK_HZ / (2UL * HIL_PWM_FREQ_HZ));
}

static uint16_t dutyToCmp(float duty, uint16_t period)
{
    if(duty < 0.0f) {
        duty = 0.0f;
    }
    if(duty > 1.0f) {
        duty = 1.0f;
    }
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
    EPWM_setCounterCompareShadowLoadMode(
        epwmBase,
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
    configureOnePwm(HIL_PWM1_BASE, g_pwmDuty[0]);
    configureOnePwm(HIL_PWM2_BASE, g_pwmDuty[1]);
    configureOnePwm(HIL_PWM3_BASE, g_pwmDuty[2]);
    configureOnePwm(HIL_PWM4_BASE, g_pwmDuty[3]);

    /* One deterministic ADC scan starts at the centre-aligned PWM zero event. */
    EPWM_setADCTriggerSource(
        HIL_PWM1_BASE, EPWM_SOC_A, EPWM_SOC_TBCTR_ZERO);
    EPWM_setADCTriggerEventPrescale(HIL_PWM1_BASE, EPWM_SOC_A, 1U);
    EPWM_enableADCTrigger(HIL_PWM1_BASE, EPWM_SOC_A);

    SysCtl_enablePeripheral(SYSCTL_PERIPH_CLK_TBCLKSYNC);
}

static void updatePwmDuty(void)
{
    uint16_t period = pwmPeriodTicks();

    EPWM_setCounterCompareValue(HIL_PWM1_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[0], period));
    EPWM_setCounterCompareValue(HIL_PWM2_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[1], period));
    EPWM_setCounterCompareValue(HIL_PWM3_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[2], period));
    EPWM_setCounterCompareValue(HIL_PWM4_BASE, EPWM_COUNTER_COMPARE_A,
                                dutyToCmp(g_pwmDuty[3], period));
}

__interrupt void adca1ISR(void)
{
    sampleHilOutputsFromAdc();
    g_currentLoopCount++;

    g_voltageLoopDivider++;
    if(g_voltageLoopDivider >= HIL_VOLTAGE_LOOP_DIVIDER) {
        g_voltageLoopDivider = 0U;
        if(g_activeControlMode >= HIL_CONTROL_MODE_DROOP) {
            if(g_activeControlMode == HIL_CONTROL_MODE_MAPPO_OPT) {
                updateFastCommonVoltage();
            }
            /* Rebuild droop + secondary command from this same 1 ms window. */
            composeVoltageCommand();
            updateVoltageOuterLoop();
        }
    }

    if(g_activeControlMode >= HIL_CONTROL_MODE_DROOP) {
        updateCurrentInnerLoop();
    } else {
        updateFixedDuty();
    }
    updatePwmDuty();

    captureMappoObservation();

    ADC_clearInterruptStatus(ADCA_BASE, ADC_INT_NUMBER1);
    if(ADC_getInterruptOverflowStatus(ADCA_BASE, ADC_INT_NUMBER1)) {
        ADC_clearInterruptOverflowStatus(ADCA_BASE, ADC_INT_NUMBER1);
        ADC_clearInterruptStatus(ADCA_BASE, ADC_INT_NUMBER1);
    }
    Interrupt_clearACKGroup(INTERRUPT_ACK_GROUP1);
}

void main(void)
{
    uint16_t runSecondary;
    uint16_t i;

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

        runSecondary = 0U;
        DINT;
        if(g_secondaryLoopPending != 0U) {
            g_secondaryLoopPending = 0U;
            for(i = 0U; i < 4U; i++) {
                g_mappoSnapshotCurrentA[i] = g_actorPendingCurrentA[i];
                g_mappoSnapshotVdcV[i] = g_actorPendingVdcV[i];
            }
            runSecondary = 1U;
        }
        EINT;

        if(runSecondary != 0U) {
            if(g_activeControlMode == HIL_CONTROL_MODE_MAPPO_OPT) {
                updateMappoSecondaryVoltage(
                    g_mappoSnapshotCurrentA, g_mappoSnapshotVdcV);
            } else {
                updateBaseVoltageReference();
            }
        }
    }
}
