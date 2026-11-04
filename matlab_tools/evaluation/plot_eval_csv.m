clear;
clc;
close all;

%% 选择 normal 或 noise0 评估CSV
[fileName, folderName] = uigetfile("*.csv", "选择MAPPO评估CSV文件");
if isequal(fileName, 0)
    return;
end

filePath = fullfile(folderName, fileName);
T = readtable(filePath, "VariableNamingRule", "preserve");
variables = string(T.Properties.VariableNames);

required = [
    "step", "V0", "V1", "V2", "V3", ...
    "I0", "I1", "I2", "I3", ...
    "u0", "u1", "u2", "u3", ...
    "average_voltage_error", "max_voltage_error", ...
    "current_error", "action_change"
];
missing = required(~ismember(required, variables));
if ~isempty(missing)
    error("CSV缺少以下列：%s", strjoin(missing, ", "));
end

%% 时间和负载事件
if ismember("time_s", variables)
    t = T.time_s;
else
    t = T.step * 0.01;
end

eventTimes = [];
if ismember("load_event", variables)
    eventTimes = t(T.load_event ~= 0);
end
settlingTime = 0.20;

V = [T.V0, T.V1, T.V2, T.V3];
I = [T.I0, T.I1, T.I2, T.I3];
U = [T.u0, T.u1, T.u2, T.u3];
if ismember("average_voltage", variables)
    averageVoltage = T.average_voltage;
else
    averageVoltage = mean(V, 2);
end

%% 图1：母线电压
f1 = newFigure("母线电压");
ax1 = axes(f1);
pV = plot(ax1, t, V, "LineWidth", 1.25);
hold(ax1, "on");
pAvg = plot(ax1, t, averageVoltage, "--k", "LineWidth", 1.8);
pRef = yline(ax1, 48, "--r", "48 V目标", "LineWidth", 1.1);
yline(ax1, 47.75, ":", "47.75 V", "HandleVisibility", "off");
yline(ax1, 48.25, ":", "48.25 V", "HandleVisibility", "off");
title(ax1, "MAPPO确定性评估：母线电压");
xlabel(ax1, "时间（s）");
ylabel(ax1, "电压（V）");
legend(ax1, [pV(:); pAvg; pRef], ...
    ["V_1", "V_2", "V_3", "V_4", "平均电压", "48 V目标"], ...
    "Location", "best");
formatAxes(ax1, t, eventTimes, settlingTime);

%% 图2：支路电流
f2 = newFigure("支路电流");
ax2 = axes(f2);
pI = plot(ax2, t, I, "LineWidth", 1.25);
title(ax2, "MAPPO确定性评估：各支路电流");
xlabel(ax2, "时间（s）");
ylabel(ax2, "电流（A）");
legend(ax2, pI, ["I_{L1}", "I_{L2}", "I_{L3}", "I_{L4}"], ...
    "Location", "best");
formatAxes(ax2, t, eventTimes, settlingTime);

%% 图3：实际施加的二次控制动作
f3 = newFigure("MAPPO动作");
ax3 = axes(f3);
pU = plot(ax3, t, U, "LineWidth", 1.25);
title(ax3, "MAPPO确定性评估：二次控制动作");
xlabel(ax3, "时间（s）");
ylabel(ax3, "动作（V）");
legend(ax3, pU, ["u_1", "u_2", "u_3", "u_4"], "Location", "best");
formatAxes(ax3, t, eventTimes, settlingTime);

%% 图4：控制误差与动作变化
f4 = newFigure("误差与动作变化");
ax4 = axes(f4);
yyaxis(ax4, "left");
errors = [
    T.average_voltage_error * 100, ...
    T.max_voltage_error * 100, ...
    T.current_error * 100
];
pErr = plot(ax4, t, errors, "LineWidth", 1.25);
ylabel(ax4, "误差（%）");

yyaxis(ax4, "right");
pChange = plot(ax4, t, T.action_change, ...
    "Color", [0.85, 0.10, 0.10], "LineWidth", 1.25);
ylabel(ax4, "动作变化量（V）");
title(ax4, "控制误差与动作变化");
xlabel(ax4, "时间（s）");
legend(ax4, [pErr(:); pChange], ...
    ["平均电压误差", "最大母线电压误差", "均流误差", "动作变化量"], ...
    "Location", "best");
formatAxes(ax4, t, eventTimes, settlingTime);

%% 图5：新增稳态波动指标（CSV包含这些列时绘制）
stabilityColumns = [
    "action_delta_rms_v", ...
    "current_per_capacity_delta_rms_a", ...
    "weighted_steady_stability_penalty"
];
if all(ismember(stabilityColumns, variables))
    f5 = newFigure("稳态波动指标");
    ax5 = axes(f5);
    yyaxis(ax5, "left");
    pStability = plot(ax5, t, ...
        [T.action_delta_rms_v, T.current_per_capacity_delta_rms_a], ...
        "LineWidth", 1.25);
    ylabel(ax5, "动作/电流变化RMS");

    yyaxis(ax5, "right");
    pPenalty = plot(ax5, t, T.weighted_steady_stability_penalty, ...
        "Color", [0.55, 0.15, 0.70], "LineWidth", 1.25);
    ylabel(ax5, "加权稳态惩罚");
    title(ax5, "稳态动作与电流波动指标");
    xlabel(ax5, "时间（s）");
    legend(ax5, [pStability(:); pPenalty], ...
        ["动作变化RMS（V）", "容量归一化电流变化RMS（A）", "加权稳态惩罚"], ...
        "Location", "best");
    formatAxes(ax5, t, eventTimes, settlingTime);
end

%% 图6：公共电压PI与Actor差模动作
decoupledColumns = [
    "common_voltage_action_v", ...
    "du0", "du1", "du2", "du3"
];
if all(ismember(decoupledColumns, variables))
    f6 = newFigure("公共电压PI与差模动作");
    ax6 = axes(f6);
    if ismember("common_voltage_filtered_v", variables)
        commonApplied = T.common_voltage_filtered_v;
    else
        commonApplied = T.common_voltage_action_v;
    end
    if all(ismember(["duf0", "duf1", "duf2", "duf3"], variables))
        differentialApplied = [T.duf0, T.duf1, T.duf2, T.duf3];
    else
        differentialApplied = [T.du0, T.du1, T.du2, T.du3];
    end
    pCommon = plot(ax6, t, commonApplied, ...
        "--k", "LineWidth", 1.8);
    hold(ax6, "on");
    pDiff = plot(ax6, t, differentialApplied, ...
        "LineWidth", 1.15);
    title(ax6, "平均电压公共PI与MAPPO差模均流动作");
    xlabel(ax6, "时间（s）");
    ylabel(ax6, "二次电压修正（V）");
    legend(ax6, [pCommon; pDiff(:)], ...
        ["u_{common}", "du_1", "du_2", "du_3", "du_4"], ...
        "Location", "best");
    formatAxes(ax6, t, eventTimes, settlingTime);
end

fprintf("已绘制评估文件：%s\n", filePath);

%% 本地函数
function f = newFigure(name)
    f = figure( ...
        "Color", "white", ...
        "Name", name, ...
        "NumberTitle", "off", ...
        "MenuBar", "figure", ...
        "ToolBar", "figure");
end

function formatAxes(ax, t, eventTimes, settlingTime)
    grid(ax, "on");
    box(ax, "on");
    set(ax, "FontName", "Microsoft YaHei", "FontSize", 11, "LineWidth", 1);
    if max(t) > min(t)
        xlim(ax, [min(t), max(t)]);
    end
    hold(ax, "on");
    for eventTime = reshape(eventTimes, 1, [])
        xline(ax, eventTime, ":", "负载事件", ...
            "Color", [0.50, 0.00, 0.60], "LineWidth", 1.1, ...
            "HandleVisibility", "off");
        xline(ax, eventTime + settlingTime, "--", "0.2 s稳定期", ...
            "Color", [0.40, 0.40, 0.40], "LineWidth", 0.9, ...
            "HandleVisibility", "off");
    end
end
