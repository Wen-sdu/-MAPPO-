%% Typhoon HIL CSV 波形绘制
% 运行后选择 CSV 文件，生成三个独立 Figure：
% Figure 1：四个母线电压；Figure 2：平均母线电压；Figure 3：四条母线电流。
% 适用于列名：Time,Vdc1,Vdc2,Vdc3,Vdc4,Vaverage,IL1,IL2,IL3,IL4

clearvars; clc;

%% 1. 选择 CSV 文件
[fileName, filePath] = uigetfile({'*.csv', 'CSV Files (*.csv)'}, ...
    '请选择 Typhoon HIL 导出的 CSV 文件');
if isequal(fileName, 0)
    disp('已取消文件选择。');
    return;
end
csvFile = fullfile(filePath, fileName);

%% 2. 读取数据
% readmatrix 会将首行文本表头识别为非数值内容并跳过。
data = readmatrix(csvFile);
if size(data, 2) < 10
    error('CSV 列数不足：需要至少 10 列（Time, Vdc1~4, Vaverage, IL1~4）。');
end
% 某些 MATLAB/CSV 格式可能留下 NaN 表头行，清理掉非数值时间行。
data = data(isfinite(data(:, 1)), :);
if isempty(data)
    error('未能读取到有效数值数据。');
end

%% 3. 提取信号、调整时间轴
% 将原 CSV 的 -4~12 s 转换为 0~16 s（通用于其他起始时刻）。
t = data(:, 1) - data(1, 1);
Vdc1 = data(:, 2); Vdc2 = data(:, 3);
Vdc3 = data(:, 4); Vdc4 = data(:, 5);
Vaverage = data(:, 6);
IL1 = data(:, 7); IL2 = data(:, 8);
IL3 = data(:, 9); IL4 = data(:, 10);

%% 4. 绘图参数：可在此处调整
lineWidth = 2.0;        % 曲线线宽
tickFontSize = 20;      % 横纵坐标刻度数字字号（不加粗）
labelFontSize = 25;     % 横纵坐标标题字号（不加粗）
titleFontSize = 25;     % 图标题字号（加粗）
legendFontSize = 15;    % 图例字号
showGrid = true;        % 是否显示网格
% 使用全部有效采样点绘图，不进行间隔抽样，保留原始暂态尖峰。
% 注意：大 CSV 文件可能使绘图和交互速度降低。
idx = 1:numel(t);
tp = t;

%% Figure 1：各母线电压
figure('Name', '各母线电压', 'NumberTitle', 'off', 'Color', 'w');
plot(tp, Vdc1(idx), 'r-', 'LineWidth', lineWidth); hold on;
plot(tp, Vdc2(idx), 'g-', 'LineWidth', lineWidth);
plot(tp, Vdc3(idx), 'k-', 'LineWidth', lineWidth);
plot(tp, Vdc4(idx), 'b-', 'LineWidth', lineWidth);
hold off;
ax = gca;
set(ax, 'FontSize', tickFontSize, 'FontWeight', 'normal');
xlim([0 16]); xticks(0:4:16);
ylim([46 50]); yticks(46:1:50);
xlabel('Time (s)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
ylabel('Voltage (V)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
title('各母线电压', 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend('Vdc1', 'Vdc2', 'Vdc3', 'Vdc4', 'Location', 'northeast', ...
    'FontSize', legendFontSize, 'FontWeight', 'normal');
if showGrid, grid on; end

%% Figure 2：平均母线电压
figure('Name', '平均母线电压', 'NumberTitle', 'off', 'Color', 'w');
plot(tp, Vaverage(idx), 'r-', 'LineWidth', lineWidth);
ax = gca;
set(ax, 'FontSize', tickFontSize, 'FontWeight', 'normal');
xlim([0 16]); xticks(0:4:16);
ylim([46 50]); yticks(46:1:50);
xlabel('Time (s)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
ylabel('Voltage (V)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
title('平均母线电压', 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend('Average Voltage', 'Location', 'northeast', ...
    'FontSize', legendFontSize, 'FontWeight', 'normal');
if showGrid, grid on; end

%% Figure 3：各母线电流
figure('Name', '各母线电流', 'NumberTitle', 'off', 'Color', 'w');
plot(tp, IL1(idx), 'r-', 'LineWidth', lineWidth); hold on;
plot(tp, IL2(idx), 'g-', 'LineWidth', lineWidth);
plot(tp, IL3(idx), 'k-', 'LineWidth', lineWidth);
plot(tp, IL4(idx), 'b-', 'LineWidth', lineWidth);
hold off;
ax = gca;
set(ax, 'FontSize', tickFontSize, 'FontWeight', 'normal');
xlim([0 16]); xticks(0:4:16);
ylim([0 6]); yticks(0:1:6);
xlabel('Time (s)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
ylabel('Current (A)', 'FontSize', labelFontSize, 'FontWeight', 'normal');
title('各母线电流', 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend('IL1', 'IL2', 'IL3', 'IL4', 'Location', 'northeast', ...
    'FontSize', legendFontSize, 'FontWeight', 'normal');
if showGrid, grid on; end

disp(['绘图完成：', csvFile]);
fprintf('时间范围：%.6f s ~ %.6f s；总采样点数：%d；绘图点数：%d。\n', ...
    t(1), t(end), numel(t), numel(idx));
