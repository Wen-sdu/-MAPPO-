%% Typhoon HIL CSV 波形绘制（统一论文图格式）
% Figure 1：四个母线电压；Figure 2：平均母线电压；Figure 3：四条母线电流。
% CSV列：Time,Vdc1,Vdc2,Vdc3,Vdc4,Vaverage,IL1,IL2,IL3,IL4
clearvars; clc;

%% 1. 选择CSV文件
[fileName, filePath] = uigetfile({'*.csv', 'CSV Files (*.csv)'}, ...
    '请选择 Typhoon HIL 导出的 CSV 文件');
if isequal(fileName, 0)
    disp('已取消文件选择。');
    return;
end
csvFile = fullfile(filePath, fileName);

%% 2. 读取CSV并提取信号
raw = readmatrix(csvFile);
if size(raw, 2) < 10
    error('CSV列数不足：需要Time、Vdc1~4、Vaverage、IL1~4（至少10列）。');
end
raw = raw(isfinite(raw(:, 1)), :);
if isempty(raw)
    error('CSV中没有有效的时间数据。');
end

t = raw(:,1) - raw(1,1);  % CSV起始时间平移至0秒，保留全部采样点
Vdc = raw(:,2:5);
Vaverage = raw(:,6);
IL = raw(:,7:10);

%% 3. 全局绘图配置
cfg.lineWidth        = 4;
cfg.tickFontSize     = 36;
cfg.labelFontSize    = 40;
cfg.legendFontSize   = 32;
cfg.noteFontSize     = 24;
cfg.eventColor       = [0.64, 0.08, 0.18];
cfg.eventLineWidth   = 2;
cfg.arrowLineWidth   = 10;
cfg.arrowHeadLength  = 20;
cfg.arrowHeadWidth   = 20;
cfg.showGrid         = true;
cfg.figPosition      = [100, 100, 2200, 1050];

%% Figure 1：各母线电压
fig1 = figure('Name','各母线电压','NumberTitle','off', ...
    'Color','w','Position',cfg.figPosition);
ax1 = axes('Parent',fig1);
h = plot(ax1,t,Vdc,'LineWidth',cfg.lineWidth);
set(h(1),'Color','r'); set(h(2),'Color','g');
set(h(3),'Color','k'); set(h(4),'Color','b');
formatWaveform(ax1, '电压（V）', [46,50], 46:1:50, ...
    h, {'Vdc1','Vdc2','Vdc3','Vdc4'}, cfg);
addLoadEventsExact(fig1,ax1,cfg,'voltage');

%% Figure 2：平均母线电压
fig2 = figure('Name','平均母线电压','NumberTitle','off', ...
    'Color','w','Position',cfg.figPosition);
ax2 = axes('Parent',fig2);
h = plot(ax2,t,Vaverage,'r-','LineWidth',cfg.lineWidth);
formatWaveform(ax2, '平均电压（V）', [46,50], 46:1:50, ...
    h, {'Average Voltage'}, cfg);
addLoadEventsExact(fig2,ax2,cfg,'voltage');

%% Figure 3：各母线电流
fig3 = figure('Name','各母线电流','NumberTitle','off', ...
    'Color','w','Position',cfg.figPosition);
ax3 = axes('Parent',fig3);
h = plot(ax3,t,IL,'LineWidth',cfg.lineWidth);
set(h(1),'Color','r'); set(h(2),'Color','g');
set(h(3),'Color','k'); set(h(4),'Color','b');
formatWaveform(ax3, '电流（A）', [0,6], 0:1:6, ...
    h, {'IL1','IL2','IL3','IL4'}, cfg);
addLoadEventsExact(fig3,ax3,cfg,'current');

disp(['绘图完成：',csvFile]);
fprintf('时间范围：%.6f s ~ %.6f s；总采样点数：%d。\n', ...
    t(1),t(end),numel(t));

%% ===== 局部函数 =====
function formatWaveform(ax,yLabel,yRange,yTick,curveHandles,legendLabels,cfg)
    set(ax,'FontSize',cfg.tickFontSize,'FontWeight','bold', ...
        'FontName','Microsoft YaHei','Box','on','Layer','top');
    xlim(ax,[0,16]);
    xticks(ax,0:4:16);
    ylim(ax,yRange);
    yticks(ax,yTick);

    xlabel(ax,'时间（s）','FontSize',cfg.labelFontSize, ...
        'FontWeight','bold','FontName','Microsoft YaHei');
    ylabel(ax,yLabel,'FontSize',cfg.labelFontSize, ...
        'FontWeight','bold','FontName','Microsoft YaHei');
    title(ax,'');

    if cfg.showGrid
        grid(ax,'on');
    else
        grid(ax,'off');
    end

    lgd = legend(ax,curveHandles,legendLabels, ...
        'Location','north','Orientation','horizontal', ...
        'FontSize',cfg.legendFontSize,'FontWeight','normal', ...
        'FontName','Microsoft YaHei','Box','on');
    if isprop(lgd,'IconColumnWidth')
        lgd.IconColumnWidth = 40;
    end
end

function addLoadEventsExact(fig,ax,cfg,plotType)
    hold(ax,'on');
    h4 = xline(ax,4,'--','Color',cfg.eventColor, ...
        'LineWidth',cfg.eventLineWidth,'HandleVisibility','off');
    h12 = xline(ax,12,'--','Color',cfg.eventColor, ...
        'LineWidth',cfg.eventLineWidth,'HandleVisibility','off');
    if isprop(h4,'Annotation')
        h4.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    if isprop(h12,'Annotation')
        h12.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    hold(ax,'off');

    % 各母线电压与平均电压采用相同位置；各母线电流单独采用截图位置。
    if strcmp(plotType,'voltage')
        arrow1X = [0.4303, 0.3237];
        arrow1Y = [0.3119, 0.3119];
        arrow2X = [0.8228, 0.7162];
        arrow2Y = [0.3081, 0.3081];
        note1Pos = [7.7592, 47.0684, 0];
        note2Pos = [15.8108, 47.0862, 0];
    else
        arrow1X = [0.4350, 0.3284];
        arrow1Y = [0.2234, 0.2234];
        arrow2X = [0.8264, 0.7198];
        arrow2Y = [0.2261, 0.2261];
        note1Pos = [7.8022, 0.9319, 0];
        note2Pos = [15.8108, 0.8782, 0];
    end

    % 箭头采用整幅Figure的normalized坐标，以复现截图中的精确位置。
    a1 = annotation(fig,'arrow');
    set(a1,'Color',cfg.eventColor,'LineStyle','-', ...
        'LineWidth',cfg.arrowLineWidth,'HeadStyle','plain', ...
        'HeadLength',cfg.arrowHeadLength,'HeadWidth',cfg.arrowHeadWidth, ...
        'Units','normalized','X',arrow1X,'Y',arrow1Y);

    % 文本保持 data 单位，黑色、微软雅黑、24号、加粗、右对齐、顶部对齐。
    txt1 = text(ax,note1Pos(1),note1Pos(2),'R_{L2} = 40Ω →15Ω', ...
        'Units','data','Color',[0,0,0],'Interpreter','tex', ...
        'FontName','Microsoft YaHei','FontSize',cfg.noteFontSize, ...
        'FontWeight','bold','Rotation',0, ...
        'EdgeColor','none','BackgroundColor','none', ...
        'HorizontalAlignment','right','VerticalAlignment','top', ...
        'Clipping','on');

    a2 = annotation(fig,'arrow');
    set(a2,'Color',cfg.eventColor,'LineStyle','-', ...
        'LineWidth',cfg.arrowLineWidth,'HeadStyle','plain', ...
        'HeadLength',cfg.arrowHeadLength,'HeadWidth',cfg.arrowHeadWidth, ...
        'Units','normalized','X',arrow2X,'Y',arrow2Y);

    txt2 = text(ax,note2Pos(1),note2Pos(2),'R_{L2} = 15Ω →40Ω', ...
        'Units','data','Color',[0,0,0],'Interpreter','tex', ...
        'FontName','Microsoft YaHei','FontSize',cfg.noteFontSize, ...
        'FontWeight','bold','Rotation',0, ...
        'EdgeColor','none','BackgroundColor','none', ...
        'HorizontalAlignment','right','VerticalAlignment','top', ...
        'Clipping','on');

    if isprop(txt1,'AffectAutoLimits')
        txt1.AffectAutoLimits = 'off';
    end
    if isprop(txt2,'AffectAutoLimits')
        txt2.AffectAutoLimits = 'off';
    end
end
