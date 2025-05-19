%% Two-Dimensional Time-Domain Measures Analysis
% This script analyzes fixation density maps by calculating 2D distribution
% properties (skewness and kurtosis) and compares these metrics between Group 1 and Group 2.
%
% REQUIRES: Output files from detect_bilateral_peaks_and_extract_v1.m
%
% Author: Y. Shigemune
% Created: 5/10/2025
% Last Modified: 5/10/2025
% Version: 1.0.0
%
% Description:
%   This script implements Mardia's 2D skewness and kurtosis measures to
%   compare distributional properties of gaze patterns between two groups.
%   It performs statistical comparisons with Bonferroni correction for
%   multiple comparisons across phases.
%
% Prerequisites:
%   - MATLAB R2017b or later
%   - Statistics and Machine Learning Toolbox (for ttest2)
%   - External functions: readfromexcel, imgaussfilt (optional)
%   - Completed bilateral peak detection with detect_bilateral_peaks_and_extract_v1.m
%
% Input files:
%   - PixelMap Excel files in Output_Group1/ and Output_Group2/ directories
%   - Format: Output from detect_bilateral_peaks_and_extract_v1.m
%   - Structure: 201x201 pixel regions with phase-specific data
%
% Output files:
%   - TDM_Results/[Phase]/ directories containing:
%     - Statistical analysis figures (heat maps, boxplots)
%     - Numerical results in MAT and text formats
%     - Cross-phase comparison visualizations
%
% Key Parameters:
%   - Resolution_X, Resolution_Y: Analysis grid size (default: 201x201)
%   - sigma: Gaussian smoothing parameter (default: 10)
%   - alpha: Statistical significance level (default: 0.05)
%   - use_single_precision: Memory optimization option
%
% Analysis Methods:
%   - 1D skewness and kurtosis for X and Y dimensions
%   - Mardia's multivariate 2D skewness and kurtosis
%   - Independent samples t-tests with Bonferroni correction
%   - Effect size calculations (Cohen's d)
%
% Usage:
%   1. Complete detect_bilateral_peaks_and_extract_v1.m
%   2. Place output files in Output_Group1/ and Output_Group2/
%   3. Run script: two_sample_TDM_v1
%   4. Check results in TDM_Results/ directory
%
% References:
%   Part of TFCE Moments Eyetracking Toolbox
%   Requires peak area data from detect_bilateral_peaks_and_extract_v1.m

%% Parameter settings - Modify as needed
% Analysis parameters
Resolution_X = 201;      % Display width (pixels)
Resolution_Y = 201;      % Display height (pixels)
center_x = 101;          % Center X coordinate (default is center of the image)
center_y = 101;          % Center Y coordinate (default is center of the image)
sigma = 10;              % Gaussian smoothing sigma (pixels)
alpha = 0.05;            % Statistical significance level

% Memory optimization settings
use_single_precision = false; % Use single precision to save memory
clear_temp_data = false;     % Clear temporary data to free memory

% Performance measurement
enable_timing = true;     % Enable timing measurements for performance analysis

% Paths - Edit according to your folder structure
curDir = pwd;
group1_folder = strcat(curDir, filesep, 'Output_Group1');  % Folder containing Group 1 PixelMap Excel files
group2_folder = strcat(curDir, filesep, 'Output_Group2');  % Folder containing Group 2 PixelMap Excel files
results_base_folder = strcat(curDir, filesep, 'TDM_Results'); % Base folder to save results

% Phases to analyze (Excel column indices)
phases = {'DecisionL', 'DecisionR', 'FeedbackL', 'FeedbackR'};
phase_cols = [3, 4, 5, 6]; % Phase Excel column indices

% Create base results folder if it doesn't exist
if ~exist(results_base_folder, 'dir')
    mkdir(results_base_folder);
end

% Start timer for overall performance measurement
if enable_timing
    total_start_time = tic;
end

% Set precision type based on parameters
if use_single_precision
    precision_type = 'single';
else
    precision_type = 'double';
end

%% Process each phase separately
for phase_idx = 1:length(phases)
    current_phase = phases{phase_idx};
    current_phase_col = phase_cols(phase_idx);
    
    fprintf('\n\n========== Processing %s phase ==========\n\n', current_phase);
    
    % Start timer for phase performance measurement
    if enable_timing
        phase_start_time = tic;
    end
    
    % Create phase-specific results folder
    results_folder = fullfile(results_base_folder, current_phase);
    if ~exist(results_folder, 'dir')
        mkdir(results_folder);
    end
    
    %% Load data from Excel files and read PixelMap format
    fprintf('Reading data from %s phase PixelMap Excel files...\n', current_phase);
    
    % Load Group 1 data
    group1_files = dir(fullfile(group1_folder, '*_PixelMap.xlsx'));
    n_subj_g1 = length(group1_files);
    fprintf('Detected %d subjects in Group 1\n', n_subj_g1);
    
    % Initialize Group 1 data structures (subjects x X x Y)
    group1_fixations = zeros(n_subj_g1, Resolution_X, Resolution_Y, precision_type);
    group1_skewness_x = zeros(n_subj_g1, 1, precision_type);
    group1_skewness_y = zeros(n_subj_g1, 1, precision_type);
    group1_kurtosis_x = zeros(n_subj_g1, 1, precision_type);
    group1_kurtosis_y = zeros(n_subj_g1, 1, precision_type);
    group1_skewness_2d = zeros(n_subj_g1, 1, precision_type);
    group1_kurtosis_2d = zeros(n_subj_g1, 1, precision_type);
    
    % Process each Group 1 file
    for i = 1:n_subj_g1
        % Read Excel file
        file_path = fullfile(group1_folder, group1_files(i).name);
        fprintf('Group 1, Subject %d: Processing %s\n', i, group1_files(i).name);
        
        % Try reading using readfromexcel
        try
            % Read all data using readfromexcel
            data = readfromexcel(file_path, 'sheet', 'Sheet1', 'All');
            header = data(1,:);
            data = data(2:end,:);
            
            % Extract data
            x_coords = cell2mat(data(:, 1));
            y_coords = cell2mat(data(:, 2));
            
            % Select appropriate phase data
            fixation_data = cell2mat(data(:, current_phase_col));
            
        catch
            warning('Could not read file %s. Skipping.', file_path);
            continue;
        end
        
        % Create fixation map - Using sparse matrix for efficiency
        fixation_map = sparse(x_coords, y_coords, fixation_data, Resolution_X, Resolution_Y);
        
        % Convert to full matrix for smoothing
        fixation_map = full(fixation_map);
        
        % Apply Gaussian smoothing using sigma from parameter section
        if exist('imgaussfilt', 'file')
            fixation_map = imgaussfilt(fixation_map, sigma);
        else
            % Fallback to custom Gaussian filter if imgaussfilt is unavailable
            filter_size = ceil(sigma * 3) * 2 + 1;
            if mod(filter_size, 2) == 0
                filter_size = filter_size + 1;
            end
            [x, y] = meshgrid(-(filter_size-1)/2:(filter_size-1)/2, -(filter_size-1)/2:(filter_size-1)/2);
            gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
            fixation_map = conv2(double(fixation_map), gaussian_kernel, 'same');
        end
        
        % Store in 3D array with specified precision
        group1_fixations(i, :, :) = cast(fixation_map, precision_type);
        
        % Calculate distribution properties (skewness and kurtosis)
        % ----------------------------------
        % Normalize the fixation map (probability distribution)
        total_fixation = sum(fixation_map(:));
        
        % Avoid division by zero
        if total_fixation > 0
            normalized_map = fixation_map / total_fixation;
        else
            normalized_map = fixation_map;
            warning('Subject %d has zero total fixation.', i);
            continue;
        end
        
        % Create coordinate grids for moments calculation
        [y_grid, x_grid] = meshgrid(1:Resolution_Y, 1:Resolution_X);
        
        % Calculate centers of mass (means)
        mean_x = sum(sum(normalized_map .* x_grid));
        mean_y = sum(sum(normalized_map .* y_grid));
        
        % Calculate central moments
        % Variances
        var_x = sum(sum(normalized_map .* (x_grid - mean_x).^2));
        var_y = sum(sum(normalized_map .* (y_grid - mean_y).^2));
        
        % Standard deviations
        std_x = sqrt(var_x);
        std_y = sqrt(var_y);
        
        % Skewness (third standardized moment) - indicates asymmetry
        if std_x > 0
            skew_x = sum(sum(normalized_map .* ((x_grid - mean_x) / std_x).^3));
        else
            skew_x = 0;
        end
        
        if std_y > 0
            skew_y = sum(sum(normalized_map .* ((y_grid - mean_y) / std_y).^3));
        else
            skew_y = 0;
        end
        
        % Kurtosis (fourth standardized moment) - indicates peakedness
        if std_x > 0
            kurt_x = sum(sum(normalized_map .* ((x_grid - mean_x) / std_x).^4)) - 3; % Excess kurtosis (normal = 0)
        else
            kurt_x = 0;
        end
        
        if std_y > 0
            kurt_y = sum(sum(normalized_map .* ((y_grid - mean_y) / std_y).^4)) - 3; % Excess kurtosis (normal = 0)
        else
            kurt_y = 0;
        end
        
        % Calculate 2D skewness and kurtosis using Mardia's formal measures
        % Normalize the fixation map (probability distribution)
        total_fixation = sum(fixation_map(:));
        
        % Avoid division by zero
        if total_fixation > 0
            normalized_map = fixation_map / total_fixation;
        else
            normalized_map = fixation_map;
            warning('Subject %d has zero total fixation.', i);
            continue;
        end
        
        % Calculate Mardia's 2D skewness and kurtosis measures
        [skew_2d, kurt_2d] = calculate_mardia_2d_measures(normalized_map);
        
        % Store calculated metrics
        group1_skewness_x(i) = skew_x;
        group1_skewness_y(i) = skew_y;
        group1_kurtosis_x(i) = kurt_x;
        group1_kurtosis_y(i) = kurt_y;
        group1_skewness_2d(i) = skew_2d;
        group1_kurtosis_2d(i) = kurt_2d;
        
        % Display calculated metrics
        fprintf('  Mean position: (%.2f, %.2f)\n', mean_x, mean_y);
        fprintf('  Standard deviations: (%.2f, %.2f)\n', std_x, std_y);
        fprintf('  Skewness (X, Y): (%.4f, %.4f)\n', skew_x, skew_y);
        fprintf('  Kurtosis (X, Y): (%.4f, %.4f)\n', kurt_x, kurt_y);
        fprintf('  2D Skewness: %.4f\n', skew_2d);
        fprintf('  2D Kurtosis: %.4f\n', kurt_2d);
    end
    
    % Load Group 2 data (similar process)
    group2_files = dir(fullfile(group2_folder, '*_PixelMap.xlsx'));
    n_subj_g2 = length(group2_files);
    fprintf('Detected %d subjects in Group 2\n', n_subj_g2);
    
    % Initialize Group 2 data structures
    group2_fixations = zeros(n_subj_g2, Resolution_X, Resolution_Y, precision_type);
    group2_skewness_x = zeros(n_subj_g2, 1, precision_type);
    group2_skewness_y = zeros(n_subj_g2, 1, precision_type);
    group2_kurtosis_x = zeros(n_subj_g2, 1, precision_type);
    group2_kurtosis_y = zeros(n_subj_g2, 1, precision_type);
    group2_skewness_2d = zeros(n_subj_g2, 1, precision_type);
    group2_kurtosis_2d = zeros(n_subj_g2, 1, precision_type);
    
    % Process each Group 2 file
    for i = 1:n_subj_g2
        % Read Excel file
        file_path = fullfile(group2_folder, group2_files(i).name);
        fprintf('Group 2, Subject %d: Processing %s\n', i, group2_files(i).name);
        
        % Try reading using readfromexcel
        try
            % Read all data using readfromexcel
            data = readfromexcel(file_path, 'sheet', 'Sheet1', 'All');
            header = data(1,:);
            data = data(2:end,:);
            
            % Extract data
            x_coords = cell2mat(data(:, 1));
            y_coords = cell2mat(data(:, 2));
            
            % Select appropriate phase data
            fixation_data = cell2mat(data(:, current_phase_col));
            
        catch
            warning('Could not read file %s. Skipping.', file_path);
            continue;
        end
        
        % Create fixation map - Using sparse matrix for efficiency
        fixation_map = sparse(x_coords, y_coords, fixation_data, Resolution_X, Resolution_Y);
        
        % Convert to full matrix for smoothing
        fixation_map = full(fixation_map);
        
        % Apply Gaussian smoothing using sigma from parameter section
        if exist('imgaussfilt', 'file')
            fixation_map = imgaussfilt(fixation_map, sigma);
        else
            % Fallback to custom Gaussian filter if imgaussfilt is unavailable
            filter_size = ceil(sigma * 3) * 2 + 1;
            if mod(filter_size, 2) == 0
                filter_size = filter_size + 1;
            end
            [x, y] = meshgrid(-(filter_size-1)/2:(filter_size-1)/2, -(filter_size-1)/2:(filter_size-1)/2);
            gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
            fixation_map = conv2(double(fixation_map), gaussian_kernel, 'same');
        end
        
        % Store in 3D array with specified precision
        group2_fixations(i, :, :) = cast(fixation_map, precision_type);
        
        % Calculate distribution properties (skewness and kurtosis)
        % ----------------------------------
        % Normalize the fixation map (probability distribution)
        total_fixation = sum(fixation_map(:));
        
        % Avoid division by zero
        if total_fixation > 0
            normalized_map = fixation_map / total_fixation;
        else
            normalized_map = fixation_map;
            warning('Subject %d has zero total fixation.', i);
            continue;
        end
        
        % Create coordinate grids for moments calculation
        [y_grid, x_grid] = meshgrid(1:Resolution_Y, 1:Resolution_X);
        
        % Calculate centers of mass (means)
        mean_x = sum(sum(normalized_map .* x_grid));
        mean_y = sum(sum(normalized_map .* y_grid));
        
        % Calculate central moments
        % Variances
        var_x = sum(sum(normalized_map .* (x_grid - mean_x).^2));
        var_y = sum(sum(normalized_map .* (y_grid - mean_y).^2));
        
        % Standard deviations
        std_x = sqrt(var_x);
        std_y = sqrt(var_y);
        
        % Skewness (third standardized moment) - indicates asymmetry
        if std_x > 0
            skew_x = sum(sum(normalized_map .* ((x_grid - mean_x) / std_x).^3));
        else
            skew_x = 0;
        end
        
        if std_y > 0
            skew_y = sum(sum(normalized_map .* ((y_grid - mean_y) / std_y).^3));
        else
            skew_y = 0;
        end
        
        % Kurtosis (fourth standardized moment) - indicates peakedness
        if std_x > 0
            kurt_x = sum(sum(normalized_map .* ((x_grid - mean_x) / std_x).^4)) - 3; % Excess kurtosis (normal = 0)
        else
            kurt_x = 0;
        end
        
        if std_y > 0
            kurt_y = sum(sum(normalized_map .* ((y_grid - mean_y) / std_y).^4)) - 3; % Excess kurtosis (normal = 0)
        else
            kurt_y = 0;
        end
        
        % Calculate 2D skewness and kurtosis using Mardia's formal measures
        % Normalize the fixation map (probability distribution)
        total_fixation = sum(fixation_map(:));
        
        % Avoid division by zero
        if total_fixation > 0
            normalized_map = fixation_map / total_fixation;
        else
            normalized_map = fixation_map;
            warning('Subject %d has zero total fixation.', i);
            continue;
        end
        
        % Calculate Mardia's 2D skewness and kurtosis measures
        [skew_2d, kurt_2d] = calculate_mardia_2d_measures(normalized_map);
        
        % Store calculated metrics
        group2_skewness_x(i) = skew_x;
        group2_skewness_y(i) = skew_y;
        group2_kurtosis_x(i) = kurt_x;
        group2_kurtosis_y(i) = kurt_y;
        group2_skewness_2d(i) = skew_2d;
        group2_kurtosis_2d(i) = kurt_2d;
        
        % Display calculated metrics
        fprintf('  Mean position: (%.2f, %.2f)\n', mean_x, mean_y);
        fprintf('  Standard deviations: (%.2f, %.2f)\n', std_x, std_y);
        fprintf('  Skewness (X, Y): (%.4f, %.4f)\n', skew_x, skew_y);
        fprintf('  Kurtosis (X, Y): (%.4f, %.4f)\n', kurt_x, kurt_y);
        fprintf('  2D Skewness: %.4f\n', skew_2d);
        fprintf('  2D Kurtosis: %.4f\n', kurt_2d);
    end
    
    %% Statistical comparison of distribution metrics
    fprintf('Performing statistical comparison of distribution metrics...\n');
    
    % Initialize result matrices for statistical tests
    % Structure: [skewness_x, skewness_y, kurtosis_x, kurtosis_y, skewness_2d, kurtosis_2d]
    metrics_names = {'Skewness_X', 'Skewness_Y', 'Kurtosis_X', 'Kurtosis_Y', '2D_Skewness', '2D_Kurtosis'};
    p_values = zeros(1, 6);
    stats = zeros(1, 6);
    ci = zeros(2, 6);
    effect_sizes = zeros(1, 6);
    
    % Group averages
    mean_g1 = [mean(group1_skewness_x), mean(group1_skewness_y), mean(group1_kurtosis_x), ...
        mean(group1_kurtosis_y), mean(group1_skewness_2d), mean(group1_kurtosis_2d)];
    
    mean_g2 = [mean(group2_skewness_x), mean(group2_skewness_y), mean(group2_kurtosis_x), ...
        mean(group2_kurtosis_y), mean(group2_skewness_2d), mean(group2_kurtosis_2d)];
    
    % Standard deviations
    std_g1 = [std(group1_skewness_x), std(group1_skewness_y), std(group1_kurtosis_x), ...
        std(group1_kurtosis_y), std(group1_skewness_2d), std(group1_kurtosis_2d)];
    
    std_g2 = [std(group2_skewness_x), std(group2_skewness_y), std(group2_kurtosis_x), ...
        std(group2_kurtosis_y), std(group2_skewness_2d), std(group2_kurtosis_2d)];
    
    % Perform t-tests for each metric
    % Compare skewness X
    [h, p, ci_temp, stats_temp] = ttest2(group1_skewness_x, group2_skewness_x, 'Alpha', alpha);
    p_values(1) = p;
    stats(1) = stats_temp.tstat;
    ci(:,1) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_skewness_x) + (n_subj_g2-1)*var(group2_skewness_x))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(1) = (mean(group1_skewness_x) - mean(group2_skewness_x)) / pooled_std;
    
    % Compare skewness Y
    [h, p, ci_temp, stats_temp] = ttest2(group1_skewness_y, group2_skewness_y, 'Alpha', alpha);
    p_values(2) = p;
    stats(2) = stats_temp.tstat;
    ci(:,2) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_skewness_y) + (n_subj_g2-1)*var(group2_skewness_y))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(2) = (mean(group1_skewness_y) - mean(group2_skewness_y)) / pooled_std;
    
    % Compare kurtosis X
    [h, p, ci_temp, stats_temp] = ttest2(group1_kurtosis_x, group2_kurtosis_x, 'Alpha', alpha);
    p_values(3) = p;
    stats(3) = stats_temp.tstat;
    ci(:,3) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_kurtosis_x) + (n_subj_g2-1)*var(group2_kurtosis_x))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(3) = (mean(group1_kurtosis_x) - mean(group2_kurtosis_x)) / pooled_std;
    
    % Compare kurtosis Y
    [h, p, ci_temp, stats_temp] = ttest2(group1_kurtosis_y, group2_kurtosis_y, 'Alpha', alpha);
    p_values(4) = p;
    stats(4) = stats_temp.tstat;
    ci(:,4) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_kurtosis_y) + (n_subj_g2-1)*var(group2_kurtosis_y))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(4) = (mean(group1_kurtosis_y) - mean(group2_kurtosis_y)) / pooled_std;
    
    % Compare 2D skewness
    [h, p, ci_temp, stats_temp] = ttest2(group1_skewness_2d, group2_skewness_2d, 'Alpha', alpha);
    p_values(5) = p;
    stats(5) = stats_temp.tstat;
    ci(:,5) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_skewness_2d) + (n_subj_g2-1)*var(group2_skewness_2d))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(5) = (mean(group1_skewness_2d) - mean(group2_skewness_2d)) / pooled_std;
    
    % Compare 2D kurtosis
    [h, p, ci_temp, stats_temp] = ttest2(group1_kurtosis_2d, group2_kurtosis_2d, 'Alpha', alpha);
    p_values(6) = p;
    stats(6) = stats_temp.tstat;
    ci(:,6) = ci_temp;
    
    % Cohen's d effect size
    pooled_std = sqrt(((n_subj_g1-1)*var(group1_kurtosis_2d) + (n_subj_g2-1)*var(group2_kurtosis_2d))/(n_subj_g1+n_subj_g2-2));
    effect_sizes(6) = (mean(group1_kurtosis_2d) - mean(group2_kurtosis_2d)) / pooled_std;
    
    % Multiple comparison correction (Bonferroni)
    p_values_corrected = min(p_values * size(phases,2), 1); % Multiply by number of tests (4)
    
    % Display statistical results
    fprintf('\nStatistical Results:\n');
    fprintf('  Metric           | Group 1 Mean | Group 2 Mean | p-value | Corrected p-value | Effect Size (d)\n');
    fprintf('  ---------------------------------------------------------------------------------------\n');
    for i = 1:length(metrics_names)
        fprintf('  %-15s | %11.4f | %11.4f | %7.4f | %17.4f | %13.4f\n', ...
            metrics_names{i}, mean_g1(i), mean_g2(i), p_values(i), p_values_corrected(i), effect_sizes(i));
    end
    
    %% Visualization of results
    fprintf('Creating visualizations with statistical test p-values for %s phase...\n', current_phase);
    
    % 1. Average fixation maps for each group with overlaid distribution characteristics and p-values
    fig1 = figure('Position', [100, 100, 1200, 500], 'Visible', 'on');
    
    % Group 1 Average Map
    subplot(1,2,1);
    mean_group1 = squeeze(mean(group1_fixations, 1))';
    imagesc(mean_group1);
    title(sprintf('Group 1 Average %s Map', current_phase), 'FontSize', 14);
    colormap(hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Calculate average center of mass for Group 1
    [y_grid, x_grid] = meshgrid(1:Resolution_Y, 1:Resolution_X);
    normalized_mean_g1 = mean_group1 / sum(mean_group1(:));
    mean_x_g1 = sum(sum(normalized_mean_g1 .* x_grid));
    mean_y_g1 = sum(sum(normalized_mean_g1 .* y_grid));
    
    % Plot center of mass
    plot(mean_y_g1, mean_x_g1, 'w+', 'LineWidth', 2, 'MarkerSize', 12);
    
    % Add skewness arrows (direction and magnitude)
    arrow_scale = 20; % Scale factor for arrow length
    quiver(mean_y_g1, mean_x_g1, mean_g1(2)*arrow_scale, mean_g1(1)*arrow_scale, 0, 'w', 'LineWidth', 2);
    
    % Add text for distribution metrics
    text(10, 10, sprintf('Skew X: %.2f', mean_g1(1)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 30, sprintf('Skew Y: %.2f', mean_g1(2)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 50, sprintf('Kurt X: %.2f', mean_g1(3)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 70, sprintf('Kurt Y: %.2f', mean_g1(4)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 90, sprintf('2D Skew: %.2f', mean_g1(5)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 110, sprintf('2D Kurt: %.2f', mean_g1(6)), 'Color', 'w', 'FontWeight', 'bold');
    
    hold off;
    
    % Group 2 Average Map
    subplot(1,2,2);
    mean_group2 = squeeze(mean(group2_fixations, 1))';
    imagesc(mean_group2);
    title(sprintf('Group 2 Average %s Map', current_phase), 'FontSize', 14);
    colormap(hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Calculate average center of mass for Group 2
    normalized_mean_g2 = mean_group2 / sum(mean_group2(:));
    mean_x_g2 = sum(sum(normalized_mean_g2 .* x_grid));
    mean_y_g2 = sum(sum(normalized_mean_g2 .* y_grid));
    
    % Plot center of mass
    plot(mean_y_g2, mean_x_g2, 'w+', 'LineWidth', 2, 'MarkerSize', 12);
    
    % Add skewness arrows (direction and magnitude)
    quiver(mean_y_g2, mean_x_g2, mean_g2(2)*arrow_scale, mean_g2(1)*arrow_scale, 0, 'w', 'LineWidth', 2);
    
    % Add text for distribution metrics
    text(10, 10, sprintf('Skew X: %.2f', mean_g2(1)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 30, sprintf('Skew Y: %.2f', mean_g2(2)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 50, sprintf('Kurt X: %.2f', mean_g2(3)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 70, sprintf('Kurt Y: %.2f', mean_g2(4)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 90, sprintf('2D Skew: %.2f', mean_g2(5)), 'Color', 'w', 'FontWeight', 'bold');
    text(10, 110, sprintf('2D Kurt: %.2f', mean_g2(6)), 'Color', 'w', 'FontWeight', 'bold');
    
    % Add p-value information in the figure
    annotation('textbox', [0.3, 0.95, 0.4, 0.05], 'String', ...
        sprintf('Statistical Comparison (p-values): Skew X: %.3f, Skew Y: %.3f, Kurt X: %.3f, Kurt Y: %.3f, 2D Skew: %.3f, 2D Kurt: %.3f', ...
        p_values_corrected(1), p_values_corrected(2), p_values_corrected(3), ...
        p_values_corrected(4), p_values_corrected(5), p_values_corrected(6)), ...
        'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'FitBoxToText', 'on');
    
    
    hold off;
    
    % Save figure
    print(fig1, fullfile(results_folder, sprintf('Group_Average_Maps_Metrics_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig1, fullfile(results_folder, sprintf('Group_Average_Maps_Metrics_%s.fig', current_phase)));
    
    % 2. Create boxplots for comparison of metrics between groups
    fig2 = figure('Position', [100, 100, 1200, 800], 'Visible', 'on');
    
    % Arrange subplots in a 3x2 grid
    for i = 1:6
        subplot(3, 2, i);
        
        % Prepare data for boxplot
        if strcmp(metrics_names{i}, '2D_Skewness')
            group_data = [double(group1_skewness_2d); double(group2_skewness_2d)];
        elseif strcmp(metrics_names{i}, '2D_Kurtosis')
            group_data = [double(group1_kurtosis_2d); double(group2_kurtosis_2d)];
        elseif strcmp(metrics_names{i}, 'Skewness_X')
            group_data = [double(group1_skewness_x); double(group2_skewness_x)];
        elseif strcmp(metrics_names{i}, 'Skewness_Y')
            group_data = [double(group1_skewness_y); double(group2_skewness_y)];
        elseif strcmp(metrics_names{i}, 'Kurtosis_X')
            group_data = [double(group1_kurtosis_x); double(group2_kurtosis_x)];
        elseif strcmp(metrics_names{i}, 'Kurtosis_Y')
            group_data = [double(group1_kurtosis_y); double(group2_kurtosis_y)];
        end
        group_labels = [ones(n_subj_g1, 1); 2*ones(n_subj_g2, 1)];
        
        % Create boxplot
        boxplot(group_data, group_labels, 'Labels', {'Group 1', 'Group 2'});
        title(strrep(metrics_names{i}, '_', ' '), 'FontSize', 14);
        ylabel('Value');
        grid on;
        
        % Add p-value annotation
        % Always show p-value, regardless of significance
        % Determine y position for text (above the boxes)
        y_max = max(max(group_data)) * 1.1;
        
        % Add significance stars
        stars = '';
        if p_values_corrected(i) < 0.05
            stars = '*';
        end
        if p_values_corrected(i) < 0.01
            stars = '**';
        end
        if p_values_corrected(i) < 0.001
            stars = '***';
        end
        
        % Add p-value text
        if p_values_corrected(i) < 0.001
            text_str = 'p < 0.001';
        else
            text_str = sprintf('p = %.3f', p_values_corrected(i));
        end
        
        text(1.5, y_max, [text_str ' ' stars], 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    % Adjust layout
    annotation('textbox', [0.3, 0.95, 0.4, 0.05], 'String', ...
        sprintf('Distribution Metrics Comparison - %s Phase', current_phase), ...
        'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'FitBoxToText', 'on');
    
    % Save figure
    print(fig2, fullfile(results_folder, sprintf('Metrics_Boxplots_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig2, fullfile(results_folder, sprintf('Metrics_Boxplots_%s.fig', current_phase)));
    
    %% Save numerical results
    fprintf('Saving numerical results for %s phase...\n', current_phase);
    
    % Save results structure
    results = struct();
    results.parameters = struct('center_x', center_x, 'center_y', center_y, ...
        'Resolution_X', Resolution_X, 'Resolution_Y', Resolution_Y, ...
        'sigma', sigma, 'precision', precision_type, 'alpha', alpha);
    
    % Group 1 metrics
    results.group1_skewness_x = group1_skewness_x;
    results.group1_skewness_y = group1_skewness_y;
    results.group1_kurtosis_x = group1_kurtosis_x;
    results.group1_kurtosis_y = group1_kurtosis_y;
    results.group1_skewness_2d = group1_skewness_2d;
    results.group1_kurtosis_2d = group1_kurtosis_2d;
    
    % Group 2 metrics
    results.group2_skewness_x = group2_skewness_x;
    results.group2_skewness_y = group2_skewness_y;
    results.group2_kurtosis_x = group2_kurtosis_x;
    results.group2_kurtosis_y = group2_kurtosis_y;
    results.group2_skewness_2d = group2_skewness_2d;
    results.group2_kurtosis_2d = group2_kurtosis_2d;
    
    % Group means and standard deviations
    results.mean_g1 = mean_g1;
    results.mean_g2 = mean_g2;
    results.std_g1 = std_g1;
    results.std_g2 = std_g2;
    
    % Statistical test results
    results.p_values = p_values;
    results.p_values_corrected = p_values_corrected;
    results.effect_sizes = effect_sizes;
    results.ci = ci;
    results.stats = stats;
    
    % Average fixation maps
    results.mean_group1_map = mean_group1;
    results.mean_group2_map = mean_group2;
    
    % Sample sizes
    results.n_subj_g1 = n_subj_g1;
    results.n_subj_g2 = n_subj_g2;
    results.phase = current_phase;
    
    % Save as MAT format for further analysis
    save(fullfile(results_folder, sprintf('SkewnessKurtosis_Results_%s.mat', current_phase)), 'results');
    
    % Save key numerical results to a text file
    fid = fopen(fullfile(results_folder, sprintf('SkewnessKurtosis_Summary_%s.txt', current_phase)), 'w');
    fprintf(fid, 'Skewness and Kurtosis Analysis Summary - %s Phase\n', current_phase);
    fprintf(fid, '==================================================\n\n');
    fprintf(fid, 'Parameters:\n');
    fprintf(fid, '  Center X coordinate = %d\n', center_x);
    fprintf(fid, '  Center Y coordinate = %d\n', center_y);
    fprintf(fid, '  Gaussian smoothing sigma = %.1f\n', sigma);
    fprintf(fid, '  Data precision = %s\n', precision_type);
    fprintf(fid, '  Significance level (alpha) = %.2f\n', alpha);
    
    fprintf(fid, '\nData Information:\n');
    fprintf(fid, '  Group 1: %d subjects\n', n_subj_g1);
    fprintf(fid, '  Group 2: %d subjects\n', n_subj_g2);
    fprintf(fid, '  Resolution: %d x %d pixels\n\n', Resolution_X, Resolution_Y);
    
    fprintf(fid, 'Results:\n');
    fprintf(fid, '  Metric            | Group 1 Mean (SD)      | Group 2 Mean (SD)      | p-value | Corrected p-value | Effect Size (d)\n');
    fprintf(fid, '  --------------------------------------------------------------------------------------------------------\n');
    for i = 1:length(metrics_names)
        fprintf(fid, '  %-17s | %8.4f (%8.4f) | %8.4f (%8.4f) | %7.4f | %17.4f | %13.4f\n', ...
            metrics_names{i}, mean_g1(i), std_g1(i), mean_g2(i), std_g2(i), ...
            p_values(i), p_values_corrected(i), effect_sizes(i));
    end
    
    % Interpretation guidelines
    fprintf(fid, '\nInterpretation of Results:\n');
    fprintf(fid, '  Skewness: Measures asymmetry of distribution\n');
    fprintf(fid, '    - Positive skewness indicates tail on right side\n');
    fprintf(fid, '    - Negative skewness indicates tail on left side\n');
    fprintf(fid, '    - Zero skewness indicates symmetry\n');
    fprintf(fid, '  Kurtosis: Measures peakedness of distribution\n');
    fprintf(fid, '    - Positive kurtosis indicates heavy tails and peaked center (leptokurtic)\n');
    fprintf(fid, '    - Negative kurtosis indicates light tails and flat center (platykurtic)\n');
    fprintf(fid, '    - Zero kurtosis indicates normal distribution (mesokurtic)\n');
    fprintf(fid, '  2D Measures: Combined skewness/kurtosis in both X and Y dimensions\n');
    
    % Add performance information (if timing is enabled)
    if enable_timing
        phase_time = toc(phase_start_time);
        fprintf(fid, '\nPerformance Information:\n');
        fprintf(fid, '  Total processing time for %s phase: %.2f seconds\n', current_phase, phase_time);
    end
    
    fclose(fid);
    
    fprintf('%s phase analysis complete! Results saved to %s\n', current_phase, results_folder);
    
    % Report timing for current phase
    if enable_timing
        phase_time = toc(phase_start_time);
        fprintf('Total time for %s phase: %.2f seconds\n', current_phase, phase_time);
    end
    
    % Clear phase-specific variables to free memory
    if clear_temp_data
        clear group1_fixations group2_fixations
        clear mean_group1 mean_group2
    end
end % End of phase loop

% Report total timing across all phases
if enable_timing
    total_time = toc(total_start_time);
    fprintf('\nTotal processing time across all phases: %.2f seconds (%.2f minutes)\n', ...
        total_time, total_time/60);
end

%% Combine results from all phases
fprintf('\nCompiling results from all phases...\n');

% Store results from each phase
all_phase_results = struct();

for phase_idx = 1:length(phases)
    current_phase = phases{phase_idx};
    results_folder = fullfile(results_base_folder, current_phase);
    result_file = fullfile(results_folder, sprintf('SkewnessKurtosis_Results_%s.mat', current_phase));
    
    if exist(result_file, 'file')
        % Load results for this phase
        phase_data = load(result_file);
        % Store in all_phase_results structure
        all_phase_results.(current_phase) = phase_data.results;
    end
end

% Save combined results to one file
save(fullfile(results_base_folder, 'All_Phase_Results.mat'), 'all_phase_results');
fprintf('All results saved to %s\n', fullfile(results_base_folder, 'All_Phase_Results.mat'));

% Create a summary comparison across all phases
if length(phases) > 1
    fprintf('Creating cross-phase comparison...\n');
    
    % Set up figure for cross-phase comparison
    fig_cross = figure('Position', [100, 100, 1500, 800], 'Visible', 'on');
    
    % For each metric, create a subplot showing values across phases
    for i = 1:6
        subplot(2, 3, i);
        
        % Initialize data arrays
        phase_means_g1 = zeros(1, length(phases));
        phase_means_g2 = zeros(1, length(phases));
        phase_errors_g1 = zeros(1, length(phases));
        phase_errors_g2 = zeros(1, length(phases));
        phase_p_values = zeros(1, length(phases));
        
        % Collect data for each phase
        for p = 1:length(phases)
            if isfield(all_phase_results, phases{p})
                phase_data = all_phase_results.(phases{p});
                phase_means_g1(p) = phase_data.mean_g1(i);
                phase_means_g2(p) = phase_data.mean_g2(i);
                phase_errors_g1(p) = phase_data.std_g1(i) / sqrt(phase_data.n_subj_g1);
                phase_errors_g2(p) = phase_data.std_g2(i) / sqrt(phase_data.n_subj_g2);
                phase_p_values(p) = phase_data.p_values_corrected(i);
            end
        end
        
        % Create grouped bar chart
        bar_data = [phase_means_g1; phase_means_g2]';
        b = bar(bar_data);
        
        % Set colors for groups
        b(1).FaceColor = [0.3 0.5 0.7];  % Blue for Group 1
        b(2).FaceColor = [0.8 0.3 0.3];  % Red for Group 2
        
        % Add error bars
        hold on;
        
        % Get x positions for error bars
        ngroups = length(phases);
        nbars = 2;
        x = zeros(ngroups, nbars);
        groupwidth = min(0.8, nbars/(nbars+1.5));
        
        for j = 1:nbars
            % Calculate the center position of each bar
            x(:,j) = (1:ngroups) - groupwidth/2 + (2*j-1) * groupwidth / (2*nbars);
        end
        
        % Plot error bars
        errorbar(x(:,1), phase_means_g1, phase_errors_g1, 'k', 'linestyle', 'none', 'LineWidth', 1);
        errorbar(x(:,2), phase_means_g2, phase_errors_g2, 'k', 'linestyle', 'none', 'LineWidth', 1);
        
        % Add significance markers
        for p = 1:length(phases)
            if phase_p_values(p) < 0.05
                x_pos = mean([x(p,1), x(p,2)]);
                y_pos = max([phase_means_g1(p) + phase_errors_g1(p), phase_means_g2(p) + phase_errors_g2(p)]) * 1.1;
                
                % Significance stars
                stars = '';
                if phase_p_values(p) < 0.05
                    stars = '*';
                end
                if phase_p_values(p) < 0.01
                    stars = '**';
                end
                if phase_p_values(p) < 0.001
                    stars = '***';
                end
                
                text(x_pos, y_pos, stars, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 14);
            end
        end
        
        % Add title and labels
        title(strrep(metrics_names{i}, '_', ' '), 'FontSize', 14);
        ylabel('Value');
        set(gca, 'XTickLabel', phases);
        xtickangle(45);
        grid on;
        
        hold off;
    end
    
    % Add legend
    legend({'Group 1', 'Group 2'}, 'Location', 'best');
    
    % Adjust layout
    annotation('textbox', [0.3, 0.95, 0.4, 0.05], 'String', ...
        'Cross-Phase Comparison of Distribution Metrics', ...
        'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'FitBoxToText', 'on');
    
    % Save figure
    print(fig_cross, fullfile(results_base_folder, 'Cross_Phase_Comparison.png'), '-dpng', '-r300');
    saveas(fig_cross, fullfile(results_base_folder, 'Cross_Phase_Comparison.fig'));
    
    % Create a summary text file
    fid = fopen(fullfile(results_base_folder, 'Cross_Phase_Summary.txt'), 'w');
    fprintf(fid, 'Cross-Phase Comparison of Skewness and Kurtosis Analysis\n');
    fprintf(fid, '======================================================\n\n');
    
    % Create a table showing significant differences across phases
    fprintf(fid, 'Significant Differences (p < 0.05 after correction):\n\n');
    
    fprintf(fid, '  Phase      | Significant Metrics\n');
    fprintf(fid, '  -------------------------------------\n');
    
    for p = 1:length(phases)
        if isfield(all_phase_results, phases{p})
            phase_data = all_phase_results.(phases{p});
            sig_metrics = metrics_names(phase_data.p_values_corrected < 0.05);
            
            if ~isempty(sig_metrics)
                sig_text = sprintf('%s, ', sig_metrics{:});
                sig_text = sig_text(1:end-2); % Remove trailing comma
            else
                sig_text = 'None';
            end
            
            fprintf(fid, '  %-10s | %s\n', phases{p}, sig_text);
        end
    end
    
    % Tabulate all results
    fprintf(fid, '\n\nComplete Results Table:\n\n');
    fprintf(fid, '  Metric          | Phase      | Group 1 Mean (SD)      | Group 2 Mean (SD)      | p-value | Effect Size\n');
    fprintf(fid, '  ---------------------------------------------------------------------------------------------------\n');
    
    for i = 1:length(metrics_names)
        for p = 1:length(phases)
            if isfield(all_phase_results, phases{p})
                phase_data = all_phase_results.(phases{p});
                fprintf(fid, '  %-15s | %-10s | %8.4f (%8.4f) | %8.4f (%8.4f) | %7.4f | %10.4f\n', ...
                    metrics_names{i}, phases{p}, ...
                    phase_data.mean_g1(i), phase_data.std_g1(i), ...
                    phase_data.mean_g2(i), phase_data.std_g2(i), ...
                    phase_data.p_values_corrected(i), phase_data.effect_sizes(i));
            end
        end
        fprintf(fid, '  ---------------------------------------------------------------------------------------------------\n');
    end
    
    fclose(fid);
end

fprintf('\nAnalysis completed successfully!\n');
fprintf('To access results for a specific phase: all_phase_results.phase_name\n');
fprintf('Example: all_phase_results.DecisionL\n');

% Clear variables if needed
if clear_temp_data
    clearvars -except all_phase_results
end

function [b1_2d, b2_2d] = calculate_mardia_2d_measures(data_map)
% Calculate Mardia's 2D skewness (b1,2) and kurtosis (b2,2) measures (fast version)
%
% Inputs:
%   data_map - 2D normalized probability distribution (sum = 1)
%
% Outputs:
%   b1_2d - Mardia's 2D skewness measure
%   b2_2d - Mardia's 2D kurtosis measure

[height, width] = size(data_map);

% Coordinate grids
[y_grid, x_grid] = meshgrid(1:width, 1:height);

% Mean
mean_x = sum(sum(data_map .* x_grid));
mean_y = sum(sum(data_map .* y_grid));

% Centered coordinates
centered_x = x_grid - mean_x;
centered_y = y_grid - mean_y;

% Covariance matrix
var_x = sum(sum(data_map .* centered_x.^2));
var_y = sum(sum(data_map .* centered_y.^2));
cov_xy = sum(sum(data_map .* centered_x .* centered_y));
cov_matrix = [var_x, cov_xy; cov_xy, var_y];

% Regularization if needed
if abs(det(cov_matrix)) < 1e-10
    warning('Covariance matrix nearly singular. Adding regularization.');
    cov_matrix = cov_matrix + 1e-10 * eye(2);
end

inv_cov = inv(cov_matrix);

% Flatten and pick nonzero data
data_flat = data_map(:);
x_flat = x_grid(:);
y_flat = y_grid(:);
nonzero_idx = data_flat > 0;
data_flat = data_flat(nonzero_idx);
x_flat = x_flat(nonzero_idx);
y_flat = y_flat(nonzero_idx);

% Centered coordinates matrix
X = [x_flat - mean_x, y_flat - mean_y];

% Precompute Mahalanobis squared distances
mahal_dist_sq = sum((X * inv_cov) .* X, 2);

% --- Fast computation ---

% Skewness (b1,2)
weighted_X = X .* sqrt(data_flat);  % (nÅ~2)
cross_prod = weighted_X * inv_cov * weighted_X';  % (nÅ~n) matrix
b1_2d = sum(cross_prod(:).^3);

% Kurtosis (b2,2)
b2_2d = sum(data_flat .* mahal_dist_sq.^2);

% Adjust kurtosis for 2D normal
b2_2d = b2_2d - 8;

% Guard small negatives
if b1_2d < 0 && abs(b1_2d) < 1e-10
    b1_2d = 0;
end
end
