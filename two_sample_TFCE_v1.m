%% Two-Sample Threshold-Free Cluster Enhancement Analysis
% This script implements Threshold-Free Cluster Enhancement (TFCE) for
% 2D eye-tracking fixation data comparing two groups using CoSMoMVPA.
%
% REQUIRES: Output files from detect_bilateral_peaks_and_extract_v1.m
%
% Author: Y. Shigemune
% Created: 5/10/2025
% Last Modified: 5/10/2025
% Version: 1.0.0
%
% Description:
%   This script performs cluster-based permutation testing using TFCE
%   to identify spatially contiguous regions showing significant
%   differences in gaze patterns between two groups.
%
% Prerequisites:
%   - MATLAB R2017b or later
%   - CoSMoMVPA toolbox (must be in MATLAB path)
%   - Image Processing Toolbox (for imresize, optional)
%   - External functions: readfromexcel, imgaussfilt (optional)
%   - Completed bilateral peak detection with detect_bilateral_peaks_and_extract_v1.m
%
% Input files:
%   - PixelMap Excel files in Output_Group1/ and Output_Group2/ directories
%   - Format: Output from detect_bilateral_peaks_and_extract_v1.m
%   - Structure: 201x201 pixel regions with phase-specific data
%
% Output files:
%   - TFCE_Results/[Phase]/ directories containing:
%     - Statistical maps (Z-scores, p-values)
%     - Significant cluster visualizations
%     - Detailed cluster statistics (size, peak values, locations)
%     - Summary figures and numerical results
%
% Key Parameters:
%   - E: TFCE extent parameter (default: 0.5)
%   - H: TFCE height parameter (default: 2.0)
%   - niter: Number of permutations (default: 1000)
%   - connect: Pixel connectivity (8=diagonal, 4=cardinal, 0=none)
%   - alpha: Significance level (default: 0.05)
%   - down_factor: Downsampling factor for computational efficiency
%
% Analysis Methods:
%   - CoSMoMVPA permutation testing framework
%   - TFCE transformation for cluster identification
%   - Family-wise error rate control
%   - Cluster characterization (size, peak, centroid)
%
% Usage:
%   1. Install and configure CoSMoMVPA toolbox
%   2. Complete detect_bilateral_peaks_and_extract_v1.m
%   3. Place output files in Output_Group1/ and Output_Group2/
%   4. Update addpath() line for your CoSMoMVPA installation
%   5. Run script: two_sample_TFCE_v1
%   6. Check results in TFCE_Results/ directory
%
% References:
%   Part of TFCE Moments Eyetracking Toolbox
%   Requires peak area data from detect_bilateral_peaks_and_extract_v1.m
%   Based on: Smith, S. M., & Nichols, T. E. (2009). NeuroImage.

%% Parameters - Edit these as needed
% Analysis parameters
E = 0.5;                  % TFCE extent parameter (default from brain imaging)
H = 2;                    % TFCE height parameter (default from brain imaging)
niter = 1000;             % Number of permutations for statistical testing
connect = 8;              % Neighborhood connectivity: 8=include diagonals, 4=only cardinal directions, 0=no connectivity
alpha = 0.05;             % Significance level for statistical testing
down_factor = 1;          % Downsampling factor (1=no downsampling, 4=quarter resolution)
sigma = 10;               % Gaussian smoothing sigma (pixels) - applied before downsampling

% Memory optimization settings
use_single_precision = false; % Use single precision to save memory
clear_temp_data = false;     % Clear temporary data to free memory

% Performance measurement
enable_timing = true;     % Enable timing measurements for performance analysis

% Display settings
Resolution_X = 201;      % Width of the display in pixels
Resolution_Y = 201;       % Height of the display in pixels

% Paths - Edit these to match your folder structure
curDir = pwd;
group1_folder = strcat(curDir, filesep, 'Output_Group1');  % Folder containing Group 1 PixelMap excel files
group2_folder = strcat(curDir, filesep, 'Output_Group2');  % Folder containing Group 2 PixelMap excel files
results_base_folder = strcat(curDir, filesep, 'TFCE_Results'); % Base folder for saving results

% Phases to analyze (column indices in Excel files)
phases = {'DecisionL', 'DecisionR', 'FeedbackL' 'FeedbackR'};
phase_cols = [3, 4, 5, 6]; % Excel column indices for phases (Decision, Feedback, Fixation)

% Create base results folder if it doesn't exist
if ~exist(results_base_folder, 'dir')
    mkdir(results_base_folder);
end

% Start a timer for overall performance measurement
if enable_timing
    total_start_time = tic;
end

% Set precision type based on parameter
if use_single_precision
    precision_type = 'single';
else
    precision_type = 'double';
end

% add path for CoSMoMVPA
addpath(genpath('F:\Software\CoSMoMVPA'));

% Process each phase separately
for phase_idx = 1:length(phases)
    current_phase = phases{phase_idx};
    current_phase_col = phase_cols(phase_idx);
    
    fprintf('\n\n========== Processing %s Phase ==========\n\n', current_phase);
    
    % Start a timer for phase performance measurement
    if enable_timing
        phase_start_time = tic;
    end
    
    % Create phase-specific results folder
    results_folder = fullfile(results_base_folder, current_phase);
    if ~exist(results_folder, 'dir')
        mkdir(results_folder);
    end
    
    %% Load data from Excel files (PixelMap format)
    fprintf('Loading data from PixelMap Excel files for %s phase...\n', current_phase);
    
    % Load Group 1 data
    group1_files = dir(fullfile(group1_folder, '*_PixelMap.xlsx'));
    n_subj_g1 = length(group1_files);
    fprintf('Found %d subjects in Group 1\n', n_subj_g1);
    
    % Initialize data structure for Group 1 (subjects x X x Y)
    group1_fixations = zeros(n_subj_g1, Resolution_X, Resolution_Y, precision_type);
    
    % Process each file in Group 1
    for i = 1:n_subj_g1
        % Read Excel file
        file_path = fullfile(group1_folder, group1_files(i).name);
        fprintf('Processing Group 1, subject %d: %s\n', i, group1_files(i).name);
        
        % Try to read using readfromexcel
        try
            % Read all data using readfromexcel
            data = readfromexcel(file_path, 'sheet', 'Sheet1', 'All');
            header = data(1,:);
            data = data(2:end,:);
            
            % Extract data
            x_coords = cell2mat(data(:, 1));
            y_coords = cell2mat(data(:, 2));
            
            % Select the appropriate phase data
            fixation_data = cell2mat(data(:, current_phase_col));
            
        catch
            warning('Could not read file %s. Skipping.', file_path);
            continue;
        end
        
        % Create fixation map - use sparse matrix for efficiency
        fixation_map = sparse(x_coords, y_coords, fixation_data, Resolution_X, Resolution_Y);
        
        % Convert to full matrix for smoothing
        fixation_map = full(fixation_map);
        
        % Apply Gaussian smoothing with sigma from parameters section
        % Use MATLAB's built-in function for speed if available
        if exist('imgaussfilt', 'file')
            fixation_map = imgaussfilt(fixation_map, sigma);
        else
            % Fall back to custom Gaussian filter if imgaussfilt is not available
            filter_size = ceil(sigma * 3) * 2 + 1;
            if mod(filter_size, 2) == 0
                filter_size = filter_size + 1;
            end
            [x, y] = meshgrid(-(filter_size-1)/2:(filter_size-1)/2, -(filter_size-1)/2:(filter_size-1)/2);
            gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
            fixation_map = conv2(double(fixation_map), gaussian_kernel, 'same');
        end
        
        % Store in 3D array with the specified precision
        group1_fixations(i, :, :) = cast(fixation_map, precision_type);
        
        % Display basic statistics
        total_fixation = sum(fixation_map(:));
        fprintf('  Total %s value: %f\n', current_phase, total_fixation);
    end
    
    % Load Group 2 data (similar process)
    group2_files = dir(fullfile(group2_folder, '*_PixelMap.xlsx'));
    n_subj_g2 = length(group2_files);
    fprintf('Found %d subjects in Group 2\n', n_subj_g2);
    
    % Initialize data structure for Group 2
    group2_fixations = zeros(n_subj_g2, Resolution_X, Resolution_Y, precision_type);
    
    % Process each file in Group 2
    for i = 1:n_subj_g2
        % Read Excel file
        file_path = fullfile(group2_folder, group2_files(i).name);
        fprintf('Processing Group 2, subject %d: %s\n', i, group2_files(i).name);
        
        % Try to read using readfromexcel
        try
            % Read all data using readfromexcel
            data = readfromexcel(file_path, 'sheet', 'Sheet1', 'All');
            header = data(1,:);
            data = data(2:end,:);
            
            % Extract data
            x_coords = cell2mat(data(:, 1));
            y_coords = cell2mat(data(:, 2));
            
            % Select the appropriate phase data
            fixation_data = cell2mat(data(:, current_phase_col));
            
        catch
            warning('Could not read file %s. Skipping.', file_path);
            continue;
        end
        
        % Create fixation map - use sparse matrix for efficiency
        fixation_map = sparse(x_coords, y_coords, fixation_data, Resolution_X, Resolution_Y);
        
        % Convert to full matrix for smoothing
        fixation_map = full(fixation_map);
        
        % Apply Gaussian smoothing with sigma from parameters section
        if exist('imgaussfilt', 'file')
            fixation_map = imgaussfilt(fixation_map, sigma);
        else
            % Fall back to custom Gaussian filter if imgaussfilt is not available
            filter_size = ceil(sigma * 3) * 2 + 1;
            if mod(filter_size, 2) == 0
                filter_size = filter_size + 1;
            end
            [x, y] = meshgrid(-(filter_size-1)/2:(filter_size-1)/2, -(filter_size-1)/2:(filter_size-1)/2);
            gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
            fixation_map = conv2(double(fixation_map), gaussian_kernel, 'same');
        end
        
        % Store in 3D array with the specified precision
        group2_fixations(i, :, :) = cast(fixation_map, precision_type);
        
        % Display basic statistics
        total_fixation = sum(fixation_map(:));
        fprintf('  Total %s value: %f\n', current_phase, total_fixation);
    end
    
    %% Downsample data for computational efficiency
    fprintf('Downsampling data for computational efficiency (factor = %d)...\n', down_factor);
    
    % Adjusted dimensions after downsampling
    ds_x = ceil(Resolution_X / down_factor);
    ds_y = ceil(Resolution_Y / down_factor);
    
    % If down_factor is 1, skip downsampling
    if down_factor == 1
        fprintf('Skipping downsampling (down_factor = 1)...\n');
        ds_x = Resolution_X;
        ds_y = Resolution_Y;
        group1_fixations_ds = group1_fixations;
        group2_fixations_ds = group2_fixations;
    else
        % Downsample Group 1 data
        group1_fixations_ds = zeros(n_subj_g1, ds_x, ds_y, precision_type);
        for i = 1:n_subj_g1
            temp_map = squeeze(group1_fixations(i, :, :));
            group1_fixations_ds(i, :, :) = imresize(temp_map, [ds_x, ds_y], 'bicubic');
        end
        
        % Downsample Group 2 data
        group2_fixations_ds = zeros(n_subj_g2, ds_x, ds_y, precision_type);
        for i = 1:n_subj_g2
            temp_map = squeeze(group2_fixations(i, :, :));
            group2_fixations_ds(i, :, :) = imresize(temp_map, [ds_x, ds_y], 'bicubic');
        end
    end
    
    % Clear original high-resolution data to save memory if requested
    if clear_temp_data
        group1_fixations = [];
        group2_fixations = [];
    end
    
    %% Prepare data for CoSMoMVPA
    fprintf('Preparing data for CoSMoMVPA clustering analysis...\n');
    
    % Reshape data for CoSMoMVPA format (subjects x features)
    group1_flat = reshape(group1_fixations_ds, n_subj_g1, ds_x * ds_y);
    group2_flat = reshape(group2_fixations_ds, n_subj_g2, ds_x * ds_y);
    
    % Combine all subjects for dataset
    all_data = [group1_flat; group2_flat];
    
    % Create dataset structure properly formatted for CoSMoMVPA
    ds = struct();
    ds.samples = all_data;
    
    % Set sample attributes (sa)
    ds.sa.targets = [ones(n_subj_g1, 1); 2*ones(n_subj_g2, 1)]; % Group labels
    ds.sa.chunks = (1:(n_subj_g1 + n_subj_g2))';              % Subject IDs
    
    % Create feature dimension information (properly formatted for CoSMoMVPA)
    ds.a.fdim.labels = {'i', 'j'};
    ds.a.fdim.values = {1:ds_x, 1:ds_y};
    
    % Create proper feature coordinates
    [i_mat, j_mat] = ndgrid(1:ds_x, 1:ds_y);
    ds.fa.i = i_mat(:)';
    ds.fa.j = j_mat(:)';
    
    %% Create custom neighborhood structure
    fprintf('Creating custom neighborhood structure for 2D eye-tracking data...\n');
    
    % Determine connectivity type
    if connect == 0
        % No connectivity - each pixel is only neighbor to itself
        fprintf('Using no connectivity - each pixel is only neighbor to itself\n');
        conn_type = 'none';
        conn_offsets = [0 0]; % Self only
    elseif connect == 4
        % 4-connectivity - adjacent pixels (sharing an edge) are neighbors
        fprintf('Using 4-connectivity - pixels sharing an edge are neighbors\n');
        conn_type = '4-connected';
        conn_offsets = [-1 0; 0 -1; 0 1; 1 0];
    elseif connect == 8
        % 8-connectivity - adjacent pixels (sharing an edge or vertex) are neighbors
        fprintf('Using 8-connectivity - pixels sharing an edge or vertex are neighbors\n');
        conn_type = '8-connected';
        conn_offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    else
        error('connect must be 0, 4, or 8');
    end
    
    % Total number of pixels (features)
    n_features = ds_x * ds_y;
    
    % Initialize neighborhood structure
    cluster_nbrhood = struct();
    cluster_nbrhood.neighbors = cell(n_features, 1);
    
    % Calculate neighborhood relationships for each pixel
    for i = 1:ds_x
        for j = 1:ds_y
            % Current pixel index (linear index)
            % Calculate index in column-major order (MATLAB standard)
            pixel_idx = (j-1) * ds_x + i;
            
            % For no connectivity, set only self as neighbor
            if strcmp(conn_type, 'none')
                cluster_nbrhood.neighbors{pixel_idx} = pixel_idx;
                continue;
            end
            
            % Find neighboring pixels
            nbr_idxs = [];
            for k = 1:size(conn_offsets, 1)
                ni = i + conn_offsets(k, 1);
                nj = j + conn_offsets(k, 2);
                
                % Check image boundaries
                if ni >= 1 && ni <= ds_x && nj >= 1 && nj <= ds_y
                    nbr_idx = (nj-1) * ds_x + ni;
                    nbr_idxs = [nbr_idxs, nbr_idx];
                end
            end
            
            % Include self as a neighbor (always required)
            nbr_idxs = unique([pixel_idx, nbr_idxs]);
            
            % Store in neighborhood structure
            cluster_nbrhood.neighbors{pixel_idx} = nbr_idxs;
        end
    end
    
    % Set feature attributes
    cluster_nbrhood.fa = ds.fa;
    cluster_nbrhood.fa.sizes = ones(1, n_features);  % Set size of each pixel to 1
    cluster_nbrhood.a = ds.a;
    
    % Display simple statistics about the neighborhood structure
    neighbor_counts = cellfun(@numel, cluster_nbrhood.neighbors);
    fprintf('Neighborhood statistics:\n');
    fprintf('  Minimum neighbors per pixel: %d\n', min(neighbor_counts));
    fprintf('  Maximum neighbors per pixel: %d\n', max(neighbor_counts));
    fprintf('  Average neighbors per pixel: %.2f\n', mean(neighbor_counts));
    
    % For verification: show neighborhood relationships for the first few pixels
    fprintf('First few neighborhood relationships:\n');
    for i = 1:min(5, n_features)
        fprintf('  Pixel %d neighbors: %s\n', i, mat2str(cluster_nbrhood.neighbors{i}));
    end
    
    %% Run CoSMoMVPA's Monte Carlo cluster-based correction
    fprintf('Running CoSMoMVPA Monte Carlo cluster-based correction (%d iterations)...\n', niter);
    
    % Set options for TFCE analysis
    opt = struct();
    opt.cluster_stat = 'tfce';
    opt.niter = niter;
    opt.progress = 10;  % Show progress every 10 iterations
    opt.h = H;          % TFCE height parameter
    opt.e = E;          % TFCE extent parameter
    opt.dh = 0.1;       % TFCE delta-h parameter
    
    % Set seed for reproducibility
    opt.seed = 42;
    
    % Set seed for reproducibility if needed
    % opt.seed = 1;
    
    % Run the analysis
    if enable_timing
        tfce_start_time = tic;
    end
    
    % Use cosmo_montecarlo_cluster_stat to get z-scored results
    z_ds = cosmo_montecarlo_cluster_stat(ds, cluster_nbrhood, opt);
    
    if enable_timing
        tfce_time = toc(tfce_start_time);
        fprintf('TFCE computation completed in %.2f seconds.\n', tfce_time);
    end
    
    % Reshape z-scores to 2D map
    zmap = reshape(z_ds.samples, [ds_x, ds_y]);
    
    %% Create significance maps
    fprintf('Creating significance maps...\n');
    
    % Two-tailed significance
    alpha_two_tailed = alpha;
    z_threshold_two_tailed = norminv(1 - alpha_two_tailed/2);
    
    % Create binary significance maps
    significant_map_pos = zmap > z_threshold_two_tailed;  % Group 1 > Group 2
    significant_map_neg = zmap < -z_threshold_two_tailed; % Group 2 > Group 1
    significant_map_combined = significant_map_pos | significant_map_neg;
    
    % Calculate mean fixation maps for each group
    mean_group1 = squeeze(mean(group1_fixations_ds, 1));
    mean_group2 = squeeze(mean(group2_fixations_ds, 1));
    
    % Calculate difference maps in both directions
    diff_map_g2_g1 = mean_group2 - mean_group1; % Group2 - Group1
    diff_map_g1_g2 = mean_group1 - mean_group2; % Group1 - Group2
    
    % Transpose matrices for proper visualization
    mean_group1 = mean_group1';
    mean_group2 = mean_group2';
    zmap = zmap';
    significant_map_pos = significant_map_pos';
    significant_map_neg = significant_map_neg';
    significant_map_combined = significant_map_combined';
    diff_map_g2_g1 = diff_map_g2_g1';
    diff_map_g1_g2 = diff_map_g1_g2';
    
    % Create p-value maps
    pmap = 2*(1 - normcdf(abs(zmap))); % Two-tailed p-values
    
    %% Visualization and results output
    fprintf('Creating visualizations and saving results for %s phase...\n', current_phase);
    
    % Use jet colormap with green at center (0)
    cmap_div = jet(256);
    
    % 1. Mean fixation maps for each group
    fig1 = figure('Position', [100, 100, 1200, 500], 'Visible', 'on');
    
    % Find the maximum value across both groups for consistent scaling
    max_value = max([max(mean_group1(:)), max(mean_group2(:))]);
    
    subplot(1,2,1);
    imagesc(mean_group1, [0 max_value]);  % Set consistent color scale
    title(sprintf('Group 1 Mean %s Map', current_phase), 'FontSize', 14);
    colormap(hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    subplot(1,2,2);
    imagesc(mean_group2, [0 max_value]);  % Set consistent color scale
    title(sprintf('Group 2 Mean %s Map', current_phase), 'FontSize', 14);
    colormap(hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    % Ensure rendering is complete before saving
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig1, fullfile(results_folder, sprintf('Mean_%s_Maps.png', current_phase)), '-dpng', '-r300');
    saveas(fig1, fullfile(results_folder, sprintf('Mean_%s_Maps.fig', current_phase)));
    
    % 2. Z-statistics maps for both directions
    % Group 1 - Group 2 direction
    fig2a_clusters = figure('Position', [100, 100, 800, 700], 'Visible', 'on');
    max_abs_val = max(abs(zmap(:)));
    if max_abs_val == 0
        % If there are no non-zero values, set a small default
        max_abs_val = 0.001;
    end
    % Now use symmetric limits for the colormap
    imagesc(zmap, [-max_abs_val max_abs_val]);
    colormap(jet);
    colorbar;
    title(sprintf('Z-statistics Map for %s with Significant Clusters (Group 1 - Group 2)', current_phase), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Overlay G1 > G2 significant clusters on the Z-map
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    hold off;
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig2a_clusters, fullfile(results_folder, sprintf('Z_Statistics_Map_With_Clusters_%s_G1minusG2.png', current_phase)), '-dpng', '-r300');
    saveas(fig2a_clusters, fullfile(results_folder, sprintf('Z_Statistics_Map_With_Clusters_%s_G1minusG2.fig', current_phase)));
    
    % Modified Z-statistics map with significant clusters for Group 2 > Group 1
    fig2b_clusters = figure('Position', [100, 100, 800, 700], 'Visible', 'on');
    imagesc(-zmap, [-max_abs_val max_abs_val]);
    colormap(jet);
    colorbar;
    title(sprintf('Z-statistics Map for %s with Significant Clusters (Group 2 - Group 1)', current_phase), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Overlay G2 > G1 significant clusters on the Z-map
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    hold off;
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig2b_clusters, fullfile(results_folder, sprintf('Z_Statistics_Map_With_Clusters_%s_G2minusG1.png', current_phase)), '-dpng', '-r300');
    saveas(fig2b_clusters, fullfile(results_folder, sprintf('Z_Statistics_Map_With_Clusters_%s_G2minusG1.fig', current_phase)));
    
    % 3. Uncorrected vs TFCE-corrected significance
    fig4 = figure('Position', [100, 100, 1200, 500], 'Visible', 'on');
    
    subplot(1,2,1);
    imagesc(pmap < alpha);
    colormap(gca, gray);
    title(sprintf('Uncorrected Significant Areas for %s (p < %.2f)', current_phase, alpha), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    colorbar;
    
    subplot(1,2,2);
    imagesc(significant_map_combined);
    colormap(gca, gray);
    title(sprintf('TFCE-corrected Significant Areas for %s (p < %.2f)', current_phase, alpha), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    colorbar;
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig4, fullfile(results_folder, sprintf('Significance_Maps_Comparison_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig4, fullfile(results_folder, sprintf('Significance_Maps_Comparison_%s.fig', current_phase)));
    
    % 4a. Group1 > Group2 Difference map with significant clusters overlaid
    fig5a = figure('Position', [100, 100, 800, 700], 'Visible', 'on');
    max_abs_val = max(abs(diff_map_g1_g2(:)));
    imagesc(diff_map_g1_g2, [-max_abs_val max_abs_val]);
    colormap(jet);
    c = colorbar;
    c.Label.String = sprintf('Difference in %s proportion (G1 - G2)', current_phase);
    title(sprintf('Difference Map for %s (Group 1 - Group 2) with Significant Clusters', current_phase), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Create contours of significant areas without using bwboundaries
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    hold off;
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig5a, fullfile(results_folder, sprintf('Difference_Map_G1_G2_With_Clusters_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig5a, fullfile(results_folder, sprintf('Difference_Map_G1_G2_With_Clusters_%s.fig', current_phase)));
    
    % 4b. Group2 > Group1 Difference map with significant clusters overlaid
    fig5b = figure('Position', [100, 100, 800, 700], 'Visible', 'on');
    max_abs_val = max(abs(diff_map_g2_g1(:)));
    imagesc(diff_map_g2_g1, [-max_abs_val max_abs_val]);
    colormap(jet);
    c = colorbar;
    c.Label.String = sprintf('Difference in %s proportion (G2 - G1)', current_phase);
    title(sprintf('Difference Map for %s (Group 2 - Group 1) with Significant Clusters', current_phase), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Create contours of significant areas without bwboundaries
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    hold off;
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig5b, fullfile(results_folder, sprintf('Difference_Map_G2_G1_With_Clusters_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig5b, fullfile(results_folder, sprintf('Difference_Map_G2_G1_With_Clusters_%s.fig', current_phase)));
    
    % 5. Create a comprehensive summary figure with both contrasts
    fig6 = figure('Position', [100, 100, 1200, 900], 'Visible', 'on');
    
    % Find the maximum value across both groups for consistent scaling
    max_value = max([max(mean_group1(:)), max(mean_group2(:))]);
    
    % Mean fixation maps
    subplot(3,2,1);
    imagesc(mean_group1, [0 max_value]);  % Set consistent color scale
    imagesc(mean_group1);
    title(sprintf('Group 1 Mean %s Map', current_phase), 'FontSize', 12);
    colormap(gca, hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    subplot(3,2,2);
    imagesc(mean_group2);
    imagesc(mean_group1, [0 max_value]);  % Set consistent color scale
    title(sprintf('Group 2 Mean %s Map', current_phase), 'FontSize', 12);
    colormap(gca, hot);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    % Difference maps (both directions) with symmetric scaling
    max_abs_val = max(max(abs(diff_map_g1_g2(:))), max(abs(diff_map_g2_g1(:))));
    
    subplot(3,2,3);
    imagesc(diff_map_g1_g2, [-max_abs_val max_abs_val]);
    title(sprintf('Difference Map for %s (G1-G2)', current_phase), 'FontSize', 12);
    colormap(gca, jet);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    subplot(3,2,4);
    imagesc(diff_map_g2_g1, [-max_abs_val max_abs_val]);
    title(sprintf('Difference Map for %s (G2-G1)', current_phase), 'FontSize', 12);
    colormap(gca, jet);
    colorbar;
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    % G1 > G2 Significant clusters
    subplot(3,2,5);
    imagesc(diff_map_g1_g2, [-max_abs_val max_abs_val]);
    colormap(gca, jet);
    colorbar;
    title('Significant Clusters (G1 > G2)', 'FontSize', 12);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Add contours of significant areas without bwboundaries
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    hold off;
    
    % G2 > G1 Significant clusters
    subplot(3,2,6);
    imagesc(diff_map_g2_g1, [-max_abs_val max_abs_val]);
    colormap(gca, jet);
    colorbar;
    title('Significant Clusters (G2 > G1)', 'FontSize', 12);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    hold on;
    
    % Add contours of significant areas without bwboundaries
    if any(significant_map_neg(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_neg, 1), 1:size(significant_map_neg, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_neg), [0.5 0.5], 'r', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_neg = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
        %         h_leg_neg.DisplayName = sprintf('Group 2 > Group 1 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_neg, 'Location', 'best');
    end
    
    if any(significant_map_pos(:))
        % Create properly aligned coordinate grid
        [y_grid, x_grid] = ndgrid(1:size(significant_map_pos, 1), 1:size(significant_map_pos, 2));
        
        % Draw white outline first (thicker)
        [~, h_contour_outer] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'w', 'LineWidth', 3);
        set(h_contour_outer, 'DisplayName', '');
        
        % Then draw black line on top (thinner)
        [~, h_contour_inner] = contour(x_grid, y_grid, double(significant_map_pos), [0.5 0.5], 'b', 'LineWidth', 1.5);
        set(h_contour_inner, 'DisplayName', '');
        
        %         % Create a separate line object purely for the legend
        %         h_leg_pos = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
        %         h_leg_pos.DisplayName = sprintf('Group 1 > Group 2 (p < %.2f)', alpha);
        %
        %         % Create legend with only our custom entry
        %         legend(h_leg_pos, 'Location', 'best');
    end
    hold off;
    
    % Add title using annotation
    annotation('textbox', [0.3, 0.95, 0.4, 0.05], 'String', ...
        sprintf('Eye-tracking CoSMoMVPA TFCE Analysis Results - %s Phase', current_phase), ...
        'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center');
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % 6. Create the combined significant results map without using bwboundaries
    fig7 = figure('Position', [100, 100, 800, 700], 'Visible', 'on');
    
    % Create a mask for positive and negative significant clusters
    binary_pos = zeros(size(zmap));
    binary_neg = zeros(size(zmap));
    
    if any(significant_map_pos(:))
        binary_pos(significant_map_pos) = 1;
    end
    
    if any(significant_map_neg(:))
        binary_neg(significant_map_neg) = -1;
    end
    
    % Combined significant results map
    combined_sig_map = binary_pos + binary_neg;
    
    % Use a custom colormap for the significant areas
    % -1: Blue (G2 > G1), 0: White (n.s.), 1: Red (G1 > G2)
    custom_cmap = [0 0 1; 1 1 1; 1 0 0];
    
    % Display the combined map
    imagesc(combined_sig_map, [-1 1]);
    colormap(custom_cmap);
    c = colorbar('Ticks', [-1, 0, 1], 'TickLabels', {'G2 > G1', 'n.s.', 'G1 > G2'});
    title(sprintf('Significant Differences in %s (TFCE-corrected, p < %.2f)', current_phase, alpha), 'FontSize', 14);
    axis image;
    set(gca, 'YDir', 'reverse');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    
    % Ensure rendering is complete
    drawnow;
    pause(0.5);
    
    % Save figure
    print(fig7, fullfile(results_folder, sprintf('Combined_Significant_Results_%s.png', current_phase)), '-dpng', '-r300');
    saveas(fig7, fullfile(results_folder, sprintf('Combined_Significant_Results_%s.fig', current_phase)));
    
    %% Save numerical results
    fprintf('Saving numerical results for %s phase...\n', current_phase);
    
    % Save results structure
    results = struct();
    results.parameters = struct('E', E, 'H', H, 'niter', niter, 'connect', connect, ...
        'Resolution_X', Resolution_X, 'Resolution_Y', Resolution_Y, ...
        'down_factor', down_factor, 'precision', precision_type, ...
        'alpha', alpha);
    results.zmap = zmap; % z-scores
    results.pmap = pmap; % p-values
    results.significant_map_pos = significant_map_pos;
    results.significant_map_neg = significant_map_neg;
    results.threshold_z = z_threshold_two_tailed;
    results.mean_group1 = mean_group1;
    results.mean_group2 = mean_group2;
    results.diff_map_g2_g1 = diff_map_g2_g1;
    results.diff_map_g1_g2 = diff_map_g1_g2;
    results.n_subj_g1 = n_subj_g1;
    results.n_subj_g2 = n_subj_g2;
    results.phase = current_phase;
    
    % Save in MAT format for future analysis
    save(fullfile(results_folder, sprintf('CoSMoMVPA_TFCE_Results_%s.mat', current_phase)), 'results');
    
    % Save key numerical results in text file
    fid = fopen(fullfile(results_folder, sprintf('CoSMoMVPA_TFCE_Summary_%s.txt', current_phase)), 'w');
    fprintf(fid, 'CoSMoMVPA TFCE Analysis Summary - %s Phase\n', current_phase);
    fprintf(fid, '==================================\n\n');
    fprintf(fid, 'Parameters:\n');
    fprintf(fid, '  E = %.2f\n', E);
    fprintf(fid, '  H = %.2f\n', H);
    fprintf(fid, '  Number of permutations = %d\n', niter);
    fprintf(fid, '  Connectivity = %d-connected\n', connect);
    fprintf(fid, '  Data precision = %s\n', precision_type);
    fprintf(fid, '  Significance level (alpha) = %.2f\n', alpha);
    
    fprintf(fid, 'Data Information:\n');
    fprintf(fid, '  Group 1: %d subjects\n', n_subj_g1);
    fprintf(fid, '  Group 2: %d subjects\n', n_subj_g2);
    fprintf(fid, '  Original resolution: %d x %d pixels\n', Resolution_X, Resolution_Y);
    fprintf(fid, '  Downsampled resolution: %d x %d pixels\n', ds_x, ds_y);
    fprintf(fid, '  Downsampling factor: %d\n\n', down_factor);
    
    fprintf(fid, 'Results:\n');
    fprintf(fid, '  Z-value threshold (two-tailed): %.4f\n', z_threshold_two_tailed);
    fprintf(fid, '  Number of significant pixels (Group 1 > Group 2): %d\n', sum(significant_map_pos(:)));
    fprintf(fid, '  Number of significant pixels (Group 2 > Group 1): %d\n', sum(significant_map_neg(:)));
    fprintf(fid, '  Total significant pixels: %d (%.2f%% of all pixels)\n', ...
        sum(significant_map_combined(:)), 100*sum(significant_map_combined(:))/(ds_x*ds_y));
    
    % Add performance information if timing was enabled
    if enable_timing
        phase_time = toc(phase_start_time);
        fprintf(fid, '\nPerformance Information:\n');
        fprintf(fid, '  Total processing time for %s phase: %.2f seconds\n', current_phase, phase_time);
        if exist('tfce_time', 'var')
            fprintf(fid, '  Time for Monte Carlo TFCE analysis: %.2f seconds\n', tfce_time);
        end
    end
    
    fclose(fid);
    
    fprintf('Analysis complete for %s phase! Results saved to %s\n', current_phase, results_folder);
    
    % Report timing for the current phase
    if enable_timing
        phase_time = toc(phase_start_time);
        fprintf('Total time for %s phase: %.2f seconds\n', current_phase, phase_time);
    end
    
    % Clear phase-specific variables to free memory before next phase
    if clear_temp_data
        clear group1_fixations_ds group2_fixations_ds zmap pmap
        clear mean_group1 mean_group2 diff_map_g2_g1 diff_map_g1_g2
        clear significant_map_pos significant_map_neg significant_map_combined
        clear ds z_ds cluster_nbrhood
    end
    
end % End of phase loop

% Report total timing for all phases
if enable_timing
    total_time = toc(total_start_time);
    fprintf('\nTotal processing time for all phases: %.2f seconds (%.2f minutes)\n', ...
        total_time, total_time/60);
end

%% Combine results from all phases
fprintf('\nCreating combined results from all phases...\n');

% Store the results from each phase
all_phase_results = struct();

for phase_idx = 1:length(phases)
    current_phase = phases{phase_idx};
    results_folder = fullfile(results_base_folder, current_phase);
    result_file = fullfile(results_folder, sprintf('CoSMoMVPA_TFCE_Results_%s.mat', current_phase));
    
    if exist(result_file, 'file')
        % Load the results for this phase
        phase_data = load(result_file);
        % Store in the all_phase_results structure
        all_phase_results.(current_phase) = phase_data.results;
    end
end

% Save the combined results to a single file
save(fullfile(results_base_folder, 'All_Phase_Results.mat'), 'all_phase_results');
fprintf('All results saved to %s\n', fullfile(results_base_folder, 'All_Phase_Results.mat'));

fprintf('\nAnalysis completed successfully!\n');
fprintf('To access results for a specific phase, use: all_phase_results.PhaseName\n');
fprintf('For example: all_phase_results.Decision\n');

% Clear variables as needed
if clear_temp_data
    clearvars -except all_phase_results
end