%% Gaze Peak Detection and Data Extraction
% This script detects peak gaze positions in left and right hemifields and
% extracts data around these peaks for further analysis.
%
% REQUIRES: Output files from calculate_gaze_ratios_v1.m (GazeAnalysisGambling package)
%
% Author: Y. Shigemune
% Created: 5/10/2025
% Last Modified: 5/10/2025
% Version: 1.0.0
%
% Description:
%   This script identifies bilateral peak gaze positions (left and right hemifields)
%   and extracts rectangular regions (201x201 pixels) around these peaks for
%   subsequent statistical analysis.
%
% Prerequisites:
%   - MATLAB R2017b or later
%   - External functions: readfromexcel, xlswrite
%   - Completed preprocessing with GazeAnalysisGambling package
%
% Input files:
%   - Gaze ratio Excel files in Input_Group1/ and Input_Group2/ directories
%   - Format: Output from calculate_gaze_ratios_v1.m
%   - Columns: X, Y, DecisionLeft, DecisionRight, FeedbackLeft, FeedbackRight
%
% Output files:
%   - detected_peaks_group1.xlsx and detected_peaks_group2.xlsx
%     (Peak coordinates for each participant and phase)
%   - Peak area data files for each subject in Output_Group1/ and Output_Group2/
%     (Extracted 201x201 pixel regions around detected peaks)
%
% Key Parameters:
%   - Resolution_W: Screen width in pixels (default: 1024)
%   - Range: Extraction area radius around peaks (default: 100)
%
% Usage:
%   1. Place gaze ratio files in Input_Group1/ and Input_Group2/
%   2. Run script: detect_bilateral_peaks_and_extract_v1
%   3. Check outputs in Output_Group1/ and Output_Group2/
%
% References:
%   Part of TFCE Moments Eyetracking Toolbox
%   Extends GazeAnalysisGambling package functionality

%% Parameter settings - Modify as needed
% Screen parameters
Resolution_W = 1024;         % Screen width in pixels for hemifield separation
Range = 100;  

% Paths - Edit according to your folder structure
curDir = pwd;
group1_input_folder  =  strcat(curDir, filesep, 'Input_Group1');
group2_input_folder  =  strcat(curDir, filesep, 'Input_Group2');
group1_output_folder  =  strcat(curDir, filesep, 'Output_Group1');
group2_output_folder  =  strcat(curDir, filesep, 'Output_Group2');

% Create base results folder if it doesn't exist
if ~exist(group1_output_folder, 'dir')
    mkdir(group1_output_folder);
end

if ~exist(group2_output_folder, 'dir')
    mkdir(group2_output_folder);
end

%% Get list of files to process
dataSet01 = dir(fullfile(group1_input_folder ,'*.xlsx'));
dataSet02 = dir(fullfile(group2_input_folder ,'*.xlsx'));

%% Define output headers and coordinate systems
OutPutHeader_Range = cell(1,6);
OutPutHeader_Range(1,:)=[{'X'} {'Y'} {'Response_L'} {'Response_R'} {'Feedback_L'} {'Feedback_R'}];

% Create coordinate system for peak area extraction
DataSet_Cordinate=cell((Range + 1)*(Range + 1),2);
for i = 1 : Range*2+1
    for j = 1 : Range*2+1
        DataSet_Cordinate{(Range*2+1)*(i-1)+j,1}=i;
        DataSet_Cordinate{(Range*2+1)*(i-1)+j,2}=j;
    end
end

%% Process Group 1 files
if size(dataSet01,1) > 0
    
    OutPutHeader_Peak=[{'File'} {'Res_LX'} {'Res_LY'} {'Res_RX'} {'Res_RY'} {'Fed_LX'} {'Fed_LY'} {'Fed_RX'} {'Fed_RY'}];
    
    DataSet_Peak=cell(size(dataSet01,1),9);
    DataSet_Peak(:,:)=[{''}];
    
    fprintf('\n\n========== Processing Group 1 Files ==========\n\n');
    
    for iCurFile = 1:size(dataSet01,1)        
        
        % Read data file
        SubDataSet = readfromexcel(fullfile(group1_input_folder , dataSet01(iCurFile).name),'sheet','Sheet1','All');
        SubDataSet =  SubDataSet(2:end,:);
        
        % Process each phase 
        for iCurPhase = 1:2
            
            % Find peaks by sorting data by phase values
            SubDataSet = sortrows(SubDataSet,iCurPhase+2,'descend'); 
            
            onPeakL = 0;
            onPeakR = 0;
            clear PeakL
            clear PeakR
            
            % Find first peak in each hemifield
            for i = 1 : size(SubDataSet,1)
                % Left
                if SubDataSet{i,1} < Resolution_W/2
                    if onPeakL == 0
                        PeakL = SubDataSet(i,1:2); % colum 1: X, colum 2: Y
                    end
                    onPeakL = onPeakL + 1;
                    % Right
                elseif SubDataSet{i,1} >= Resolution_W/2
                    if onPeakR == 0
                        PeakR = SubDataSet(i,1:2);
                    end
                    onPeakR = onPeakR + 1;
                end
                if onPeakR == 1 && onPeakL == 1
                    break
                end
            end
            
            % Store peak coordinates
            if iCurPhase == 1
                Peak = [PeakL PeakR];
            else
                Peak = [Peak PeakL PeakR];
            end
            
            % Extract data around peaks
            DataSetL = cell((Range*2+1)*(Range*2+1),1);
            DataSetR = cell((Range*2+1)*(Range*2+1),1);
            
            % Initialize data - set 0 to all cells
            for idx = 1:((Range*2+1)*(Range*2+1))
                DataSetL{idx,1} = 0;
                DataSetR{idx,1} = 0;
            end
            
            % Get left and right peak coordinates
            peakL_x = PeakL{1,1};
            peakL_y = PeakL{1,2};
            peakR_x = PeakR{1,1};
            peakR_y = PeakR{1,2};
            
            % Scan SubDataSet and extract only coordinates near peaks
            for row = 1:size(SubDataSet, 1)
                x = SubDataSet{row, 1};
                y = SubDataSet{row, 2};
                
                % Check if coordinates are around left peak
                x_offset_L = x - peakL_x;
                y_offset_L = y - peakL_y;
                if abs(x_offset_L) <= Range && abs(y_offset_L) <= Range
                    i = x_offset_L + Range + 1;
                    j = y_offset_L + Range + 1;
                    idx = (Range*2+1)*(i-1) + j;
                    DataSetL{idx, 1} = SubDataSet{row, iCurPhase+2};
                end
                
                % Check if coordinates are around right peak
                x_offset_R = x - peakR_x;
                y_offset_R = y - peakR_y;
                if abs(x_offset_R) <= Range && abs(y_offset_R) <= Range
                    i = x_offset_R + Range + 1;
                    j = y_offset_R + Range + 1;
                    idx = (Range*2+1)*(i-1) + j;
                    DataSetR{idx, 1} = SubDataSet{row, iCurPhase+2};
                end
            end
            
            if iCurPhase == 1
                DataSet = [DataSetL DataSetR];
            else
                DataSet = [DataSet DataSetL DataSetR];
            end            
            
        end
        
        % Save peak coordinates
        [x,FileName,y] = fileparts(dataSet01(iCurFile).name);
        DataSet_Peak(iCurFile,:) =[{FileName} Peak];
        
        % Save data for current file
        FileName = strcat(FileName,'.xlsx');
        cd (group1_output_folder );
        xlswrite ([DataSet_Cordinate DataSet],'',OutPutHeader_Range, FileName);
        cd (curDir);
        
    end
    
    % Save peak detection results for Group 1
    xlswrite (DataSet_Peak,'',OutPutHeader_Peak, 'detected_peaks_group1.xlsx');
    
end

%% Process Group 2 files
fprintf('\n\n========== Processing Group 2 Files ==========\n\n');

if size(dataSet02,1) > 0
    
    OutPutHeader_Peak=[{'File'} {'Res_LX'} {'Res_LY'} {'Res_RX'} {'Res_RY'} {'Fed_LX'} {'Fed_LY'} {'Fed_RX'} {'Fed_RY'}];
    
    DataSet_Peak=cell(size(dataSet02,1),9);
    DataSet_Peak(:,:)=[{''}];
    
    for iCurFile = 1:size(dataSet02,1)
        
        % Read data file
        SubDataSet = readfromexcel(fullfile(group2_input_folder , dataSet02(iCurFile).name),'sheet','Sheet1','All');
        SubDataSet =  SubDataSet(2:end,:);
        
        % Process each phase 
        for iCurPhase = 1:2
            
            % Find peaks by sorting data by phase values
            SubDataSet = sortrows(SubDataSet,iCurPhase+2,'descend'); 
            
            onPeakL = 0;
            onPeakR = 0;
            clear PeakL
            clear PeakR
            
            % Find first peak in each hemifield
            for i = 1 : size(SubDataSet,1)
                % Left
                if SubDataSet{i,1} < Resolution_W/2
                    if onPeakL == 0
                        PeakL = SubDataSet(i,1:2); % colum 1: X, colum 2: Y
                    end
                    onPeakL = onPeakL + 1;
                    % Right
                elseif SubDataSet{i,1} >= Resolution_W/2
                    if onPeakR == 0
                        PeakR = SubDataSet(i,1:2);
                    end
                    onPeakR = onPeakR + 1;
                end
                if onPeakR == 1 && onPeakL == 1
                    break
                end
            end
            
            % Store peak coordinates
            if iCurPhase == 1
                Peak = [PeakL PeakR];
            else
                Peak = [Peak PeakL PeakR];
            end
            
            % Extract data around peaks
            DataSetL = cell((Range*2+1)*(Range*2+1),1);
            DataSetR = cell((Range*2+1)*(Range*2+1),1);
            
            % Initialize data - set 0 to all cells
            for idx = 1:((Range*2+1)*(Range*2+1))
                DataSetL{idx,1} = 0;
                DataSetR{idx,1} = 0;
            end
            
            % Get left and right peak coordinates
            peakL_x = PeakL{1,1};
            peakL_y = PeakL{1,2};
            peakR_x = PeakR{1,1};
            peakR_y = PeakR{1,2};
            
            % Scan SubDataSet and extract only coordinates near peaks
            for row = 1:size(SubDataSet, 1)
                x = SubDataSet{row, 1};
                y = SubDataSet{row, 2};
                
                % Check if coordinates are around left peak
                x_offset_L = x - peakL_x;
                y_offset_L = y - peakL_y;
                if abs(x_offset_L) <= Range && abs(y_offset_L) <= Range
                    i = x_offset_L + Range + 1;
                    j = y_offset_L + Range + 1;
                    idx = (Range*2+1)*(i-1) + j;
                    DataSetL{idx, 1} = SubDataSet{row, iCurPhase+2};
                end
                
                % Check if coordinates are around right peak
                x_offset_R = x - peakR_x;
                y_offset_R = y - peakR_y;
                if abs(x_offset_R) <= Range && abs(y_offset_R) <= Range
                    i = x_offset_R + Range + 1;
                    j = y_offset_R + Range + 1;
                    idx = (Range*2+1)*(i-1) + j;
                    DataSetR{idx, 1} = SubDataSet{row, iCurPhase+2};
                end
            end
            
            if iCurPhase == 1
                DataSet = [DataSetL DataSetR];
            else
                DataSet = [DataSet DataSetL DataSetR];
            end
            
            
        end
        
        % Save peak coordinates
        [x,FileName,y] = fileparts(dataSet02(iCurFile).name);
        DataSet_Peak(iCurFile,:) =[{FileName} Peak];
        
        % Save data for current file
        FileName = strcat(FileName,'.xlsx');
        cd (group2_output_folder );
        xlswrite ([DataSet_Cordinate DataSet],'',OutPutHeader_Range, FileName);
        cd (curDir);
        
    end
    
    % Save peak detection results for Group 2
    xlswrite (DataSet_Peak,'',OutPutHeader_Peak, 'detected_peaks_group2.xlsx');
    
end

