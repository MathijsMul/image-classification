function [descriptors] = get_features(image_path, sift_mode, color)
%GET_FEATURES gets feature point descriptors for an input image using SIFT using either
%dense sampling or key points. 
%   image_path : path to input image
%   sift_mode : 'key' for keypoint SIFT or 'dense' for dense SIFT
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'

    image = imread(image_path);
    gray_im = rgb2gray(image);
    single_im = single(gray_im);

    if strcmp(sift_mode, 'key') 
        % Find keypoints by using grayscale version of the image
        [frames, descriptors] = vl_sift(single_im);
    end

    % Store color channels separately
    if strcmp(color, 'gray') == 0
        channel1 = single(image(:,:,1)); 
        channel2 = single(image(:,:,2)); 
        channel3 = single(image(:,:,3)); 
    end
    
    % RGB color space
    if strcmp(color, 'RGB')
        % For each channel, calculate descriptors separately for frames
        % found in grayscale
        
        if strcmp(sift_mode, 'key') == 1
            [~, d1] = vl_sift(channel1, 'Frames', frames);    
            [~, d2] = vl_sift(channel2, 'Frames', frames);  
            [~, d3] = vl_sift(channel3, 'Frames', frames);
        elseif strcmp(sift_mode, 'dense') == 1
            [~, d1] = vl_dsift(channel1, 'step', 5, 'size', 2);
            [~, d2] = vl_dsift(channel2, 'step', 5, 'size', 2);  
            [~, d3] = vl_dsift(channel3, 'step', 5, 'size', 2);
        end
        
        % Concatenate descriptors
        descriptors = [d1; d2; d3];
    
    % Convert the image into normalized RGB color space
    elseif strcmp(color, 'nrgb')
        norm = channel1 + channel2 + channel3;
        
        new_channel1 = channel1 ./ norm;  
        new_channel2 = channel2 ./ norm;     
        new_channel3 = channel3 ./ norm;
        
        if strcmp(sift_mode, 'key') == 1
            [~, d1] = vl_sift(new_channel1, 'Frames', frames);    
            [~, d2] = vl_sift(new_channel2, 'Frames', frames);
            [~, d3] = vl_sift(new_channel3, 'Frames', frames);
        elseif strcmp(sift_mode, 'dense') == 1
            [~, d1] = vl_dsift(new_channel1, 'step', 5, 'size', 2);
            [~, d2] = vl_dsift(new_channel2, 'step', 5, 'size', 2);  
            [~, d3] = vl_dsift(new_channel3, 'step', 5, 'size', 2);
        end
        
        % Concatenate descriptors
        descriptors = [d1; d2; d3];
    
    % Convert the image into opponent color space
    elseif strcmp(color, 'opponent')
        new_channel1 = (channel1 - channel2) / sqrt(2);
        new_channel2 = (channel1 + channel2 - 2*channel3) / sqrt(6);
        new_channel3 = (channel1 + channel2 + channel3) / sqrt(3);
        
        if strcmp(sift_mode, 'key') == 1
            [~, d1] = vl_sift(new_channel1, 'Frames', frames);    
            [~, d2] = vl_sift(new_channel2, 'Frames', frames);
            [~, d3] = vl_sift(new_channel3, 'Frames', frames);
        elseif strcmp(sift_mode, 'dense') == 1
            [~, d1] = vl_dsift(new_channel1, 'step', 5, 'size', 2);
            [~, d2] = vl_dsift(new_channel2, 'step', 5, 'size', 2);  
            [~, d3] = vl_dsift(new_channel3, 'step', 5, 'size', 2);
        end

        % Concatenate descriptors
        descriptors = [d1; d2; d3];
    end
    
end
