function [data] = data_folder(folder_path, vocabulary, start, batch_size, sift_mode, color)
%DATA_FOLDER constructs the training/test data from a specific folder
%(containing either positive or negative examples). It takes a number of
%images from the given folder, computes their histograms with respect to the provided
%vocabulary and returns the histograms as an N x V vector, where N denotes
%the number of training (or test) instances (batch_size) and V denotes the vocabulary size. 
%   folder_path : location of the directory
%   vocabulary : set of visual words for histogram computation
%   start : position in the folder from where to start 
%   batch_size : number of images to be considered (number of training or
%       test instances per class)
%   sift_mode : 'key' or 'dense'
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'

folder = dir(fullfile(folder_path, '*.jpg') );
k = size(vocabulary, 2); 
data = [];
image_count = 0; 
idx = start;

while image_count < batch_size
    image_path = strcat(folder_path, folder(idx).name);
    image = imread(image_path);
    
    if size(image, 3) ~= 1
        % Skip grayscale images
        histogram_vector = image2hist(image_path, vocabulary, k, sift_mode, color);
        data = [data ; histogram_vector];
        image_count = image_count + 1;
    end
    
    idx = idx + 1;
end

% Training/test instance matrix must be sparse 
data = sparse(data);

end
