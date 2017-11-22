function [histogram_vector] = image2hist(image_path, vocabulary, k, sift_mode, color)
%IMAGE2HIST converts an image to a histogram representing the relative
%frequencies of the visual words closest to the image descriptors, where
%the visual words are taken from the vocabulary.
%For training, if the input image is from the class that defines the
%classifier, the output histograms serve as positive
%examples. Otherwise they are negative examples.
%   image_path : location of input image
%   vocabulary : visual words dictionary
%   k : vocabulary size (number of k-means clusters) 
%   sift_mode : 'key' or 'dense' 
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'

image = imread(image_path);
descriptors = get_features(image_path, sift_mode, color);
closest_words = [];

% Find closest visual words for all descriptors
for i = 1 : size(descriptors, 2)
    [~, word] = min(vl_alldist(single(descriptors(:,i)), vocabulary));
    closest_words = [closest_words word];
end

% Compute histogram
edges = 1 : k + 1; 
histogram_vector = histcounts(closest_words, edges, 'Normalization','probability');

end 
