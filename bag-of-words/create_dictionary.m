function [visual_words] = create_dictionary(n, k, sift_mode, color)
%CREATE_DICTIONARY creates a visual dictionary of k words using the first n pictures
%of the train folder of each class. These images should not be used for 
%training later on. 
%   n : number of pictures to be used per class (vocabulary batch)
%   k : number of clusters used in k-means, equals vocabulary size
%   sift_mode : 'key' or 'dense'
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'

classes = cellstr(['airplanes '; 'cars      '; 'faces     '; 'motorbikes']);

descriptor_matrix = [];

for class_idx = 1 : 4
    image_count = 0; 
    class = classes{class_idx};

    folder_name = strcat('Caltech4/ImageData/', class , '_train/');
    folder = dir(fullfile(folder_name, '*.jpg') );
    idx = 1;
    
    % Keep on reading images until required number has been reached
    while image_count < n
        image_path = strcat(folder_name, folder(idx).name);
        image = imread(image_path);
    
        if size(image, 3) ~= 1
            % Skip grayscale images
            
            descriptors = get_features(image_path, sift_mode, color);
            descriptor_matrix = [descriptor_matrix descriptors];
            image_count = image_count + 1;
        end
        
        idx = idx + 1;
        
     end
    
end

% Perform k-means clustering on descriptors
[visual_words, ~] = vl_kmeans(single(descriptor_matrix), k); 
vocabulary = single(visual_words);

% Save vocabulary
filename = char(strcat('part1_vocabs/', sift_mode, color, '_vocabsize', string(k), '_vocabbatch', string(n)));
save(filename, 'vocabulary');

end




