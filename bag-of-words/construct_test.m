function [test_data] = construct_test(vocabulary, sift_mode, color, test_instances)
%CONSTRUCT_TEST constructs the test data. It takes all test images from all
%classes and computes their histograms with respect to the provided visual
%vocabulary. It returns the histograms of all images. 
%   vocabulary : set of visual words for histogram computation
%   sift_mode : 'key' or 'dense'
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'
%   test_instances : number of test images per class (usually 50)

classes = cellstr(['airplanes '; 'cars      '; 'faces     '; 'motorbikes']);
test_data = [];

for idx = 1 : 4
    test_class = classes{idx};
    test_folder = strcat('Caltech4/ImageData/', test_class , '_test/');
    data = data_folder(test_folder, vocabulary, 1, test_instances, sift_mode, color);
    test_data = [test_data ; data];
end

end
