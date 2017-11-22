function [train_data] = construct_train(vocabulary, vocab_batch, batch_size, sift_mode, color)
%CONSTRUCT_TRAIN constructs the full data set that can actually be used for
%training. Labels are constructed per set separately. 
%   class : class of the current classifier, one of 'airplanes', 'cars',
%               'faces', 'motorbikes'
%   vocabulary : set of visual words for histogram computation
%   vocab_batch : number of images used for vocabulary construction
%   batch_size : number of training instances per class 
%   sift_mode : 'key' or 'dense'
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'

classes = cellstr(['airplanes '; 'cars      '; 'faces     '; 'motorbikes']);
train_data = [];
for idx = 1 : 4
    train_class = classes{idx};
    train_folder = strcat('Caltech4/ImageData/', train_class , '_train/');
    data = data_folder(train_folder, vocabulary, (vocab_batch + 1), batch_size, sift_mode, color);
    train_data = [train_data ; data];
end

end
