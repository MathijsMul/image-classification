%% main function 


%% fine-tune cnn

% [net, info, expdir] = finetune_cnn();

%% extract features and train svm

nets.fine_tuned = net; % sload(fullfile(expdir, 'fine_tuned_net.mat')); nets.fine_tuned = nets.fine_tuned.net;

nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); nets.pre_trained = nets.pre_trained.net; 
data = load(fullfile(expdir, 'imdb-caltech.mat'));
 
data.images.data = single(data.images.data);
train_svm(nets, data);


%% Visualise Features 
nets.pre_trained.layers{end}.type = 'softmax';
nets.fine_tuned.layers{end}.type = 'softmax';

% Pick the number of image features to plot. 
N = 500;
fine_tuned_outputs = zeros(N, 4);
pre_trained_outputs = zeros(N, 10);

labels = zeros(N, 1);
for i = randperm(N)
    % Get features
   pre_trained_net_out = vl_simplenn(nets.pre_trained, data.images.data(:,:,:, i));
   fine_tuned_net_out = vl_simplenn(nets.fine_tuned, data.images.data(:,:,:, i));
   
   % Save features
   pre_trained_outputs(i, :) = squeeze(pre_trained_net_out(end).x)';
   fine_tuned_outputs(i,:) = squeeze(fine_tuned_net_out(end).x)';
   
   % Record the feature label.
   labels(i) = data.images.labels(i);
end

% Perform dimensionality reduction on the image features.
pre_trained_points = tsne(pre_trained_outputs);
fine_tuned_points = tsne(fine_tuned_outputs);

% Plot the results.
figure(2);
subplot(1,2,1)
scatter(pre_trained_points(:,1), pre_trained_points(:,2), 4, labels);
title('Pre-Trained Network.')
subplot(1,2,2)
scatter(fine_tuned_points(:,1), fine_tuned_points(:,2), 4, labels);
title('Fine-Tuned Network.')



