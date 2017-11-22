run('vlfeat-0.9.20/toolbox/vl_setup')
addpath('liblinear-2.1/matlab');

% Declare parameters 
vocab_batch = 50;
vocab_size = 400;
sift_mode = 'key';
color = 'RGB';
train_batch_size = 100;
test_batch = 50;
classes = cellstr(['airplanes '; 'cars      '; 'faces     '; 'motorbikes']);

%%
% Create visual vocabulary

vocabulary = create_dictionary(vocab_batch, vocab_size, sift_mode, color);

%% 
% Load visual vocabulary (if necessary) 
vocab_name = char(strcat('part1_vocabs/', sift_mode, color, '_vocabsize', string(vocab_size), '_vocabbatch', string(vocab_batch)));
load_vocab = load(vocab_name); 
vocabulary = load_vocab.vocabulary; 

%%
% Train classifiers 

train_data = construct_train(vocabulary, vocab_batch, train_batch_size, sift_mode, color);

for idx = 1 : 4
    class = classes{idx};
    
    % Construct labels
    train_labels = -1 * ones((4*train_batch_size), 1);
    train_labels((train_batch_size*(idx - 1) + 1): train_batch_size*idx) = 1;
    
    % Train model
    model = train(train_labels, train_data); %linear SVM with L2-loss function.

    % Save the model 
    model_name = char(strcat('part1_models/', class, '-', sift_mode, color, '_vocabsize', string(vocab_size), '_vocabbatch', string(vocab_batch), '_trainbatch', string(train_batch_size)));
    save(model_name,'model');

end

%%
% Test classifiers

test_data = construct_test(vocabulary, sift_mode, color, test_batch);

predicted_labels = [];
accuracy = [];
prob_estimates = [];

for idx = 1 : 4
    disp(strcat('Testing classifier ', string(idx)));
    
    class = classes{idx};
    model_name = char(strcat('part1_models/', class, '-', sift_mode, color, '_vocabsize', string(vocab_size), '_vocabbatch', string(vocab_batch), '_trainbatch', string(train_batch_size), '.mat'));
    loaded_model = load(model_name, 'model'); 
    
    % Construct labels
    test_labels = -1 * ones((4*test_batch), 1);
    test_labels((test_batch*(idx - 1) + 1): test_batch*idx) = 1;
    
    % Predict
    [predict_label, pred_accuracy, pred_prob_estimates] = predict(test_labels, test_data, loaded_model.model);
    predicted_labels = [predicted_labels predict_label];
    accuracy = [accuracy pred_accuracy];
    prob_estimates = [prob_estimates pred_prob_estimates];
end

% Save results
results_name = char(strcat('part1_results/', sift_mode, color, '_vocabsize', string(vocab_size), '_vocabbatch', string(vocab_batch), '_trainbatch', string(train_batch_size), '.mat'));
save(results_name,'predicted_labels', 'accuracy', 'prob_estimates');

%%
% Generate HTML results file

% Load results if necessary
results_name = char(strcat('part1_results/', sift_mode, color, '_vocabsize', string(vocab_size), '_vocabbatch', string(vocab_batch), '_trainbatch', string(train_batch_size), '.mat'));
results = load(results_name);

make_html(results, sift_mode, color, vocab_size, vocab_batch, train_batch_size);

