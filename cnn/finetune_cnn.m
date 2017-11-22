function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', '..', '..', 'matlab', 'vl_setupnn.m')) ;

run(fullfile('/Users/Jed/GitHub/matconvnet/matlab/vl_setupnn.m')) ;


opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
%if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
%opts.train.gpus = [1];

opts.train.gpus = [];




%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

%%
trainfn = @cnn_train ;
imdb.images.data = single(imdb.images.data);
[net, info] = trainfn(net, imdb, getBatch(opts), ...
'expDir', opts.expDir, ...
net.meta.trainOpts, ...
opts.train, ...
'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getCaltechIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
% splits = {'train', 'test'};

% Setup Arguments.
path_image_sets = '../Caltech4/ImageSets/';
path_image_data = '../Caltech4/ImageData/';
im_size = 32    ;

% Init output data structures. 
data = single(zeros(im_size, im_size, 3, 1, 4131));
labels = zeros(1, 4131);
sets = zeros(1, 4131);

% Open the folder containing the images set details.
folder = dir(path_image_sets);

% Loop over files in folder.
idx = 0;
for i = 3 : size(folder, 1)

    % Open the file
    filename = strcat(path_image_sets, folder(i).name);
    file = fopen(filename,'r');
    line = fgets(file);

    % Loop over every line in the file.
    while ischar(line)
        idx = idx + 1;
        % Determine if the image is in the training or test set. 
        if strfind(line, 'train')
            sets(idx) = 1;
        elseif strfind(line, 'test')
            sets(idx) = 2;
        end

        % Determine image label. 
        if strfind(line, 'airplane')
            labels(idx) = 1;
        elseif strfind(line, 'car')
            labels(idx) = 2;
        elseif strfind(line, 'face')
            labels(idx) = 3;
        elseif strfind(line, 'motorbike')
            labels(idx) = 4;
        end

        % Read in the image and add it to imdb.
        im = imread(strcat(path_image_data, line, '.jpg'));
        im = imresize(im,[im_size im_size]);
        
        if (size(im, 3) == 1)
            im = cat(3, im, im, im);
        end

        % Add image to imdb.
        data(:,:,:, idx) = im2single(im);

        % Get next line. 
        line = fgets(file);

    end

    fclose(file);

end

%%
% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);

data = bsxfun(@minus, data, dataMean);
imdb.images.data = single(data) ;


imdb.images.labels = single(labels) ;
imdb.images.set = single(sets);
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;
 
perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end
