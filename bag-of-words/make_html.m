function [] = make_html(results, sift_mode, color, vocab_size, vocab_batch, train_batch_size)
%MAKE_HTML writes the results to an HTML file in the required format. 
%   results : .mat struct containing test results, stored in 'part1_results' 
%   sift_mode : 'key' for keypoint SIFT or 'dense' for dense SIFT
%   color : color mode, either 'gray', 'RGB, 'nrgb' or 'opponent'
%   vocab_size : vocabulary size
%   vocab_batch : number of images used to create visual vocabulary
%   train_batch_size : number of training instances per class (same for
%       positive and negative examples) 

classes = cellstr(['airplanes '; 'cars      '; 'faces     '; 'motorbikes']);
prob_estimates = results.prob_estimates;
rankings = [];
aps = [];

% Determine the predicted orderings and (Mean) Average Precision scores 
for idx = 1 : 4
    class = classes{idx};
    [s, ap] = evaluate_scores(prob_estimates(:,idx), class);
    rankings = [rankings s];
    aps = [aps ap];
end

total_map = mean(aps);

out_path = strcat(sift_mode, color, '_vocabsize', string(vocab_size), ...
    '_vocabbatch', string(vocab_batch), '_trainbatch', string(train_batch_size), '.html');

fileout = fopen(out_path, 'w');
fid = fopen('Template_Result.html');
count_lines = 1;
tline = fgets(fid);

% Construct the HTML results file by changing relevant info per line,
% reading from the template HTML results file
while ischar(tline) 
    if count_lines == 14
        fprintf(fileout, '<h2> Jedda Boyle, Mathijs Mul </h2>');
        fprintf(fileout,'\n'); 
    elseif count_lines == 17
        if strcmp(sift_mode, 'key') == 1
            fprintf(fileout, '<tr><th>SIFT step size</th><td> keypoints </td></tr>');
        else
            fprintf(fileout, '<tr><th>SIFT step size</th><td> 5 px </td></tr>');
        end
        fprintf(fileout,'\n'); 
    elseif count_lines == 18
        fprintf(fileout, '<tr><th>SIFT block sizes</th><td>2x2 pixels</td></tr>');
        fprintf(fileout,'\n'); 
    elseif count_lines == 19
        fprintf(fileout, strcat('<tr><th>SIFT method</th><td>',color,'-SIFT</td></tr>')); 
        fprintf(fileout,'\n'); 
    elseif count_lines == 20
        fprintf(fileout, strcat('<tr><th>Vocabulary size</th><td>',string(vocab_size),' words</td></tr>')); 
        fprintf(fileout,'\n'); 
    elseif count_lines == 21
        fprintf(fileout, strcat('<tr><th>Vocabulary fraction</th><td>', string(vocab_batch / 500), '</td></tr>')); 
        fprintf(fileout,'\n');
    elseif count_lines == 22
        fprintf(fileout, strcat('<tr><th>SVM training data</th><td>', string(train_batch_size), ' positive, ', string(train_batch_size), ' negative per class</td></tr>'));  
        fprintf(fileout,'\n'); 
    elseif count_lines == 23
        fprintf(fileout, '<tr><th>SVM kernel type</th><td> n/a </td></tr>');
        fprintf(fileout,'\n'); 
    elseif count_lines == 25
        fprintf(fileout, strcat('<h1>Prediction lists (MAP:', string(total_map), ')</h1>'));
        fprintf(fileout,'\n'); 
    elseif count_lines == 26
        fprintf(fileout, '<h3><font color="red">Following are the ranking lists for the four categories. </font></h3>');
        fprintf(fileout, '\n');
    elseif count_lines == 27
        fprintf(fileout, '<h3><font color="red">The length of each column is 200 (containing all test images).</font></h3>');
        fprintf(fileout, '\n');
    elseif count_lines == 31
        line = strcat('<th>Airplanes (AP: ', string(aps(1)), ...
            ')</th><th>Cars (AP: ', string(aps(2)), ')</th><th>Faces (AP: ', ...
            string(aps(3)), ')</th><th>Motorbikes (AP: ', string(aps(4)), ')</th>');
        fprintf(fileout, line);
        fprintf(fileout,'\n');   
    elseif count_lines == 35 
        for i = 1 : 200
            tline = '<tr>';
            image_tags = rankings(i,:);
            for i = 1 : 4
                class_tag = string(image_tags(i)){1}(1);
                length = size(string(image_tags(i)){1}, 2);
                image_nr = string(image_tags(i)){1}(2:length);
                
                if length == 2
                    image_nr = strcat('00', image_nr);
                elseif length == 3
                    image_nr = strcat('0', image_nr);
                end

                if strcmp(class_tag, 'a') == 1
                    class = 'airplanes';
                elseif strcmp(class_tag, 'f') == 1
                    class = 'faces';
                elseif strcmp(class_tag, 'c') == 1
                    class = 'cars';
                elseif strcmp(class_tag, 'm') == 1
                    class = 'motorbikes';
                end

                image_path = strcat('<td><img src="Caltech4/ImageData/', class, '_test/img', image_nr, '.jpg" /></td>');
                tline = strcat(tline, image_path);
            end
            tline = strcat(tline, '</tr>');
            fprintf(fileout, tline);
            fprintf(fileout, '\n');
        end
    elseif count_lines < 35 || count_lines > 41
        fprintf(fileout, tline);
    end
    
    count_lines = count_lines + 1;
    tline = fgets(fid);
end

fclose(fid);

end