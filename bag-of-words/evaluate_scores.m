function [sorted_ids, ap] = evaluate_scores(prob_vector, class)
%EVALUATE_SCORES takes the classification scores for one classifier, orders
%them from high to low and returns the corresponding image IDs as well as the 
%Average Precision. 
%   prob_vector : classification scores
%   class : class of the current classifier, one of 'airplanes', 'cars',
%               'faces', 'motorbikes'

% Abbreviate class identifiers 
classes = cellstr(['a'; 'c'; 'f'; 'm']);
all_ids = cellstr(['']);

% Generate unranked list of abbreviated names of all test images
count = 1;
for idx = 1 : 4
    test_class = classes{idx};
    
    for id = 1 : 50
       all_ids{count} = strcat(test_class, string(id));
       count = count + 1;
    end

end

all_ids = all_ids';

% Order probability estimates form high to low
[~, ordering] = sort(prob_vector, 'descend');

% Order test image IDs accordingly 
sorted_ids = all_ids(ordering);

% Compute Average Precision
correct_classifications = 0;
sum_precision = 0;
for idx = 1 : 200

   if strcmp(string(sorted_ids{idx}){1}(1), class(1)) == 1
       correct_classifications = correct_classifications + 1;
       score = correct_classifications;    
   else
       score = 0;
   end
   
   sum_precision = sum_precision + (score / idx);

end

ap = sum_precision / 50;

end