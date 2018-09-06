clear all; close all; clc

[class,attrib1, attrib2, attrib3, attrib4, attrib5,attrib6,attrib7,attrib8,attrib9, attrib10,attrib11,attrib12,attrib13] = textread('wine.data', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f', 'delimiter', ',');
attrib = [attrib1'; attrib2'; attrib3'; attrib4'; attrib5';attrib6';attrib7';attrib8';attrib9'; attrib10';attrib11';attrib12';attrib13']';
label = zeros(178, 1);
label(class==1) = 0;
label(class==2) = 1;
data=[attrib,label];

% [attrib1, attrib2, attrib3, attrib4, class] = textread('iris.data', '%f%f%f%f%s', 'delimiter', ',');
% attrib = [attrib1'; attrib2'; attrib3'; attrib4']';
% label = zeros(150, 1);
% label(strcmp(class, 'Iris-setosa')) = 0;
% label(strcmp(class, 'Iris-versicolor')) =1;
% label(strcmp(class, 'Iris-virginica')) = 1;
% data=[attrib,label];

accuracy=zeros(1,10);
   [A,B]=size(data);
    indices=crossvalind('Kfold',data(1:A,B),10);
    for k=1:10
        temp_count=0;
        test = (indices == k); 
        train = ~test;
        train_data=data(train,:);
       % disp(train_data);
        train_targets=train_data(:,size(train_data,2))';
        test_data=data(test,:);
        test_targets=test_data(:,size(test_data,2))';
       
        [test_targets_predict, a_star] = SVM(train_data', train_targets, test_data','RBF', 0.05, 'Perceptron', inf);

        for i=1:size(test_targets_predict,2)
              if test_targets(:,i)==test_targets_predict(:,i)
                 temp_count=temp_count+1;
              end
              accuracy(k)=temp_count/size(test_targets,2);
               disp(accuracy);
         end
    end      
 
M=mean(accuracy);
S=std(accuracy);
fprintf('The mean is %f\n',M);
fprintf('The std is %f\n',S);