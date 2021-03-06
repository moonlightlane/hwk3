%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELEC 345 Assignment 4 Part 2 Algorithm Implementation
% Author: Zichao Wang
% Date  : April 21th, 2015
%
% Note: the implementation here deal with the reduced dataset ONLY
% Using SURF functions 
% histogram vector for each image in the training image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Feature Extraction (test: using 5 images from each class)
% get all training images
Trainingdir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_expanded/TrainingDataset';
IMG = dir(Trainingdir); % struct for all sub directories (25 directories)
IMG = IMG(3:end);
% load images, class by class
D = []; % predefine matrix to store all features
F = struct(); % struct to store feature vector for each class
N = 300; % number of cluster centers for each class
C = []; % cluster center matrix
th = 100; % threshold to dicard useless features
for i = 1:length(IMG) % for each directory
    img = dir(strcat(Trainingdir,'/',IMG(i).name)); % class directory
    img = img(3:end);
    d = []; % predefine matrix to store features for a class
    for j = 1:length(img) % for each image of that class
        I = imread(strcat(Trainingdir,'/',IMG(i).name,'/',img(j).name));
        if size(size(I),2) == 3
            I = rgb2gray(I);
        end
        points = detectSURFFeatures(I,'NumOctaves',3,'NumScaleLevels',5);
        [features, validpoints] = extractFeatures(I,points);
        d = [d features'];
    end
    str = strcat('class',num2str(i));
    F.(str) = d;
    D = [D d]; % collect features 
    % kmeans clustering for each class
    D = single(D);
    [c,~] = vl_kmeans(D,N);
    C = [C c];
end
%% find histogram - descriptor for each image in each class using knnsearch
H = []; % predefine matrix for histogram vectors. This is a three dimensional matrix, with the third dimension indicating the class
classes = fieldnames(F); % all class names
for i = 1:numel(fieldnames(F)) % for each class
    d = F.(classes{i});
    [IDX, Dist] = knnsearch(C',single(d)');
    th_idx = find(Dist >= th);
    IDX(th_idx) = 0;
    hist = [];
    for m = 1:length(C)
        k = find(IDX == m);
        if isempty(k)
            hist = [hist 0];
        else
            hist = [hist length(k)]; 
        end
    end
    hist = hist./sum(hist); % normalize histogram
    H = [H hist'];
end
%% test
% construct confusion matrix
confusion = zeros(length(IMG));
% read test images, form histogram vector, and assign class
testdir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_expanded/TestDataSet';
test_IMG = dir(testdir); % stores values for test image directories
test_IMG = test_IMG(4:end);
for i = 1:length(test_IMG) % for each test class
    test_img = dir(strcat(testdir,'/',test_IMG(i).name));
    test_img = test_img(3:end);
    for j = 1:length(test_img) % for each image in that test class
        I = imread(strcat(testdir,'/',test_IMG(i).name,'/',test_img(j).name)); % read image
        if size(size(I),2) == 3
            I = rgb2gray(I);
        end
        points = detectSURFFeatures(I);
        [d, validpoints] = extractFeatures(I,points);
        [IDX_test Dist_test] = knnsearch(C',single(d')');
        th_idx = find(Dist_test >= th);
        IDX_test(th_idx) = 0;
        hist = [];
        % assign featuer to cluster
        for m = 1:length(C)
            k = find(IDX_test == m);
            if isempty(k)
                hist = [hist 0];
            else
                hist = [hist length(k)]; % PROBLEM: length of hist1 is longer than d1
            end
        end
        % normalize histogram
        hist = hist./sum(hist);
        % assign class
        class = knnsearch(H',hist);
        confusion(i,class) = confusion(i,class) + 1;
    end
    confusion(i,:) = confusion(i,:)./length(test_img);
end