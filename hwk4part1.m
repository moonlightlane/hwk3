%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELEC 345 Assignment 4 Part 1.1 Algorithm Implementation
% Author: Zichao Wang
% Date  : April 14th, 2015
%
% Note: the implementation here deal with the reduced dataset ONLY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Feature Extraction (test: using 5 images from each class)
% get all training images
buddha_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TrainingDataset/022.buddha-101';
butterfly_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TrainingDataset/024.butterfly';
airplane_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TrainingDataset/251.airplanes';
im1 = dir(buddha_dir);
im1 = im1(3:end);
im2 = dir(butterfly_dir);
im2 = im2(3:end);
im3 = dir(airplane_dir);
im3 = im3(3:end);
% predefine struct data type for each class. To fill feature values later
c1 = 'buddha';
d1 = [];
c2 = 'butterfly';
d2 = [];
c3 = 'airplanes';
d3 = [];
% load images, class by class
for i = 1:length(im1)
    % budhha-101 class
    I = imread(strcat(buddha_dir,'/',im1(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    d1 = [d1 d];
end
for i = 1:length(im2)
    % butterfly class
    I = imread(strcat(butterfly_dir,'/',im2(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    d2 = [d2 d];
end
for i = 1:length(im3)
    % airplanes class
    I = imread(strcat(airplane_dir,'/',im3(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    d3 = [d3 d];
end
%% kmeans clustering
% put everything in struct
class = struct(c1,d1,c2,d2,c3,d3);
% matrix for all features (convert to single) POTENTIAL PROBLEM ??????????
D = [d1,d2,d3];
D = single(D);
% k-means clustering
N = 1000; % number of centers
[C,A] = vl_kmeans(D,N);
% find histogram - descriptor for each class using knnsearch
th = 250; % distance threshold
    % budhha class
[IDX1, D1] = knnsearch(C',single(d1)');
th_idx = find(D1 >= th);
IDX1(th_idx) = 0;
hist1 = [];
for i = 1:length(C)
    k = find(IDX1 == i);
    if isempty(k)
        hist1 = [hist1 0];
    else
        hist1 = [hist1 length(k)]; % PROBLEM: length of hist1 is longer than d1
    end
end
    % butterfly class
[IDX2, D2] = knnsearch(C',single(d2)');
th_idx = find(D2 >= th);
IDX2(th_idx) = 0;
hist2 = [];
% assign featuer to cluster
for i = 1:length(C)
    k = find(IDX2 == i);
    if isempty(k)
        hist2 = [hist2 0];
    else
        hist2 = [hist2 length(k)]; % PROBLEM: length of hist1 is longer than d1
    end
end
    % airplanes class
[IDX3, D3] = knnsearch(C',single(d3)');
th_idx = find(D3 >= th);
IDX3(th_idx) = 0;
hist3 = [];
for i = 1:length(C)
    k = find(IDX3 == i);
    if isempty(k)
        hist3 = [hist3 0];
    else
        hist3 = [hist3 length(k)]; % PROBLEM: length of hist1 is longer than d1
    end
end
% normalize histogram
hist1 = hist1./sum(hist1);
hist2 = hist2./sum(hist2);
hist3 = hist3./sum(hist3);
% feature vector for each class; 1000-by-3. Sequence: buddha, butterfly, airplane
H = [hist1', hist2', hist3']; 
% plot histograms for all 3 classes
figure
subplot(1,3,1)
bar(hist1)
title('buddha','Fontsize',16)
subplot(1,3,2)
bar(hist2)
title('butterfly','Fontsize',16)
subplot(1,3,3)
bar(hist3)
title('airplanes','Fontsize',16)
%% test
% construct confusion matrix
confusion = zeros(3);
% read test images, form histogram vector, and assign class
test_buddha_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TestDataset_1';
test_butterfly_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TestDataset_2';
test_airplane_dir = '/Users/zichaowang/Dropbox/RICE/year spring 15/ELEC 345/hwk3/midterm_data/midterm_data_reduced/TestDataset_3';
test_im1 = dir(test_buddha_dir);
test_im1 = test_im1(3:end);
test_im2 = dir(test_butterfly_dir);
test_im2 = test_im2(3:end);
test_im3 = dir(test_airplane_dir);
test_im3 = test_im3(3:end);
for i = 1:length(test_im1)
    % buddha class test images
    I = imread(strcat(test_buddha_dir,'/',test_im1(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    [IDX1_test D1_test] = knnsearch(C',single(d)');
    th_idx = find(D1_test >= th);
    IDX1_test(th_idx) = 0;
    hist = [];
    % assign featuer to cluster
    for j = 1:length(C)
        k = find(IDX1_test == j);
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
    confusion(1,class) = confusion(1,class) + 1;
end
confusion(1,:) = confusion(1,:)./length(test_im1); % normalize confusion matrix
for i = 1:length(test_im2)
    % butterfly class
    I = imread(strcat(test_butterfly_dir,'/',test_im2(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    [IDX2_test D2_test] = knnsearch(C',single(d)');
    th_idx = find(D2_test >= th);
    IDX2_test(th_idx) = 0;
    hist = [];
    % assign featuer to cluster
    for j = 1:length(C)
        k = find(IDX2_test == j);
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
    confusion(2,class) = confusion(2,class) + 1;
end
confusion(2,:) = confusion(2,:)./length(test_im2);
for i = 1:length(test_im3)
    % airplane class
    I = imread(strcat(test_airplane_dir,'/',test_im3(i).name));
    if size(size(I),2) == 3
        I = single(rgb2gray(I));
    else
        I = single(I);
    end
    [f,d] = vl_sift(I);
    [IDX3_test D3_test] = knnsearch(C',single(d)');
    th_idx = find(D3_test >= th);
    IDX3_test(th_idx) = 0;
    hist = [];
    % assign featuer to cluster
    for j = 1:length(C)
        k = find(IDX3_test == j);
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
    confusion(3,class) = confusion(3,class) + 1;
end
confusion(3,:) = confusion(3,:)./length(test_im3);