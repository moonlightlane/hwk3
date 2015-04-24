%% 2 Can you fix it? Yes you can!
% Training
which_Class = cell(25,1);      % set up a cell for classes
Training = dir(strcat('C:\Users\pc\Desktop\作业\ELEC 345\Assignment\HW 2\midterm_data\midterm_data_expanded\','TrainingDataset\'));     % Get the training image directory
Training(1)=[];                % First row is invaild
Training(1)=[];                % Second row is also invalid
[numClass, ~] = size(Training);    % get number of classes

for i = 1: numClass             
    Struct(i).Class = dir(strcat('C:\Users\pc\Desktop\作业\ELEC 345\Assignment\HW 2\midterm_data\midterm_data_expanded\','TrainingDataset\', Training(i).name)); % get each training images
    Struct(i).Class(1)=[];     
    Struct(i).Class(1)=[];
    which_Class{i} = strtok(Training(i).name, '.');        % Get the index of class
end

feature=[];     
for i = 1: numClass                                       % for loop of each class
    [numImage, ~] = size(Struct(i).Class);                % get the number of images in each class
    d_Class= [];                                          
    for j = 1: numImage                                  % for loop of each image in the certain class
        I_Class = im2double(imread(strcat('C:\Users\pc\Desktop\作业\ELEC 345\Assignment\HW 2\midterm_data\midterm_data_expanded\','TrainingDataset\',Training(i).name,'\', Struct(i).Class(j).name)));
        if size(size(I_Class),2) == 3     % if not grayscale
            I_Class=rgb2gray(I_Class);    % convert to grayscale
           % [~,d_Class1] = vl_sift(I_Class);
%         else
%             I_Class=single(I_Class);
           % [~,d_Class1] = vl_sift(I_Class);
        end
        pts = detectSURFFeatures(I_Class,'NumOctaves',3,'NumScaleLevels',5);      % call matlab built in SURF function
        [d_Class1,~] = extractFeatures(I_Class,pts);                              % call matlab built in extract function
        d_Class = [d_Class d_Class1'];                                            % build up the descriptor vector
    end
    %[~,d_Class] = pca(double(d_Class'),'numcomponents',20);
    Struct(i).Descrip = d_Class';                                                 % save the descriptor into Struct.Descrip
    feature = [feature d_Class];                                                  % build up the feature matrix
end

numCluster = 2000;                          % set the number of clusters
threshpercent=0.95;                         % set the threshold percent
[Center, ~] = vl_kmeans(feature, numCluster,'distance','l2','algorithm','ann');    % Use vl_kmeans to do the clustering

Hist = zeros(numClass, numCluster);         % generate the histogram matrix
for i = 1:numClass                          % for loop of each class
    Classhist=[];                 
    [Class_IDX, Class_d] = knnsearch(Center', double(Struct(i).Descrip));         % call knnsearch function 
    Classthresh = max(Class_d) * threshpercent;                                   % apply the threshold 
    Class_IDX(Class_d > Classthresh) = 0;                                         % apply the threshold
    for j=1:length(Center)                                                        % for loop of each cluster
        k=find(Class_IDX==j);                                                     % find index that close to the certain cluster center
        if isempty(k)                                                             % if none
           Classhist=[Classhist 0];                                               % set it to zero
        end
        Classhist(j)=length(k);                                                   % histogram value is the number of indexes been found
    end               
    Classhist=Classhist./sum(Classhist);                                          % Normalize the histogram
    Hist(i,:) = Classhist;                                                        % Combine it in the large histogram matrix
end
%% Test
Struct_test= dir(strcat('C:\Users\pc\Desktop\作业\ELEC 345\Assignment\HW 2\midterm_data\midterm_data_expanded\','TestDataset\'));    % read in the test data directory
Struct_test(1)=[];
Struct_test(1)=[];
[numImage, ~] = size(Struct_test);            % get the number of test images
confusion = zeros(25,25);                     % create the confusion matrix for future use
for i = 1: numImage                           % for loop of each test image
    file = Struct_test(i).name;               % read its name
    Image_test = im2double(imread(strcat('C:\Users\pc\Desktop\作业\ELEC 345\Assignment\HW 2\midterm_data\midterm_data_expanded\','TestDataset\', file)));    % input the image to Matlab
    if size(size(Image_test),2) == 3                % check if not graysacle
        Image_test=rgb2gray(Image_test);            % if not, convert to grayscale
        %[~,d_test] =vl_sift(single(rgb2gray(Image_test)));
    %else
        %Image_test=single(Image_test);
        %[~,d_test] = vl_sift(single(Image_test));
    end
     pts = detectSURFFeatures(Image_test,'NumOctaves',3,'NumScaleLevels',5);     % call matlab built in SURF function
    [d_test,~] = extractFeatures(Image_test,pts);                                % call matlab built in extract function
    %[~,d_test] = pca(double(d_test'),'numcomponents',20);
 

    %[test_match_idx, test_dist] = vl_kdtreequery(forest,centroids, d_test');
    [Class_test_IDX,Class_d_test]=knnsearch(Center',double(d_test));             % call knnsearch function
    Classthresh_test = max(Class_d_test) * threshpercent;                        % apply threshold
    Class_test_IDX(Class_d_test > Classthresh_test) = 0;                         % apply threshold
    Classhist_test=[]; 
    for n=1:length(Center)                                    % for loop of each cluster
        j=find(Class_test_IDX==n);                            % find all the descriptor index that belongs to the ith cluster 
        if isempty(j)                                         % if empty set
            Classhist_test=[Classhist_test 0];                % set it to zero
        end                                                   % if not empty
        Classhist_test(n)=length(j);                          % put the total number of these descriptors to the value of Buddhahist_test(i) 
    end
    Classhist_test=Classhist_test./sum(Classhist_test);      % Normalize the histogram
    Class = knnsearch(Hist, double(Classhist_test));%,'Distance','euclidean');    % call knnsearch function to assign class
    Real = find(strcmp(strtok(Struct_test(i).name, '_'), which_Class));           % find out the real class index for this certain test image
    confusion(Real,Class) =confusion(Real,Class) + 1;                             % add one to the (real,predicted) class of this image
end
for i = 1:25                                                 % for all 25 classes
    confusion(i,:) = confusion(i,:)/sum(confusion(i,:));     % normalize each row to make it sum up to one
end
rateAvg = sum(diag(confusion))/25;                           % calculate the average of the diagonal of the confusion matrix. (average accuracy of the algorithm)