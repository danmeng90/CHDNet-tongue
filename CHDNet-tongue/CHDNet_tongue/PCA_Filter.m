function V = PCA_Filter(InputImg, PatchSize, NumFilters) 
addpath('./Utils')
a = 1;
NumInputImg = length(InputImg);
NumChannel = size(InputImg{1},3);
S = zeros(NumChannel*PatchSize^2,NumChannel*PatchSize^2);

for i = 1:NumInputImg
    im = im2col_mean_removal(InputImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    S = S + im*im'; % sum of all the input images' covariance matrix
    %S = a*sqrt(S.^2 + 1e-8)+(1-a)*log(1+exp(sqrt(im*im'))); %soft-absolute function as the active functioin
    S = sqrt(S.^2 + 1e-8);
end
S = S/(NumInputImg*size(im,2));
[E D] = eig(S);
[~, ind] = sort(diag(D),'descend');
V = E(:,ind(1:NumFilters));  % principal eigenvectors 




 



