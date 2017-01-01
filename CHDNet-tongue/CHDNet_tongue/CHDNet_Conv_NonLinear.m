function [OutputImg, OutputImgIdx] = CHDNet_Conv_NonLinear(InputImg, InputImgIdx, PatchSize, NumFilters, V)
addpath('./Utils')

NumInputImg = length(InputImg);
mag = (PatchSize-1)/2;
OutputImg = cell(NumFilters*NumInputImg,1);
cnt = 0;

for i = 1:NumInputImg
    [ImgX, ImgY, NumChls] = size(InputImg{i});
    img = zeros(ImgX+PatchSize-1,ImgY+PatchSize-1, NumChls);
    img((mag+1):end-mag,(mag+1):end-mag,:) = InputImg{i};
    im = im2col_mean_removal(img,[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    for j = 1:NumFilters
        cnt = cnt + 1;
        OutputImg{cnt} = reshape(V(:,j)'*im,ImgX,ImgY);  % convolution output
    end
    InputImg{i} = [];
end
OutputImgIdx = kron(InputImgIdx,ones(NumFilters,1));





