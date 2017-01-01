function [f V] = CHDNet_train(InputImg,CHDNet)

addpath('./Utils')
if length(CHDNet.NumFilters)~= CHDNet.NumStages;
    display('Length(CHDNet.NumFilters)~=CHDNet.NumStages')
    return
end

NumImg = length(InputImg);

V = cell(CHDNet.NumStages,1);
OutputImg = InputImg;
ImgIndex = (1:NumImg)';
clear InputImg;

for stage = 1:CHDNet.NumStages
    display(['Computing PCA filter bank at stage ' num2str(stage) '...'])
    V{stage} = PCA_Filter(OutputImg, CHDNet.PatchSize(stage), CHDNet.NumFilters(stage)); % compute PCA filters
    if stage ~= CHDNet.NumStages % compute the PCA outputs only when it is NOT the last stage
        display(['Computing convolution and non-linear transformation layer at stage ' num2str(stage) '...'])
        [OutputImg, ImgIndex] = CHDNet_Conv_NonLinear(OutputImg, ImgIndex, CHDNet.PatchSize(stage), CHDNet.NumFilters(stage), V{stage});
    end
end

display(['Computing convolution and non-linear transformation layer at stage ' num2str(stage) '...'])
f = cell(NumImg,1); % compute the CHDNet training feature one by one
for idx = 1:NumImg
    if(0==mod(idx,NumImg/100));
        display(['Extract feature for ' num2str(idx) 'th image...'])
    end
    OutputImgIndex = ImgIndex==idx; % select feature maps corresponding to image "idx"
    [OutputImg_k, ImgIndex_k] = CHDNet_Conv_NonLinear(OutputImg(OutputImgIndex), ones(sum(OutputImgIndex),1),CHDNet.PatchSize(end), CHDNet.NumFilters(end), V{end});  % compute the last PCA outputs of image "idx"
    
    f{idx} = Feature_Pooling_Layer(CHDNet,OutputImg_k,ImgIndex_k); % compute the feature of image "idx"
    %OutputImg(OutputImgIndex) = cell(sum(OutputImgIndex),1);    
end
f = sparse([f{:}]);    
OutputImg = [];








