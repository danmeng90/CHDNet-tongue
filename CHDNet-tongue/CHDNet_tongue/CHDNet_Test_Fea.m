function f = CHDNet_FeaExt(InputImg,V,CHDANet)
addpath('./Utils')

if length(CHDANet.NumFilters)~= CHDANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InputImg);

OutputImg = InputImg; 
ImgIndex = (1:NumImg)';
clear InputImg;
for stage = 1:CHDANet.NumStages
     [OutputImg ImgIndex] = CHDNet_Conv_NonLinear(OutputImg, ImgIndex, CHDANet.PatchSize(stage), CHDANet.NumFilters(stage), V{stage});  
end
f = Feature_Pooling_Layer(CHDANet,OutputImg,ImgIndex);





