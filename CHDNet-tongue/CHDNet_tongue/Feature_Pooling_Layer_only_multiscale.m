function f = Feature_Pooling_Layer_only_multiscale(CHDNet,OutputImg,ImgIndex)
addpath('./Utils')


f=[];

NumImg = max(ImgIndex);
%f = cell(NumImg,1);
fea_map_w = 2.^((CHDNet.NumFilters(end)-1):-1:0); % weights for binary to decimal conversion
H_temp = [];

for Index = 1:NumImg
    Index_span = find(ImgIndex == Index);
    NumSameImg = length(Index_span)/CHDNet.NumFilters(end); % the number of feature maps belong to the same image
    F = cell(NumSameImg,1);
    for i = 1:NumSameImg
        H = 0;
        ImgSize = size(OutputImg{Index_span(CHDNet.NumFilters(end)*(i-1) + 1)});
        for j = 1:CHDNet.NumFilters(end)
            H_temp = sign(OutputImg{Index_span(CHDNet.NumFilters(end)*(i-1)+j)});
            H_temp(H_temp<=0) = 0;
            H = H + fea_map_w(j)*H_temp;
            OutputImg{Index_span(CHDNet.NumFilters(end)*(i-1)+j)} = [];
        end
        %H = (H - min(min(H)))/(max(max(H))-min(min(H)))*255;
        %         if isempty(CHDNet.HistBlockSize)
        %             NumBlk = ceil((CHDNet.ImgBlkRatio - 1)./CHDNet.BlkOverLapRatio) + 1;
        %             HistBlockSize = ceil(size(H)./CHDNet.ImgBlkRatio);
        %             OverLapinPixel = ceil((size(H) - HistBlockSize)./(NumBlk - 1));
        %             NImgSize = (NumBlk-1).*OverLapinPixel + HistBlockSize;
        %             Tmp = zeros(NImgSize);
        %             Tmp(1:size(H,1), 1:size(H,2)) = H;
        %             F{i} = sparse(histc(im2col_general(Tmp,HistBlockSize,OverLapinPixel),(0:2^CHDNet.NumFilters(end)-1)'));
        %         else
        stride = CHDNet.HistBlockSize;
        F_j = sparse(histc(im2col_general(H,CHDNet.HistBlockSize,stride),(0:2^CHDNet.NumFilters(end)-1)'));
        if ~isempty(CHDNet.Pyramid)
            F_j = Multiscale_Fea_Analysis(F_j, ImgSize,stride, CHDNet)';
        end
        F{i} = F_j;
        if ~isempty(CHDNet.Pyramid)
            F{i} = sparse(F{i}/norm(F{i}));
        end
        f = [f,vec(F{i})'];
    end
    
    
end











