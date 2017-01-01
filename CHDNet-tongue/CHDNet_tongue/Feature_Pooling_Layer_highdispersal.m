function f = Feature_Pooling_Layer_localresponse(CHDNet,OutputImg,ImgIndex)
addpath('./Utils')


F_vet = [];

NumImg = max(ImgIndex);
f = cell(NumImg,1);
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
           %==================================high dispersal=================================
           F_vet = [F_vet,vec(F{i})/norm(vec(F{i}))];%normalization by row            
        
    end       
   F_vet = F_vet/norm(F_vet);%normalization by column
   deta = sqrt(size(F_vet,1)*size(F_vet,2));%scale factor
   F_vet = deta * F_vet;
    %%==================================local response normalization=================================
%     for i = 1: NumSameImg
%         k = max(1,ceil(i - 5/2));
%         k_end = min(NumSameImg,i + ceil(5/2));
%         F{i} = sigm(F_vet(:,i)./(2+10^(-4)*sum(F_vet(:,k:k_end),2).^0.75));
%     end
    f{Index} = vec(F_vet);
    f{Index} = sparse(f{Index});
end
%clear F{i};
f = [f{:}];










