function f = Multiscale_Fea_Analysis(F, ImgSize, stride, CHDNet)
img_width = ImgSize(2);
img_height = ImgSize(1);
x_start = ceil(CHDNet.HistBlockSize(2)/2);
y_start = ceil(CHDNet.HistBlockSize(1)/2);
x_end = floor(img_width - CHDNet.HistBlockSize(2)/2);
y_end = floor(img_height - CHDNet.HistBlockSize(1)/2);
sam_coordinate = [kron(x_start:stride:x_end,ones(1,length(y_start:stride: y_end)));kron(ones(1,length(x_start:stride:x_end)),y_start:stride: y_end)];
[FSize, ~] = size(F);
% spatial levels
G = length(CHDNet.Pyramid);
diction_word = CHDNet.Pyramid.^2;
tBins = sum(diction_word);
f = zeros(FSize, tBins);
countNum = 0;

for j = 1:G,
    Num_Bins = diction_word(j);
    wUnit = img_width / CHDNet.Pyramid(j);
    hUnit = img_height / CHDNet.Pyramid(j);
    % find to which spatial bin each local descriptor belongs
    xWord = ceil(sam_coordinate(1,:) / wUnit);
    yWord = ceil(sam_coordinate(2,:) / hUnit);
    IndexWord = (yWord - 1)*CHDNet.Pyramid(j) + xWord;
    for k = 1:Num_Bins,
        countNum = countNum + 1;
        sameIndexBin = find(IndexWord == k);
        if isempty(sameIndexBin),
            continue;
        end
        f(:, countNum) = max(F(:, sameIndexBin), [], 2);
    end
end