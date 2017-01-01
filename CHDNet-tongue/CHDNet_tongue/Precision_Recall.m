%predictTestLabels为模型预测的标签，TestLabels为真实标签

TP = 0; % True Positive （真正, TP）被模型预测为正的正样本；可以称作判断为真的正确率
TN = 0; % True Negative（真负 , TN）被模型预测为负的负样本 ；可以称作判断为假的正确率
FP = 0; % False Positive （假正, FP）被模型预测为正的负样本；可以称作误报率
FN = 0; % False Negative（假负 , FN）被模型预测为负的正样本；可以称作漏报率

for i = 1:length(TestLabels)
    if predictTestLabels(i) == 1 && TestLabels(i) ==1
        TP = TP + 1;
    elseif predictTestLabels(i) == 0 && TestLabels(i) ==0
        TN = TN + 1;
    elseif predictTestLabels(i) == 1 && TestLabels(i) ==0
        FP = FP + 1;
    elseif predictTestLabels(i) == 0 && TestLabels(i) ==1
        FN = FN + 1;
    end
end

TPR = TP /(TP + FN); %True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）:正样本预测结果数 / 正样本实际数 
TNR = TN /(TN + FP); %True Negative Rate（真负率 , TNR）或特指度（specificity）: 负样本预测结果数 / 负样本实际数。明显的这个和召回率是对应的指标，只是用它在衡量类别0 的判定能力。
FPR = FP /(FP + TN); %False Positive Rate （假正率, FPR）:被预测为正的负样本结果数 /负样本实际数
FNR = FN /(TP + FN); %False Negative Rate（假负率 , FNR）:被预测为负的正样本结果数 / 正样本实际数

Precision = TP /(TP + FP); %反映了被分类器判定的正例中真正的正例样本的比重 
Recall = TP / (TP + TN); %反映了被正确判定的正例占总的正例的比重
Accuracy = (TP + TN) / (TP + TN + FP + FN); %反映了分类器统对整个样本的判定能力——能将正的判定为正，负的判定为负
g = TPR*TNR;
F = 2 * Recall * Accuracy / (Recall + Accuracy); %F = 2 *  召回率 *  准确率/ (召回率+准确率)

% fprintf('\n    False Positive Rate: %.2f%%', 100*FPR);
% fprintf('\n    False Negative Rate: %.2f%%', 100*FNR);
% 
fprintf('\n    Accuracy: %.2f%%', 100*Accuracy);
fprintf('\n    Sensitivity: %.2f%%', 100*TPR);
fprintf('\n    Specificity: %.2f%%', 100*TNR);
%fprintf('\n    Precision: %.2f%%', 100*Precision);
%fprintf('\n    Recall: %.2f%%', 100*Recall);
%fprintf('\n    g: %.2f%%', 100*g);
fprintf('\n    F1 measure: %.2f%%\n', 100*F);
