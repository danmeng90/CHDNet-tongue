%predictTestLabelsΪģ��Ԥ��ı�ǩ��TestLabelsΪ��ʵ��ǩ

TP = 0; % True Positive ������, TP����ģ��Ԥ��Ϊ���������������Գ����ж�Ϊ�����ȷ��
TN = 0; % True Negative���渺 , TN����ģ��Ԥ��Ϊ���ĸ����� �����Գ����ж�Ϊ�ٵ���ȷ��
FP = 0; % False Positive ������, FP����ģ��Ԥ��Ϊ���ĸ����������Գ�������
FN = 0; % False Negative���ٸ� , FN����ģ��Ԥ��Ϊ���������������Գ���©����

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

TPR = TP /(TP + FN); %True Positive Rate�������� , TPR���������ȣ�sensitivity��:������Ԥ������ / ������ʵ���� 
TNR = TN /(TN + FP); %True Negative Rate���渺�� , TNR������ָ�ȣ�specificity��: ������Ԥ������ / ������ʵ���������Ե�������ٻ����Ƕ�Ӧ��ָ�ֻ꣬�������ں������0 ���ж�������
FPR = FP /(FP + TN); %False Positive Rate ��������, FPR��:��Ԥ��Ϊ���ĸ���������� /������ʵ����
FNR = FN /(TP + FN); %False Negative Rate���ٸ��� , FNR��:��Ԥ��Ϊ��������������� / ������ʵ����

Precision = TP /(TP + FP); %��ӳ�˱��������ж������������������������ı��� 
Recall = TP / (TP + TN); %��ӳ�˱���ȷ�ж�������ռ�ܵ������ı���
Accuracy = (TP + TN) / (TP + TN + FP + FN); %��ӳ�˷�����ͳ�������������ж����������ܽ������ж�Ϊ���������ж�Ϊ��
g = TPR*TNR;
F = 2 * Recall * Accuracy / (Recall + Accuracy); %F = 2 *  �ٻ��� *  ׼ȷ��/ (�ٻ���+׼ȷ��)

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
