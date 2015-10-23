% POSTPROC  Evaluates result of a CIFAR-10 experiment.

load('Deploy.mat');  % creates 'y', 'Prob'

classes = {'plane', 'auto', 'bird', 'cat', 'deer', ...
           'dog', 'frog', 'horse', 'ship', 'truck'};

Mu = mean(Prob,3);
[~,yHat] = max(Mu,[],2);

% correct for fact that matlab is 1-indexed 
yHat = yHat - 1;

acc = 100*sum(yHat == y) / numel(y);

C = confusionmat(double(y), yHat)
figure; imagesc(C);
set(gca, 'YTick', 1:10, 'YTickLabel', classes);
set(gca, 'XTick', 1:10, 'XTickLabel', classes);
title(sprintf('CIFAR-10; acc=%0.2f', acc));
xlabel('predicted class');
ylabel('true class');


for ii = 1:20
    Xi = squeeze(Prob(ii,:,:));   %  (#classes x #samples)
    Xi = Xi';                     %  -> rows-as-examples
    Cov = cov(Xi);
  
    figure; imagesc(Xi);
    xlabel('sample'); 
    set(gca, 'XTick', 1:10, 'XTickLabel', classes);
    title(sprintf('Example %d; class=%s', ii, classes{y(ii)+1}));
    colorbar;
    
    figure; imagesc(Cov);
    set(gca, 'YTick', 1:10, 'YTickLabel', classes);
    set(gca, 'XTick', 1:10, 'XTickLabel', classes);
    title('covariance');
end
