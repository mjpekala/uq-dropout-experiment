% POSTPROC  Evaluates result of a CIFAR-10 experiment.

load('Deploy.mat');  % creates 'X', 'y', 'Prob'
X = permute(X, [3, 4, 2, 1]);

tau = 50; % TODO: choose this properly

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
xlabel('predicted class');
ylabel('true class');
title(sprintf('CIFAR-10 confusion matrix; acc=%0.2f', acc));


muMax = max(Mu, [], 2);



%-------------------------------------------------------------------------------
% Visualize a few covariance matrices
%-------------------------------------------------------------------------------
for ii = 1:20
    Xi = squeeze(Prob(ii,:,:));   %  (#classes x #samples)
    Xi = Xi';                     %  -> rows-as-examples
    Cov = cov(Xi);
    Cov = eye(size(Cov)) / tau + Cov;

    
    figure('Position', [200, 200, 400, 1200]);
    ha = tight_subplot(3,1, [.03, .03]);
    
    axes(ha(1));
    imagesc(X(:,:,:,ii));
    title(sprintf('Example %d; class=%s', ii, classes{y(ii)+1}));
   
    axes(ha(2));
    boxplot(Xi);
    hold on;
    plot(1:10, mean(Xi,1), 'ro');
    hold off;
    set(gca, 'XTick', 1:10, 'XTickLabel', classes);
   
    axes(ha(3));
    imagesc(Cov); colorbar;
    set(gca, 'YTick', 1:10, 'YTickLabel', classes);
    set(gca, 'XTick', 1:10, 'XTickLabel', classes);
end
