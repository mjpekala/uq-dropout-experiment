% POSTPROC_CIFAR10  Evaluates result of a CIFAR-10 experiment.
%
%

load('Deploy.mat');  % creates 'X', 'y', 'Prob'
X = permute(X, [3, 4, 2, 1]);

tau = 50; % TODO: choose this properly

classes = {'plane', 'auto', 'bird', 'cat', 'deer', ...
           'dog', 'frog', 'horse', 'ship', 'truck'};

Mu = mean(Prob,3);
[~,yHatOneBased] = max(Mu,[],2);

% correct for fact that matlab is 1-indexed 
yHat = yHatOneBased - 1;

acc = 100*sum(yHat == y) / numel(y);

C = confusionmat(double(y), yHat)
figure; imagesc(C);
set(gca, 'YTick', 1:10, 'YTickLabel', classes);
set(gca, 'XTick', 1:10, 'XTickLabel', classes);
xlabel('predicted class');
ylabel('true class');
title(sprintf('CIFAR-10 confusion matrix; acc=%0.2f', acc));


% calculate some measure of uncertainty for each prediction
variance = zeros(size(yHat));
for ii = 1:size(Prob,1)
    Pii = squeeze(Prob(ii,:,:));
    samps = Pii(yHatOneBased(ii),:);
    variance(ii) = var(samps);
end

muGap = zeros(size(yHat));
for ii = 1:size(Mu,1)
    ordered = sort(Mu(ii,:), 'descend');
    muGap(ii) = ordered(1) - ordered(2);
end




%-------------------------------------------------------------------------------
% Visualize different ways of characterizing uncertainty
%-------------------------------------------------------------------------------
% compare distribution of max(mu) for correct and incorrect examples.
figure;
boxplot(1 - max(Mu,[],2), yHat==y);
set(gca, 'XTickLabel', {'incorrect', 'correct'});
ylabel('1 - max(\mu)');
title('Uncertainty as 1-max(mu)');
grid on;


figure;
boxplot(1 - muGap, yHat==y);
set(gca, 'XTickLabel', {'incorrect', 'correct'});
ylabel('1 - (largest(\mu) - second_largest(mu))');
title('Uncertainty as 1 - muGap');
grid on;


figure;
boxplot(variance, yHat==y);
set(gca, 'XTickLabel', {'incorrect', 'correct'});
ylabel('var');
title('Uncertainty as var from predicted class');
grid on;



%-------------------------------------------------------------------------------
% Visualize a few covariance matrices
%-------------------------------------------------------------------------------
for ii = 1:2
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

    pause()
end
