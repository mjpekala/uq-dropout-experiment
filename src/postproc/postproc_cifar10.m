% POSTPROC_CIFAR10  Evaluates result of a CIFAR-10 experiment.
%
%


classes = {'plane', 'auto', 'bird', 'cat', 'deer', ...
           'dog', 'frog', 'horse', 'ship', 'truck'};

load('Deploy.mat');  % creates 'X', 'y', 'Prob'
X = permute(X, [3, 4, 2, 1]);  % python -> matlab canonical ordering
tau = 50; % TODO: choose this properly

%-------------------------------------------------------------------------------
% Measures of uncertainty
%-------------------------------------------------------------------------------
% See the october version of Gal&Ghahramani for the
% variation ratio discussion.
[~,ArgMax] = max(Prob,[], 2);  ArgMax = squeeze(ArgMax);
yHatOneBased = mode(ArgMax, 2);
yHat = yHatOneBased - 1;  % the -1 is because y \in [0,9]

variationRatio = 1 - sum(bsxfun(@eq, ArgMax, yHatOneBased), 2) / size(ArgMax,2);



%-------------------------------------------------------------------------------
% Some standard classification metrics
%-------------------------------------------------------------------------------
acc = 100*sum(yHat == y) / numel(y);
C = confusionmat(double(y), yHat)
figure; imagesc(C);
set(gca, 'YTick', 1:10, 'YTickLabel', classes);
set(gca, 'XTick', 1:10, 'XTickLabel', classes);
xlabel('predicted class');
ylabel('true class');
title(sprintf('CIFAR-10 confusion matrix; acc=%0.2f', acc));



%-------------------------------------------------------------------------------
% Visualize uncertainty
%-------------------------------------------------------------------------------
figure;
subplot(3,1,1);
%boxplot(variationRatio, y==2, 'labels', {'y != auto', 'y == auto'});
boxplot(variationRatio, y, 'labels', classes);
%
subplot(3,1,2);
hist(variationRatio);
title('variation ratio - CIFAR-10 all test data');
xlim([0 1]);
%
subplot(3,1,3);
hist(variationRatio(y==2));
xlim([0 1]);
title('variation ratio - CIFAR-10, y=auto test data');


idx = (variationRatio >= .1);
accHighVR = 100 * sum(yHat(idx) == y(idx)) / sum(idx);
fprintf('[%s]: accuracy on objects with high variation ratio: %0.2f%%\n', ...
        mfilename, accHighVR); 

figure;
boxplot(variationRatio, y==yHat, 'labels', {'incorrect', 'correct'});
title('Variation Ratio');


plot_cifar_example(X, y, ArgMax, 1);
return;


%-------------------------------------------------------------------------------
% calculate some measures of uncertainty for each prediction
%-------------------------------------------------------------------------------

% look at the variance of the predicted class
variance = zeros(size(yHat));
for ii = 1:size(Prob,1)
    Pii = squeeze(Prob(ii,:,:));
    samps = Pii(yHatOneBased(ii),:);
    variance(ii) = var(samps);
end

% difference between maximum mean and second-largest mean.
muGap = zeros(size(yHat));
for ii = 1:size(Mu,1)
    ordered = sort(Mu(ii,:), 'descend');
    muGap(ii) = ordered(1) - ordered(2);
end




%-------------------------------------------------------------------------------
% Visualize different ways of characterizing uncertainty
%-------------------------------------------------------------------------------

figure;
plot(muMax, variance, 'o');
xlabel('mean of class estimate');
ylabel('variance of class estimate');
title('uncertainty measures; mean and variance');



figure;
plot(muMax, muGap, 'o');
xlabel('mean of class estimate');
ylabel('mu gap');
title('uncertainty measures based only on the mean');

figure;
plot(muGap, variance, 'o');



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
plot_cifar_example(X, y, Prob, 1);
plot_cifar_example(X, y, Prob, 2);


%-------------------------------------------------------------------------------
% Combined measures of uncertainty
%-------------------------------------------------------------------------------

% for each test example, let's compare the variance relative to
% other test examples with similar means
muRank = zeros(size(y));
for ii = 1:length(muRank)
    nbrs = abs(muMax - muMax(ii)) < .05;
    nbrs(ii) = 0;
    muRank(ii) = sum(variance(nbrs) > variance(ii)) / sum(nbrs);
end

idx1 = (muRank < .1);
%idx1 = (muRank < .1) & (y ~= 2);
fprintf('[%s] Accuracy for low-rank test examples: %0.2f%%\n', ...
        mfilename, 100*sum(y(idx1) == yHat(idx1)) / sum(idx1));

%idx2 = (muMax < quantile(muMax, .1));
%idx2 = (muMax < quantile(muMax, .1)) & (y ~= 2);
idx2 = (muRank >= quantile(muRank, .9));
fprintf('[%s] Accuracy for low-mean test examples: %0.2f%%\n', ...
        mfilename, 100*sum(y(idx2) == yHat(idx2)) / sum(idx2));

figure;
plot(muMax, variance, 'bo', ...
     muMax(idx1), variance(idx1), 'ro');
xlabel('mean of class estimate');
ylabel('variance of class estimate');

figure;
plot(muMax, variance, 'bo', ...
     muMax(idx2), variance(idx2), 'ro');
xlabel('mean of class estimate');
ylabel('variance of class estimate');


figure;
subplot(1,2,1);
boxplot(muRank, y==yHat);
set(gca, 'XTickLabel', {'incorrect', 'correct'});
ylabel('muRank');

subplot(1,2,2);
plot(muMax(y == yHat), variance(y == yHat), 'bo', ...
     muMax(y ~= yHat), variance(y ~= yHat), 'rx');
xlabel('mean of class estimate');
ylabel('variance of class estimate');
legend('correct', 'incorrect');
