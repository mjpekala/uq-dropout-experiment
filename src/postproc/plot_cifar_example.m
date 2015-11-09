function plot_cifar_example(X, y, Prob, idx0)
    
if nargin < 4, idx0 = 1; end
    
% The standard CIFAR-10 classes
classes = {'plane', 'auto', 'bird', 'cat', 'deer', ...
           'dog', 'frog', 'horse', 'ship', 'truck'};

tau = 1; % TODO: set this properly

fig = figure('Position', [200, 200, 800, 800]);
ha = tight_subplot(2,2, [.03, .03]);

sId = uicontrol('Style', 'slider', ...
                'Min', 1, 'Max', length(y), ...
                'Value', idx0, ...
                'SliderStep', [1 / (length(y)-1) 1], ...
                'Units', 'Normalized', ...
                'Position', [.05 .95 .35 .04], ...
                'Callback', @slider_cb);

redraw(idx0);


function slider_cb(source, callbackdata)
  val = get(source, 'Value');
  redraw(round(val));
end  % slider_cb()


function redraw(idx)
    Xi = squeeze(Prob(idx,:,:));   %  (#classes x #samples)
    Xi = Xi';                     %  -> rows-as-examples
    Cov = cov(Xi);
    Cov = eye(size(Cov)) / tau + Cov;
    
    [~,muOrdered] = sort(mean(Xi,1), 'descend');

    figure(fig);
    cla(ha(1));  cla(ha(2)); cla(ha(3));
    
    axes(ha(1));
    imagesc(X(:,:,:,idx));
    title(sprintf('Example %d; class=%s', idx, classes{y(idx)+1}));
   
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
    
    axes(ha(4));
    hist(Xi(:,muOrdered(1)));
    legend(sprintf('scores for y=%d', muOrdered(1)), 'Location', 'NorthWest');
    
end % redraw()


end % plot_cifar_example()