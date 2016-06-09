function plotDictionary(x,dictionary,axisValues,lineWidth,fontSize,YtickStep,mu,legendArray,titleText)


stairs(x,bsxfun(@rdivide,dictionary,sum(dictionary)),'LineWidth',lineWidth);
set(gca,'FontSize',fontSize)
set(gca,'box','on')
axis(axisValues)
set(gca,'yTick',axisValues(3):YtickStep:axisValues(4))
set(gca,'xTick',mu)
set(gca,'defaulttextinterpreter','latex');
legend(legendArray,'interpreter','latex');
title(titleText)
end