clear all
close all
data0 = importdata('amp_data/amp0.txt');
data0 = data0.data;
x0 = data0(:,1);
y0 = data0(:,2);

data90 = importdata('amp_data/amp90.txt');
data90 = data90.data;
x90 = data90(:,1);
y90 = data90(:,2);

data180 = importdata('amp_data/amp180.txt');
data180 = data180.data;
x180 = data180(:,1);
y180 = data180(:,2);

data270 = importdata('amp_data/amp270.txt');
data270 = data270.data;
x270 = data270(:,1);
y270 = data270(:,2);

dataoff = importdata('amp_data/amp_off.txt');
dataoff = dataoff.data;
xoff = dataoff(:,1);
yoff = dataoff(:,2);


figure(1)
hold on
box on
set(gca,'linewidth',1.5,'FontSize',13,'FontWeight','bold')
grid on
plot(x0, y0,'-','LineWidth',1.5,'Color',[1 0 0]);

plot(x90, y90,'-','LineWidth',1.5,'Color',[0 1 0]);
plot(x180, y180,'-','LineWidth',1.5,'Color',[0 0 1]);
plot(x270, y270,'-','LineWidth',1.5,'Color',[237, 124, 50]/255);
plot(xoff, yoff,'--','LineWidth',1.5,'Color',[167,28,227]/255);

alpha = 0.2;
re = rectangle('Position',[4.211,-69.75, 0.008, 69.35],'FaceColor',[255 192 203 255*alpha]/255,'EdgeColor',[255 192 203 255*alpha]/255);
% pp;

xlim([4 4.5]);
xlabel('\textbf{Frequency (GHz)}','Interpreter','Latex','FontSize',15);
ylabel('\textbf{Normalized Amplitude (dB)}','Interpreter','Latex','FontSize',15);
legend({'$\phi = 0^{\circ}$','$\phi = 90^{\circ}$',...
    '$\phi = 180^{\circ}$','$\phi = 270^{\circ}$','OFF'},'FontSize',12,'Interpreter','Latex')
