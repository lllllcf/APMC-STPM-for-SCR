clear all
close all
data0 = importdata('phase_data/phase0.txt');
data0 = data0.data;
x0 = data0(:,1);
y0 = data0(:,2);

data90 = importdata('phase_data/phase90.txt');
data90 = data90.data;
x90 = data90(:,1);
y90 = data90(:,2);

data180 = importdata('phase_data/phase180.txt');
data180 = data180.data;
x180 = data180(:,1);
y180 = data180(:,2);

data270 = importdata('phase_data/phase270.txt');
data270 = data270.data;
x270 = data270(:,1);
y270 = data270(:,2);

figure(1)
hold on
box on
set(gca,'linewidth',1.5,'FontSize',13,'FontWeight','bold')
grid on

plot(x0, y0-180,'-','LineWidth',1.5,'Color',[1 0 0]);
plot(x90, y90+360-180,'-','LineWidth',1.5,'Color',[0 1 0]);
plot(x180, y180+360-180,'-','LineWidth',1.5,'Color',[0 0 1]);
plot(x270, y270+360-180,'-','LineWidth',1.5,'Color',[237, 124, 50]/255);

a = abs(y0-180 - (y90+360-180));
b = abs(y90+360-180 - (y180+360-180));
c = abs(y180+360-180 - (y270+360-180));

plot(x0,abs(a + b + c - 3*90)/3,'-','LineWidth',3);

%  plot(x0, -(y180-y270),'--','LineWidth',1.5);
%  plot(x90, -(y270-y0),'--','LineWidth',1.5);
%  plot(x180, y90+360-y0,'--','LineWidth',1.5);
%  fplot(90,'LineWidth',2);
% pp;

alpha = 0.2;
re = rectangle('Position',[4.211,-180, 0.008, 540],'FaceColor',[255 192 203 255*alpha]/255,'EdgeColor',[255 192 203 255*alpha]/255);
% pp;

xlim([4 4.5]);
ylim([-180 360]);
xlabel('\textbf{Frequency (GHz)}','Interpreter','Latex','FontSize',15);
ylabel('\textbf{Phase Shift ($^{\circ}$)}','Interpreter','Latex','FontSize',15);
legend({'$\phi = 0^{\circ}$','$\phi = 90^{\circ}$',...
    '$\phi = 180^{\circ}$','$\phi = 270^{\circ}$'},'FontSize',12,'Interpreter','Latex')
