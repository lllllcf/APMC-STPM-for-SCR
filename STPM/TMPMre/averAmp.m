clear all
close all
data0 = importdata('amp_data/amp0.txt');
data0 = data0.data;
x0 = data0(:,1);
y0 = data0(:,2);

y0 = y0((find(x0==4.19):find(x0==4.391)),:);

data90 = importdata('amp_data/amp90.txt');
data90 = data90.data;
x90 = data90(:,1);
y90 = data90(:,2);
y90 = y90((find(x90==4.19):find(x90==4.391)),:);

data180 = importdata('amp_data/amp180.txt');
data180 = data180.data;
x180 = data180(:,1);
y180 = data180(:,2);
y180 = y180((find(x180==4.19):find(x180==4.391)),:);

data270 = importdata('amp_data/amp270.txt');
data270 = data270.data;
x270 = data270(:,1);
y270 = data270(:,2);
y270 = y270((find(x270==4.19):find(x270==4.391)),:);

dataoff = importdata('amp_data/amp_off.txt');
dataoff = dataoff.data;
xoff = dataoff(:,1);
yoff = dataoff(:,2);
yoff = yoff((find(xoff==4.19):find(xoff==4.391)),:);

bar = (4.391-4.19)/length(y0);
s = sum(sum(bar.*( -yoff+(y0+y90+y180+y270)./4 )));
avg = s/(4.391-4.19);
disp(avg);



% bar = (4.391-4.19)/length(y0);
% s = sum(sum(bar.*(  (calR(y0)/calR(yoff) + calR(y90)/calR(yoff) + calR(y180)/calR(yoff) + calR(y270)/calR(yoff))/4 )));
% avg = s/(4.391-4.19);
% disp(avg);
% 
% 
% function res = calR(db)
%     res = exp(db/10*log(10));
% end



