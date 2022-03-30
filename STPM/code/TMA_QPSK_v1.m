% this code is the first version that manage to 
% realize TMA+QPSK communication
clear all;
%% define constant
c = 3e8; % light speed

fLO = 50e9;
fs = 1e12;
fp = 1e8;
fd = 1e7;
lambda = c/fLO;
z = lambda/2; % distance between two antennas
arrayNum = 10; % # of antennas
dataLen=40/fd;%data length
fIF=4e7;%IF Carrier Frequency


% pulse parameters
Tp = 1/fp;
tau = Tp/3;
beta = 2*pi/lambda;
delta = Tp/1000;
t = 0:1/fs:dataLen;

Ts=1/fs;%Sampling Period
Fn=fs/2;%Nyquist Frequency
BitNum=dataLen*fd;
A0=1;%IF Carrier Amplitude

%% -------------------------Data Generating in Square Wave-------------------
data=sign(randn(fd*dataLen,1));%Randomly Generated Signed Singal
dataTemp=ones((fs/fd),1)*data';%Dealing with Each Bit
dataPlot=dataTemp(:);%Makes data vector fit the time vector
t = t(1:length(dataPlot));

figure(1)
subplot(3,1,1)
plot(t/1e-6,dataPlot);%Plot the Generated Data
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Generated Input Data, P="+powerlvl(dataPlot));
%% ------------------------Generating I & Q Signal---------------------------
%Noticing that for Input data In and I-Q signal we have relation:
%In:00 01 10 11; respectively; (IQ) (-1-1) (-11) (1-1) (11)
%which means I for the even Bit and Q for the odd Bit

%------Generating I Data
Idata=data(1:2:floor(length(data)/2)*2);%odd
IdataTemp=ones((2*fs/fd),1)*Idata';
IdataPlot=IdataTemp(:);
subplot(3,1,2)
plot(t/1e-6,IdataPlot);%Plot the Generated I Data
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("I Channel Input Data, P="+powerlvl(IdataPlot));
%------Generating Q Data
Qdata=data(2:2:floor(length(data)/2)*2);%even
QdataTemp=ones((2*fs/fd),1)*Qdata';
QdataPlot=QdataTemp(:);
subplot(3,1,3)
plot(t/1e-6,QdataPlot);%Plot the Generated Q Data
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Q Channel Input Data, P="+powerlvl(QdataPlot));
%------IF Carrier
Icarrier=A0.*cos((2*pi*fIF).*t)';
Qcarrier=A0.*sin((2*pi*fIF).*t)';
%------I-Q Modulated Signal with IF Carrier
IIFModulated=Icarrier.*IdataPlot;
QIFModulated=Qcarrier.*QdataPlot;
%------Summed up QPSK Signal
sumIQIF=IIFModulated+QIFModulated;


%% -------TMA && Frequency Up-Converting

sumIQRF_I=sumIQIF.*cos((2*pi*(fLO)).*t)';
sumIQRF_Q=sumIQIF.*cos((2*pi*(fLO)).*t+pi/2)';

% shiftSum = circshift(sumIQIF, -floor(shiftT/(1/fs)));
% shiftT = 1/(4*(fLO));
% sumIQRF_Q=shiftSum.*cos((2*pi*(fLO)).*t+pi/2)';
% sumIQRF_Q = circshift(sumIQRF_I, -floor(shiftT/(1/fs)));

%
% p1 = [];
% p_1 = [];
% p5 = [];
% p_7 = [];
theta = 50; % target location
alpha = 50;

% for alpha = -70:1:70
    delta_t = z*sind(alpha)/c; % time delay of two signal
    delta_tt = delta_t/(1/fs); % transfer the time delay to the array index

    t11 = -Tp/6;
    resSig = sumIQRF_I'.*pulse(t11,tau,t11+Tp/2,Tp,t,fs,delta)- ...
        sumIQRF_Q'.*pulse(t11-Tp/4,tau,t11+Tp/4,Tp,t,fs,delta);
%     a = pulse(t11,tau,t11+Tp/2,Tp,t,fs,delta)-pulse(t11-Tp/4,tau,t11+Tp/4,Tp,t,fs,delta);
    for i=2:arrayNum
        t1i = (((i-1)*beta*z*sind(theta))/pi-1/3)/2*Tp;
        Sigi = sumIQRF_I'.*pulse(t1i,tau,t1i+Tp/2,Tp,t,fs,delta)- ...
            sumIQRF_Q'.*pulse(t1i-Tp/4,tau,t1i+Tp/4,Tp,t,fs,delta);
        resSig = resSig + circshift(Sigi, -floor(delta_tt*(i-1)));
    end

% the following code are used to plot Power vs. alpha
% [f2,FFTsig]=SSBFFT(resSig,fs);
%     p11 = find(f2>(fLO+0.7*fp),1);
%     p12 = find(f2>(fLO+1.3*fp),1);
% %     p1 = [p1,sum(FFTsig(p11:p12).^2)];
%     p1 = [p1,sum(abs(FFTsig(p11:p12)))];
%     
%     p_11 = find(f2>(fLO-1.3*fp),1);
%     p_12 = find(f2>(fLO-0.7*fp),1);
% %     p1 = [p1,sum(FFTsig(p11:p12).^2)];
%     p_1 = [p_1,sum(abs(FFTsig(p_11:p_12)))];
%     
%     p51 = find(f2>(fLO+4.7*fp),1);
%     p52 = find(f2>(fLO+5.3*fp),1);
% %     p5 = [p5,sum(FFTsig(p51:p52).^2)];
%     p5 = [p5,sum(abs(FFTsig(p51:p52)))];
%     
%     p_71 = find(f2>(fLO-7.3*fp),1);
%     p_72 = find(f2>(fLO-6.7*fp),1);
% %     p_7 = [p_7,sum(FFTsig(p_71:p_72).^2)];
%     p_7 = [p_7,sum(abs(FFTsig(p_71:p_72)))];
% end
% alpha = -70:1:70;
% m = max(p1);
% p1 = 20.*log10(p1./m);
% p_1 = 20.*log10(p_1./m);
% p5 = 20.*log10(p5./m);
% p_7 = 20.*log10(p_7./m);
% p1(p1<-35) = -35;
% p_1(p_1<-35) = -35;
% p5(p5<-35) = -35;
% p_7(p_7<-35) = -35;
% figure(11)
% plot(alpha, p1,'-','LineWidth',1.5);
% hold on
% plot(alpha, p_1,'-','LineWidth',1.5);
% plot(alpha, p5,'--','LineWidth',1.5);
% plot(alpha, p_7,'-','LineWidth',1.5);
% xlabel('($\Theta$)deg','Interpreter','Latex');
% ylabel('DB');
% legend({'m=1','m=-1','m=5','m=-7'},'FontSize',12)
% title("Relative Power Pattern at $\Theta^{scann}$ = "+theta+" , $\delta$ = "+ delta/Tp,'Interpreter','Latex');

RFReceived=resSig'; % received signal

figure(5)
subplot(2,1,1)
[fR,FFTRFReceived]=SSBFFT(RFReceived,fs);
plot((fR-fLO)./fp,FFTRFReceived);
grid on
xlabel('FREQUENCY(GHz)');ylabel('AMPLITUDE');
title("(f-fLO)/fp");
xlim([-10 10])
subplot(2,1,2); 
plot((fR-fLO)./fp,20.*log10(abs(FFTRFReceived).^2));
grid on
xlabel('FREQUENCY(GHz)');ylabel('DB');
xlim([-10 10])
%% --------Down-Converting
%Set its Central Frequency as fIF
DownReceived=RFReceived.*cos((2*pi*(fLO+fp)).*t)';


[f3,FFTDownReceived]=SSBFFT(DownReceived,fs);
figure(6)
subplot(2,1,1)
plot(f3./1e9,FFTDownReceived);
grid on
xlabel('FREQUENCY(GHz)');ylabel('AMPLITUDE');
title("Frequency Domain of Signal After Down Converting, P="+powerlvl(DownReceived));
xlim([0 4*fIF/1e9])
subplot(2,1,2)
plot(f3./1e9,FFTDownReceived);
grid on
xlabel('FREQUENCY(GHz)');ylabel('AMPLITUDE');
xlim([0 4*fIF/1e9])

%% --------Filtering-----Lowpass
% RFFiltered=DownReceived;
RFFiltered=lowpass(DownReceived,1e7,fs);
% 
% figure(7)
% subplot(2,1,1)
% [f4,FFTRFFiltered]=SSBFFT(RFFiltered,fs);
% plot(f4./1e9,FFTRFFiltered);
% % xlim([fIF/1e9-2*fp/1e9 fIF/1e9+2*fp/1e9])
% grid on
% xlabel('FREQUENCY(GHz)');ylabel('AMPLITUDE');
% title("Frequency Domain of Signal After LowPass Filtering, P="+powerlvl(RFFiltered));
% subplot(2,1,2)
% [f4,FFTRFFiltered]=SSBFFT(RFFiltered,fs);
% plot(f4./1e9,FFTRFFiltered);
% % xlim([fIF/1e9-2*fp/1e9 fIF/1e9+2*fp/1e9])
% grid on
% xlabel('FREQUENCY(GHz)');ylabel('AMPLITUDE');


%% --------------IQ demodulation
% RFFiltered = sumIQIF;
symNum = fs/(fd);
% phi0 = IQEVM(RFFiltered,A0,fIF,fd,fs,symNum, t);
phi0 = 0;
Icarrier=A0.*cos((2*pi*(fIF)).*t+phi0)';
Qcarrier=A0.*sin((2*pi*(fIF)).*t+phi0)';
IReceived=Icarrier.*RFFiltered;
QReceived=Qcarrier.*RFFiltered;
%--------Integral(Summing) to get I & Q data
Isampled=reshape(IReceived,symNum,floor(length(IReceived)/symNum));
Qsampled=reshape(QReceived,symNum,floor(length(QReceived)/symNum));
Imean=mean(Isampled,1)';
Qmean=mean(Qsampled,1)';
ITemp=ones(symNum,1)*Imean';
II=ITemp(:);
QTemp=ones(symNum,1)*Qmean';
QQ=QTemp(:);

%% Plot I and Q data after Sampling
figure(8)
subplot(2,2,1)
plot(t/1e-6,II);
axis([0 dataLen/1e-6 min(II) max(II)])
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("I Channel Data After Sampling");
subplot(2,2,3)
plot(t/1e-6,QQ);
axis([0 dataLen/1e-6 min(QQ) max(QQ)])
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Q Channel Data After Sampling");
%--------Use a comparator to recover the I & Q
ICompared=sign(II);
QCompared=sign(QQ);
subplot(2,2,2)
plot(t/1e-6,ICompared);
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("I Channel Data");
subplot(2,2,4)
plot(t/1e-6,QCompared);
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Q Channel Data");
%------------------Final Received Data
IComparedTemp=ICompared(1:2*fs/fd:floor(length(ICompared)/(2*fs/fd))*(2*fs/fd));
QComparedTemp=QCompared(1:2*fs/fd:floor(length(ICompared)/(2*fs/fd))*(2*fs/fd));
sumCompared=[IComparedTemp QComparedTemp]';
sumCompared=sumCompared(:);
sumCompared=ones(fs/fd,1)*sumCompared';
sumCompared=sumCompared(:);
figure(9)
subplot(2,1,1)
plot(t/1e-6,sumCompared);
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Final Received Data, P="+powerlvl(sumCompared));
subplot(2,1,2)
plot(t/1e-6,sumCompared-dataPlot);
BitErrorRate=sum(abs(sumCompared-dataPlot))/(fs*dataLen)/2;
ylim([-1.5*A0 1.5*A0]);
xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
title("Difference between Data Received and Input, P="+powerlvl(sumCompared-dataPlot));
%% -----debug the sig
x = IIFModulated+QIFModulated;
figure(10)
% xlim([1e3 3e3])
hold on
plot(t,x,'LineWidth',2)
plot(t,RFFiltered)
% xlim([0 5e-6])
% plot(t,a,'LineWidth',1.5);
legend({'IQsig','RFFiltered'},'FontSize',12)

%% ========================FFT Function=====================================
function [f1,FFTSSB]=SSBFFT(signal1,fs)
L=length(signal1);
NFFY=2^nextpow2(L);
%To make FFT efficient, we want NFFY to be the 2's integer's power.
FFTSSB=fft(signal1,NFFY);%to last several terms pad with zeros
%To divide the spectrum into 2 while taking the DC term into considering 
NumUniquePts=ceil((NFFY+1)/2);
FFTSSB=FFTSSB(1:NumUniquePts);
FFTSSB=abs(FFTSSB)*2/L;%*2 for SSB, /length for normalize
%Cut the DC part and the last Part since they belongs to both sides
FFTSSB(1)=FFTSSB(1)/2;FFTSSB(end)=FFTSSB(end)/2;
%for SSB, divide Sampling Rate by 2
f1=(fs/2*linspace(0,1-1/NumUniquePts,NumUniquePts))';
end
function power1=powerlvl(signal1)
power1=sum(abs(signal1).^2)/length(signal1);
end
%% Power calculation
function res = sigPower(sig, f)
    power = sig.^2.*(1/f);
    res = sum(power);
end

%% pulse function
function res=pulse(t1,tau,t2,Tp,totalTime,fs,delta)
    % repmat
    temp = 0:1/fs:delta;
    slope = temp/delta;
    len = length(slope);
    
    pp(1:len) = slope;
    pp(floor((delta)*fs)+1 : floor((tau)*fs)+1) = 1;
    
    
    pp(floor((tau)*fs)+1 : floor((tau)*fs)+len) = 1-slope;
    pp(floor((tau+delta)*fs)+1 : floor((t2-t1)*fs)+1) = 0;
%     % debug
%     floor((tau+delta)*fs)+1
%     floor((Tp/2)*fs)+1
%     floor((t2-t1)*fs)+1
       
    pp(floor((t2-t1)*fs)+1 : floor((t2-t1)*fs)+len) = -slope;   
    pp(floor((t2-t1+delta)*fs)+1 : floor((t2-t1+tau)*fs)+1) = -1;
       
    
    pp(floor((t2-t1+tau)*fs)+1 : floor((t2-t1+tau)*fs)+len) = slope-1;
    pp(floor((t2-t1+tau)*fs+len)+1 : (floor(Tp*fs)+1)) = 0;

%         plot(pp(1:length(pp)));
    
    pp = circshift(pp, floor(t1*fs));
    pp = repmat(pp, 1, ceil(totalTime(length(totalTime))/Tp)+1);

    res = pp(1:length(totalTime));
end