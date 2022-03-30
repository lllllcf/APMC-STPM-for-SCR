close all;
clear all;
%% define constant
c = 3e8; % light speed
 
fLO = 77e9;
% fLO = 70e12;
fs = 1e13;
fp = 1e9;
fd = 2e8;
lambda = c/fLO;
z = lambda/2; % distance between two antennas
arrayNum = 5; % # of antennas
dataLen=4/fd;%data length
fIF=4e8;%IF Carrier Frequency
 
 
% pulse parameters
Tp = 1/fp;
tau = Tp/3;
beta = 2*pi/lambda;
t = 0:1/fs:dataLen;
rms = 0e-12; % rms for jitter
 
Ts=1/fs;%Sampling Period
Fn=fs/2;%Nyquist Frequency
BitNum=dataLen*fd;
A0=1;%IF Carrier Amplitude
 
%% -------------------------Data Generating in Square Wave-------------------
% data=sign(randn(fd*dataLen,1));%Randomly Generated Signed Singal
% dataTemp=ones((fs/fd),1)*data';%Dealing with Each Bit
% dataPlot=dataTemp(:);%Makes data vector fit the time vector
% t = t(1:length(dataPlot));
% 
% figure(1)
% subplot(3,1,1)
% plot(t/1e-6,dataPlot);%Plot the Generated Data
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Generated Input Data, P="+powerlvl(dataPlot));
% %% ------------------------Generating I & Q Signal---------------------------
% %Noticing that for Input data In and I-Q signal we have relation:
% %In:00 01 10 11; respectively; (IQ) (-1-1) (-11) (1-1) (11)
% %which means I for the even Bit and Q for the odd Bit
% 
% %------Generating I Data
% Idata=data(1:2:floor(length(data)/2)*2);%odd
% IdataTemp=ones((2*fs/fd),1)*Idata';
% IdataPlot=IdataTemp(:);
% subplot(3,1,2)
% plot(t/1e-6,IdataPlot);%Plot the Generated I Data
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("I Channel Input Data, P="+powerlvl(IdataPlot));
% %------Generating Q Data
% Qdata=data(2:2:floor(length(data)/2)*2);%even
% QdataTemp=ones((2*fs/fd),1)*Qdata';
% QdataPlot=QdataTemp(:);
% subplot(3,1,3)
% plot(t/1e-6,QdataPlot);%Plot the Generated Q Data
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Q Channel Input Data, P="+powerlvl(QdataPlot));
% %------IF Carrier
% Icarrier=A0.*cos((2*pi*fIF).*t)';
% Qcarrier=A0.*sin((2*pi*fIF).*t)';
% %------I-Q Modulated Signal with IF Carrier
% IIFModulated=Icarrier.*IdataPlot;
% QIFModulated=Qcarrier.*QdataPlot;
% %------Summed up QPSK Signal
% sumIQIF=IIFModulated+QIFModulated;
 
 
%% -------TMA && Frequency Up-Converting
sumIQIF=1;
 
sumIQRF_I=sumIQIF.*cos((2*pi*(fLO)).*t)';
sumIQRF_Q=sumIQIF.*cos((2*pi*(fLO)).*t+pi/2)';
%  
% delay = pi;
%  
% sumIQRF_I2=sumIQIF.*cos((2*pi*(fLO)).*t+delay)';
% sumIQRF_Q2=sumIQIF.*cos((2*pi*(fLO)).*t+delay+pi/2)';
%  
 
Eff = [];
Efftot = [];
PBO = 0;
% for PBO=-10:1:0
a = 10^(PBO/10);
tau1 = a*tau;
% para of modified signal
ptot =[];
p1 = [];
p_1 = [];
p_3 = [];
p5 = [];
p_7 = [];
p9 = [];
theta = 20; % target location
 
% modify tau of TMA
 
% delta = Tp/3*a;
delta = Tp/10000;
% tic
% alpha = 20;
% m = 3;
p_int=0.8;
% 
% figure
% p0 = pulse(0,tau1,0+Tp/2,Tp,t,fs,delta);
% p180 = pulse(-Tp/3,tau1,-Tp/3+Tp/2,Tp,t,fs,delta);
% subplot(2,1,1)
% hold on
% plot(t,p0,'b');
% plot(t,p180 ,'r');
% xlim([0 max(2*Tp)]);
% legend("0","180")
% xlabel('Time (second)');ylabel('Amp');
% 
% subplot(2,1,2)
% hold on
% plot(t,p0+p180,'b');
% xlim([0 max(2*Tp)]);
% xlabel('Time (second)');ylabel('Amp');
% title('p0+p180')



m = -1/3;
for alpha = -89:1:89
    delta_t = z*sind(alpha)/c; % time delay of two signal
    delta_tt = delta_t/(1/fs) % transfer the time delay to the array index
    
%     t11 = -Tp/4;
    t11 = 0;
    s11 = t11-m*Tp;
 
        resSig = sumIQRF_I'.*pulse(t11,tau1,t11+Tp/2,Tp,t,fs,delta)- ...
                sumIQRF_Q'.*pulse(t11-Tp/4,tau1,t11+Tp/4,Tp,t,fs,delta);
%                 sumIQRF_I2'.*pulse(s11,tau1,s11+Tp/2,Tp,t,fs,delta)- ...
%                 sumIQRF_Q2'.*pulse(s11-Tp/4,tau1,s11+Tp/4,Tp,t,fs,delta);
%  
 
    for i=2:arrayNum
        
        
        t1i = (((i-1)*beta*z*sind(theta))/pi)/2*Tp;  
%         t1i = (((i-1)*beta*z*sind(theta))/pi-1/2)/2*Tp;  
        s1i = t1i-m*Tp;
        
        Sigi = sumIQRF_I'.*pulse(t1i,tau1,t1i+Tp/2,Tp,t,fs,delta)- ...
            sumIQRF_Q'.*pulse(t1i-Tp/4,tau1,t1i+Tp/4,Tp,t,fs,delta);
%             sumIQRF_I2'.*pulse(s1i,tau1,s1i+Tp/2,Tp,t,fs,delta)- ...
%             sumIQRF_Q2'.*pulse(s1i-Tp/4,tau1,s1i+Tp/4,Tp,t,fs,delta);
                            
        resSig = resSig + circshift(Sigi, -floor(delta_tt*(i-1)));
 
    end
    if mod(alpha,10) == 0
        figure(3)
        hold on
        [ftarget,FFTtarget]=SSBFFT(resSig,fs);
        plot((ftarget-fLO)./fp,FFTtarget);
        grid on
        xlabel("(f-fLO)/fp");ylabel('AMPLITUDE');
        xlim([-15 15])
        title("modified TMA with tau1 = "+a+"*tau, at different direction");
    end
 
% the following code are used to plot Power vs. alpha
[f2,FFTsig]=SSBFFT(resSig,fs);
 
        p11 = find(f2>(fLO+0.7*fp),1);
        p12 = find(f2>(fLO+1.3*fp),1);
        
        p_11 = find(f2>(fLO-1.3*fp),1);
        p_12 = find(f2>(fLO-0.7*fp),1);
        
        p_31 = find(f2>(fLO-3.3*fp),1);
        p_32 = find(f2>(fLO-2.7*fp),1);
        
        p51 = find(f2>(fLO+4.7*fp),1);
        p52 = find(f2>(fLO+5.3*fp),1);
        
        p_71 = find(f2>(fLO-7.3*fp),1);
        p_72 = find(f2>(fLO-6.7*fp),1);
        
        p91 = find(f2>(fLO+8.7*fp),1);
        p92 = find(f2>(fLO+9.3*fp),1);
        
ptot = [ptot, sum(FFTsig.^2)]; 
p1 = [p1,sum(FFTsig(p11:p12).^2)]; 
p_1 = [p_1,sum(FFTsig(p_11:p_12).^2)];
p_3 = [p_3,sum(FFTsig(p_31:p_32).^2)];
p5 = [p5,sum(FFTsig(p51:p52).^2)];
p_7 = [p_7,sum(FFTsig(p_71:p_72).^2)];
p9 = [p9,sum(FFTsig(p91:p92).^2)];
 
end
% toc
 
% calculate the efficiency
px = p1+p_1+p_3+p5+p_7+p9;
temp = find(p1==max(p1));
eff = sum(p1)/sum(px);
efftot = sum(p1)/sum(ptot);
Eff = [Eff eff];
Efftot = [Efftot efftot];
 
% end
 
 
 
%% plot 
alpha = -89:1:89;
 
m = max(p1);
 
p1db = 10.*log10(p1./m);
p_1db = 10.*log10(p_1./m);
p_3db = 10.*log10(p_3./m);
p5db = 10.*log10(p5./m);
p_7db = 10.*log10(p_7./m);
p9db = 10.*log10(p9./m);
 
p1db(p1db<-35) = -35;
p_1db(p_1db<-35) = -35;
p_3db(p_3db<-35) = -35;
p5db(p5db<-35) = -35;
p_7db(p_7db<-35) = -35;
p9db(p9db<-35) = -35;
 
figure(11)
% subplot(2,1,1)
hold on
box on
set(gca,'linewidth',1.5,'FontSize',13,'FontWeight','bold')
% set(gca, 'XTick', [0 1 2])
grid on
plot(alpha, p1db,'-','LineWidth',1.5);
plot(alpha, p_1db,'-','LineWidth',1.5);
plot(alpha, p_3db,'--','LineWidth',1.5);
plot(alpha, p5db,'--','LineWidth',1.5);
plot(alpha, p_7db,'-','LineWidth',1.5);
plot(alpha, p9db,'--','LineWidth',1.5);
xlim([-90,90])
xlabel('\textbf{Angle (}$^\circ$\textbf{)}','Interpreter','Latex','FontSize',15);
ylabel('\textbf{Normalized Gain (dB)}','Interpreter','Latex','FontSize',15);
% title("4path tau1 = "+a+"*tau delta = Tp/"+Tp/delta+" $f_{LO}$ = "+...
%     fLO/1e9+ "GHz, $f_{p}$ = "+fp/1e9+ "GHz, $f_{IF}$ = "+fIF/1e9+ ...
%     "GHz,  $\Theta^{scann}$ = "+theta+",1st efficiency="+eff,'Interpreter','Latex');
% title("77GHz theta= "+theta+" "+floor(10*log10(a)*10)/10+"dbPBO"+...
%                 ", efficiency="+floor(eff*1000)/10+"\%",'Interpreter','Latex');
 
legend({'\textbf{m=1}','\textbf{m=-1}','\textbf{m=-3}','\textbf{m=5}','\textbf{m=-7}','\textbf{m=9}'},...
    'FontSize',12,'Interpreter','Latex','Location','northwest','NumColumns',2)
 
 
 
% RFReceived=resSig'; % received signal
% 
% % FFT for signal received
% figure(5)
% [fR,FFTRFReceived]=SSBFFT(RFReceived,fs);
% plot((fR-fLO)./fp,FFTRFReceived);
% grid on
% xlabel("(f-fLO)/fp");ylabel('AMPLITUDE');
% xlim([-15 15])
% title("modified TMA with tau1 = "+a+"*tau at direction 89");
%% --------Down-Converting
%Set its Central Frequency as fIF
% DownReceived=RFReceived.*cos((2*pi*(fLO+fp)).*t)';
% 
% %% --------Filtering-----Lowpass
% RFFiltered=DownReceived;
% 
% %% --------------IQ demodulation
% % RFFiltered = sumIQIF;
% symNum = fs/(fd);
% % phi0 = IQEVM(RFFiltered,A0,fIF,fd,fs,symNum, t);
% phi0 = 0;
% Icarrier=A0.*cos((2*pi*(fIF)).*t+phi0)';
% Qcarrier=A0.*sin((2*pi*(fIF)).*t+phi0)';
% IReceived=Icarrier.*RFFiltered;
% QReceived=Qcarrier.*RFFiltered;
% %--------Integral(Summing) to get I & Q data
% Isampled=reshape(IReceived,symNum,floor(length(IReceived)/symNum));
% Qsampled=reshape(QReceived,symNum,floor(length(QReceived)/symNum));
% Imean=mean(Isampled,1)';
% Qmean=mean(Qsampled,1)';
% ITemp=ones(symNum,1)*Imean';
% II=ITemp(:);
% QTemp=ones(symNum,1)*Qmean';
% QQ=QTemp(:);
% 
% %% Plot I and Q data after Sampling
% figure(8)
% subplot(2,2,1)
% plot(t/1e-6,II);
% axis([0 dataLen/1e-6 min(II) max(II)])
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("I Channel Data After Sampling");
% subplot(2,2,3)
% plot(t/1e-6,QQ);
% axis([0 dataLen/1e-6 min(QQ) max(QQ)])
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Q Channel Data After Sampling");
% %--------Use a comparator to recover the I & Q
% ICompared=sign(II);
% QCompared=sign(QQ);
% subplot(2,2,2)
% plot(t/1e-6,ICompared);
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("I Channel Data");
% subplot(2,2,4)
% plot(t/1e-6,QCompared);
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Q Channel Data");
% %------------------Final Received Data
% IComparedTemp=ICompared(1:2*fs/fd:floor(length(ICompared)/(2*fs/fd))*(2*fs/fd));
% QComparedTemp=QCompared(1:2*fs/fd:floor(length(ICompared)/(2*fs/fd))*(2*fs/fd));
% sumCompared=[IComparedTemp QComparedTemp]';
% sumCompared=sumCompared(:);
% sumCompared=ones(fs/fd,1)*sumCompared';
% sumCompared=sumCompared(:);
% figure(9)
% subplot(2,1,1)
% plot(t/1e-6,sumCompared);
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Final Received Data, P="+powerlvl(sumCompared));
% subplot(2,1,2)
% plot(t/1e-6,sumCompared-dataPlot);
% BitErrorRate=sum(abs(sumCompared-dataPlot))/(fs*dataLen)/2;
% ylim([-1.5*A0 1.5*A0]);
% xlabel('TIME($\mu$s)','Interpreter','Latex');ylabel('AMPLITUDE');
% title("Difference between Data Received and Input, P="+powerlvl(sumCompared-dataPlot));
 
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
%% pulse function with jitter
function res=jitPulse(t1,tau,t2,Tp,totalTime,fs,rms)
    % repmat
%     tic
    pp=zeros(1,length(totalTime));
    loopNum = ceil(totalTime(length(totalTime))/Tp);
    j=normrnd(0,rms,[loopNum,4]); % time jitter
    j=floor(j*fs);
    len = floor(Tp*fs);
    for ii = 1:loopNum
        start = (ii-1)*len+1;
        pp(start:ii*len)=0;
        pp(start : start-1+floor((tau)*fs)+j(ii,2)) = 1;
%         start+floor((tau)*fs)+j(ii,2)
%         start-1+floor((t2-t1)*fs)+j(ii,3)
 
%         pp(start+floor((tau)*fs)+j(ii,2) : start-1+floor((t2-t1)*fs)+j(ii,3)) = 0;
        pp(start+floor((t2-t1)*fs)+j(ii,3) : start-1+floor((t2-t1+tau)*fs)+j(ii,4)) = -1;
%         pp(start+floor((t2-t1+tau)*fs)+j(ii,4) : ii*len) = 0;
 
        pp(start:ii*len) = circshift(pp(start:ii*len), floor(t1*fs)+j(ii,1));
    end
%     toc
%     pp = repmat(pp, 1, ceil(totalTime(length(totalTime))/Tp)+1);
 
    res = pp(1:length(totalTime));
end
%% phase noise
function Sout = add_phase_noise( Sin, Fs, phase_noise_freq, phase_noise_power, VALIDATION_ON )
 
if nargin < 5
     VALIDATION_ON = 0;
end
 
% Check Input
error( nargchk(4,5,nargin) );
 
if ~any( imag(Sin(:)) )
     error( 'Input signal should be complex signal' );
end
if max(phase_noise_freq) >= Fs/2
     error( 'Maximal frequency offset should be less than Fs/2');
end
     
% Make sure phase_noise_freq and  phase_noise_power are the row vectors
phase_noise_freq = phase_noise_freq(:).';
phase_noise_power = phase_noise_power(:).';
if length( phase_noise_freq ) ~= length( phase_noise_power )
     error('phase_noise_freq and phase_noise_power should be of the same length');
end
 
% Sort phase_noise_freq and phase_noise_power
[phase_noise_freq, indx] = sort( phase_noise_freq );
phase_noise_power = phase_noise_power( indx );
 
% Add 0 dBc/Hz @ DC
if ~any(phase_noise_freq == 0)
     phase_noise_power = [ 0, phase_noise_power ];
     phase_noise_freq = [0, phase_noise_freq];
end
 
% Calculate input length
N = prod( size( Sin ) );
 
% Define M number of points (frequency resolution) in the positive spectrum 
%  (M equally spaced points on the interval [0 Fs/2] including bounds), 
% then the number of points in the negative spectrum will be M-2 
%  ( interval (Fs/2, Fs) not including bounds )
%
% The total number of points in the frequency domain will be 2*M-2, and if we want 
%  to get the same length as the input signal, then
%   2*M-2 = N
%   M-1 = N/2
%   M = N/2 + 1
%
%  So, if N is even then M = N/2 + 1, and if N is odd we will take  M = (N+1)/2 + 1
%
if rem(N,2),    % N odd
     M = (N+1)/2 + 1;
else
     M = N/2 + 1;
end
 
 
% Equally spaced partitioning of the half spectrum
F  = linspace( 0, Fs/2, M );    % Freq. Grid 
dF = [diff(F) F(end)-F(end-1)]; % Delta F
 
 
% Perform interpolation of phase_noise_power in log-scale
intrvlNum = length( phase_noise_freq );
logP = zeros( 1, M );
for intrvlIndex = 1 : intrvlNum,
     leftBound = phase_noise_freq(intrvlIndex);
     t1 = phase_noise_power(intrvlIndex);
     if intrvlIndex == intrvlNum
          rightBound = Fs/2; 
          t2 = phase_noise_power(end);
          inside = find( F>=leftBound & F<=rightBound );  
     else
          rightBound = phase_noise_freq(intrvlIndex+1); 
          t2 = phase_noise_power(intrvlIndex+1);
          inside = find( F>=leftBound & F<rightBound );
     end
     logP( inside ) = ...
          t1 + ( log10( F(inside) + realmin) - log10(leftBound+ realmin) ) / ( log10( rightBound + realmin) - log10( leftBound + realmin) ) * (t2-t1);     
end
P = 10.^(real(logP)/10); % Interpolated P ( half spectrum [0 Fs/2] ) [ dBc/Hz ]
 
% Now we will generate AWGN of power 1 in frequency domain and shape it by the desired shape
% as follows:
%
%    At the frequency offset F(m) from DC we want to get power Ptag(m) such that P(m) = Ptag/dF(m),
%     that is we have to choose X(m) =  sqrt( P(m)*dF(m) );
%  
% Due to the normalization factors of FFT and IFFT defined as follows:
%     For length K input vector x, the DFT is a length K vector X,
%     with elements
%                      K
%        X(k) =       sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
%                     n=1
%     The inverse DFT (computed by IFFT) is given by
%                      K
%        x(n) = (1/K) sum  X(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
%                     k=1
%
% we have to compensate normalization factor (1/K) multiplying X(k) by K.
% In our case K = 2*M-2.
 
% Generate AWGN of power 1
 
if ~VALIDATION_ON
     awgn_P1 = ( sqrt(0.5)*(randn(1, M) +1j*randn(1, M)) );
else
     awgn_P1 = ( sqrt(0.5)*(ones(1, M) +1j*ones(1, M)) );
end
 
% Shape the noise on the positive spectrum [0, Fs/2] including bounds ( M points )
X = (2*M-2) * sqrt( dF .* P ) .* awgn_P1; 
 
% Complete symmetrical negative spectrum  (Fs/2, Fs) not including bounds (M-2 points)
X( M + (1:M-2) ) = fliplr( conj(X(2:end-1)) ); 
 
% Remove DC
X(1) = 0; 
 
% Perform IFFT 
x = ifft( X ); 
 
% Calculate phase noise 
phase_noise = exp( j * real(x(1:N)) );
 
% Add phase noise
if ~VALIDATION_ON
     Sout = Sin .* reshape( phase_noise, size(Sin) );
else
     Sout = 'VALIDATION IS ON';
end
 
if VALIDATION_ON
     figure; 
     plot( phase_noise_freq, phase_noise_power, 'o-' ); % Input SSB phase noise power
     hold on;    
     grid on;     
     plot( F, 10*log10(P),'r*-'); % Input SSB phase noise power
     X1 = fft( phase_noise );
     plot( F, 10*log10( ( (abs(X1(1:M))/max(abs(X1(1:M)))).^2 ) ./ dF(1) ), 'ks-' );% generated phase noise exp(j*x)     
     X2 = fft( 1 + j*real(x(1:N)) ); 
     plot( F, 10*log10( ( (abs(X2(1:M))/max(abs(X2(1:M)))).^2 ) ./ dF(1) ), 'm>-' ); % approximation ( 1+j*x )   
     xlabel('Frequency [Hz]');
     ylabel('dBc/Hz');
     legend( ...
          'Input SSB phase noise power', ...
          'Interpolated SSB phase noise power', ...
          'Positive spectrum of the generated phase noise exp(j*x)', ...
          'Positive spectrum of the approximation ( 1+j*x )' ...
     );     
end
end
