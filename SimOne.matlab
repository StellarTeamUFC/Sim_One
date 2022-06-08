clear all
close all

%Solar parameters
Msun=1.989E+30; % in kg
Rsun=6.957E+8; % in meters

np=inputdlg('Star radius (in solar radius):','Increment'); %do not click enter, press Ok
np=str2num(np{:});

frac=inputdlg('Fraction of solar radius (e.g., Earth (0.01) and Jupiter (0.1)):','Increment'); %do not click enter, press Ok
frac=str2num(frac{:});

% Input parameters
Rplanet=frac*Rsun;
Rstar=np.*Rsun; % Stellar radius in solar units
raizDepth=frac/np; % fraction of solar radius. Earth radius=0.00915 Rsun
depth=(raizDepth)^2; % transit depth
Porb=15; % Oribital period in days
ai=90; % inclination angle of star (between 0 and 90 degrees)
%cadence=0.020833; % cadence in days (Kepler mission)
cadence=0.0002895 % PLATO cadence


% Fundamental constants
G=6.6738E-11; % in m^3.kg^2.s^-2 (gravitational constant)
OneSec=1.15741E-5; % in days
Gday=G/OneSec.^2; % G constant in days

% Major semi-axis (aa);
aa=((Gday*Msun/4/pi^2).*Porb^2).^(1/3); % in m
aaAU=aa/(1.5E+11);

b=(aa/Rstar)*cos(pi/180*ai); % Impactor parameter, in a/Rstar

% Total time
part1=Rstar/aa;
part2=sqrt(((1+sqrt(depth)).^2)-(b^2));
part3=sqrt(((1-sqrt(depth)).^2)-(b^2));

tTotal=(Porb/pi)*asin((part1*part2)./sqrt(1-cos(pi/180*ai).^2)); % in days
tF=(Porb/pi)*asin(sin(tTotal*pi/Porb)*(part3./part2)); % in days
tTh=tTotal.*24; % in hours
tFh=tF.*24; % in hours

% Create a planetary transit
% Trapezoidal wave
%ntT=ceil(tT./cadence);
tramp=(tTotal-tF)/2; %ramp time
ntTF=tramp/cadence;
nt=ceil(ntTF); % number of points between 1st and 2nd contact 
ntF=tF/cadence;
ntF=ceil(ntF); % number of points between 2nd and 3rd contact

interTransit=Porb/cadence-tramp;
interTransit=ceil(interTransit); % number of point between two transits

rampdown=linspace(1,1-depth,nt);
rampup=linspace(1-depth,1,nt);
x1=(1-depth)*ones(1,ntF);
x2=1*ones(1,interTransit);

rep=10; % number of repetitions of transit
pulse=[rampdown x1 rampup x2];
pulse_train=repmat(pulse, rep,1).';

PT=pulse_train(:);
time=(0:(length(PT)-1))*cadence;
time=time';

plot(time,PT);

% noisy time series
stdnoise=0.0006; % noise SD
fs=length(time)-1;
noise=ffgn(stdnoise,0.500001,1,fs+1,0)'; %H=0.5 (Gaussian noise)

% noisy PT time series
nPT=PT+noise;
plot(time,nPT,'.');

%%%%% The model of the rotational modulation
%%%% Input parameters
gamma1=0.3985; % gamma 1 and 2 are parameters of Limb Darkening
gamma2=0.2586;

ap=1-gamma1-gamma2;
bp=gamma1+2*gamma2;
cp=-gamma2; % ap, bp and cp are the coefficients that give us the bolometric specific intesity
% of the photoshere as a function of the limb angle adopting a quadratic
% limb-darkening law

cs=0.850; % it id a coefficient specifying the spot bolometric contrast
cf0=1.115;
Q=0.5; % it is a function of the phase of the solar cycle
As=0.01; % it is the spot area of the ith active region in unit of the solar surface

ii=ai*pi/180; % inclination angle with respect to the line of sight
theta=0.17*pi; % it is the position of an active region on star (e.g., 30 deg)

Peq=15; % in days
AmpliP=0.2; % alpha=DeltaPer/Per
P2=Peq*(1-AmpliP*(sin(theta))^2)^(-1);
lambda=2*pi/5;

% Spot 1

omega=2*pi/Peq; % the angular velocity of the star
c=(ap+2*bp/3+cp/2)^(-1);
mii=cos(ii).*cos(theta)+sin(ii).*sin(theta).*cos(lambda+omega.*linspace(0,time(end),length(time))); % angle of limb

cf=cf0.*(1-mii'); % contrast intensity of faculae
s1=1+(As)*(ap+bp.*mii'+cp.*mii'.^2).*c.*((Q.*cf-cs).*mii');

% Spot 2
omega=2*pi/P2;
mii=cos(ii).*cos(theta)+sin(ii).*sin(theta).*cos(lambda+omega.*linspace(0,time(end),length(time))); % angle of limb

cf=cf0.*(1-mii'); % contrast intensity of faculae
s2=1+(As)*(ap+bp.*mii'+cp.*mii'.^2).*c.*((Q.*cf-cs).*mii');

spots=(s1+s2)-mean(s1+s2);

%%%%% whole time series
wts=spots+nPT;

figure;

plot(time,wts,'.');
xlabel('time (days)','FontSize',12);
ylabel('Normalized Flux','FontSize',12);







