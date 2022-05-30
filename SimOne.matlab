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
Rstar=np.*Rsun; 									% Stellar radius in solar units
Rplanet=frac*(Rstar/Rsun); 							% fraction of solar radius. Earth radius=0.00915 Rsun
depth=(Rplanet.*Rsun/Rstar)^2; 						% transit depth
Porb=15; 											% Oribital period in days
ai=90; 												% inclination angle of star (between 0 and 90 degrees)
cadence=0.020833; 									% cadence in days (Kepler mission)

% Fundamental constants
G=6.6738E-11; 										% in m^3.kg^2.s^-2 (gravitational constant)
OneSec=1.15741E-5; 									% one second in days
Gday=G/OneSec.^2; 									% G constant in days

% Major semi-axis (aa);
aa=((Gday*Msun/4/pi^2).*Porb^2).^(1/3); 			% in m
aaAU=aa/(1.5E+11);

b=(aa/Rstar)*cos(pi/180*ai); 						% Impacto parameter, in a/Rstar

% Total time
part1=Rstar/aa;
part2=sqrt(((1+sqrt(depth)).^2)-(b^2));
part3=sqrt(((1-sqrt(depth)).^2)-(b^2));

tTotal=(Porb/pi)*asin((part1*part2)./sqrt(1-cos(pi/180*ai).^2)); 	% in days
tF=(Porb/pi)*asin(sin(tTotal*pi/Porb)*(part3./part2)); 				% in days
tTh=tTotal.*24; 													% in hours
tFh=tF.*24; 														% in hours