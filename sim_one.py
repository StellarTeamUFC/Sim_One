# Primeiro simulador de trânsito planetário da disciplina de Astrofísica Observacional

# Carrega Bibliotecas

import numpy as np
from numpy import sqrt, arcsin, sin, cos, pi

from math import ceil
from astropy.constants import M_sun, R_sun, G
import astropy.units as u

import matplotlib.pyplot as plt

from stochastic.processes.noise import FractionalGaussianNoise

# Função linspace que permite float

def linspace_float(low, up, leng):
    step = ((up-low) * 1.0 / leng)
    return [low+i*step for i in range(leng)]

""" # Execução """

''' Declaração de variáveis '''

# Constantes fundamentais

M_sun = M_sun.value                                           # Massa do sol em kg
R_sun = R_sun.value                                           # Raio do sol em m

# Conversão de unidades

Gday = G.to(u.m**3 / (u.kg * u.d**2))
Gday = Gday.value                                             # Constante gravitacional (m^3 * kg^2 * dia^(-2))

# Parâmetros de entrada

raio = float(input(' Raio da estrela (em raio solar): '))
frac = float(input(' Fração do raio solar (ex.: Terra (0.01), Jupiter (0.1): '))

Rstar = raio * R_sun                                          # Raio da estrela em unidades solares
Rplanet = frac * R_sun                                        # Fração de raios solares. Raio da Terra = 0.00915 Rsun
raizDepth = frac/raio                                         # Raiz da profundidade
depth = (raizDepth)**2                                        # Profundidade de trânsito
Porb = 15                                                     # Período orbital em dias
ai = 90                                                       # Ângulo de inclinação da estrela (entre 0 e 90 graus)
cadence_kepler = 0.020833                                     # Cadência em dias (Missão Kepler)
cadence_plato = 0.0002895                                     # Cadência em dias (Missão Plato)

# Escolha da cadência

cadence_question = str(input(' Cadência utilizada, PLATO ou Kepler: '))

if cadence_question.lower() in ['plato']:
    cadence = cadence_plato
else:
    cadence = cadence_kepler

# Semi-eixo maior (aa)

aa = (Gday * M_sun * Porb ** 2/ (4 * pi **2))**(1/3)

# Parâmetro de impacto, em a/Rstar

b = (aa / Rstar) * cos(ai * pi / 180)

# Tempo total

part1 = Rstar / aa
part2 = (1 + Rplanet/Rstar)**2 - b**2
part3 = (1 - Rplanet/Rstar)**2 - b**2
part4 = 1 - (cos(ai * pi / 180))**2

tTotal = (Porb / pi) * arcsin(part1 * sqrt(part2/part4)) # em dias

tT = tTotal * 24 # em horas

tF = (Porb / pi) * arcsin(sin(tTotal * (pi / Porb))) * sqrt(part3)/sqrt(part2) # em dias

tFh = tF * 24 # em horas

print(f'\n duração do trânsito total: {tT} hr')
print(f'\n duração entre o segundo e terceiro contato: {tFh} hr')
print(f'\n diferença entre as durações: {tT - tFh} hr')

# Criando um trânsito planetário (onda trapezoidal)

tramp = (tTotal - tF)/2                    # Tempo de rampa
ntTF = tramp/cadence                
nt = ceil(ntTF)                            # Número de pontos entre o 1º e o 2º contato 
ntF = ceil(tF/cadence)                     # Número de pontos entre o 2º e o 3º contato

interTransit = ceil(Porb/cadence - tramp)  # Número de pontos entre os dois trânsitos

rampdow = linspace_float(1, 1-depth, nt)
rampup = linspace_float(1-depth,1,nt + 1)

x1 = (1 - depth) * np.ones(ntF)
x2 = np.ones(interTransit)

rep = 10
pulse = np.concatenate([rampdow,x1,rampup,x2])
pulse_train = np.repeat(pulse, 10).reshape(len(pulse),rep)  # era necessário?

PT = np.reshape(pulse_train, (len(pulse_train)*rep, 1))     # tentativa
time = [float(x)*cadence for x in range(0,len(PT))]
time = np.array(time).reshape((-1, 1))

#plt.plot(time, PT)

# Criação do ruído

fs = len(time) - 1
gn = FractionalGaussianNoise(hurst=0.5)
noise = gn.sample(fs + 1)
noise = np.array(noise).reshape((-1, 1))

nPT = PT + noise

fig, ax = plt.subplots(figsize = (20,10))
plt.scatter(time,nPT, s = 1)

plt.show()
