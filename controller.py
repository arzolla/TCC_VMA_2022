 #!/usr/bin/python
#
# Controle baseado no ivPID, com algumas modificações e mais recursos para se utilizar na FIRA.


import time
import numpy as np
from scipy import signal

class Controller:
    """Controlador não-linear
    """

    def __init__(self, K_psi=0.025, K_dx=0.2):

        # Parâmetros de ganho
        self.K_psi = K_psi
        self.K_dx = K_dx

        # Variaveis
        self.Term_psi = 0.0
        self.Term_arctan = 0.0
        self.sample_time = 0.01
        self.last_time = time.time()
        self.last_output = None

        # Limites da saída
        self.cmax = None
        self.cmin = None
        

        # Configuração do filtro
        self.num = None
        self.den = None
        self.zi_psi = None
        self.zi_dx = None


    # Função estática para limitar determinado valor
    @staticmethod
    def __saturation(value, max, min):
        if value is None:
            return None
        if (max is not None) and (value > max):
            return max
        elif (min is not None) and (value < min):
            return min
        return value

    def __filter(self, input, zi):
        if zi is None or input is None:
            return input
        output, zi = signal.lfilter(self.num, self.den, [input], zi=zi)  
        return output, zi

    def update(self, psi, dx, velocidade, current_time=None):

        # Se não for inserido tempo atual no update()
        if current_time is None:
            # Obtém tempo atual
            current_time = time.time()

        # se delta de tempo for positivo
        if current_time - self.last_time:
            # Computa a variação do tempo
            delta_time = current_time - self.last_time
        # Se delta_time for zero
        else: 
            delta_time = 1e-16

        # Só entra na rotina de atualização a cada 'sample_time' segundos
        if (self.sample_time is not None) and (delta_time < self.sample_time) and (self.last_output is not None):
            # Se não passou tempo suficiente só retorna ultimo valor da saída
            return self.last_output

        # Filtra as entradas
        psi, self.zi_psi = self.__filter(psi, self.zi_psi)
        dx, self.zi_dx = self.__filter(dx, self.zi_dx)
        
        # Cálculo dos termos
        self.Term_arctan = np.arctan(self.K_dx * (dx/velocidade))/1.57 # arctan é normalizada para intervalo entre -1 e 1
        self.Term_psi = self.K_psi * psi

        # Calcula resposta do controle
        output = self.Term_psi + self.Term_arctan

        # Limita o output do sistema
        output = self.__saturation(output, self.cmax, self.cmin)

        # Salva os valores atuais para próxima chamada de atualização
        self.last_time = current_time
        self.last_output = output

        return output, psi, dx

    def setFilter(self, n=1, wn=0.04):
        """Define os parâmetros do filtro Butterworth a ser aplicado nas entradas"""
        self.num, self.den = signal.butter(n, wn)

        self.zi_psi = np.zeros(self.num.size-1)
        self.zi_dx = np.zeros(self.num.size-1)
        self.zi_output = np.zeros(self.num.size-1)


    def setKp(self, K_psi, K_dx):
        """Determina o ganho proporcional de psi e velocidade da arctan"""
        self.K_psi = K_psi
        self.K_dx = K_dx
       

    def setOutputLimit(self, cmax, cmin=None):
        """Define os limites máximo e mínimo da resposta do controle
        """
        self.cmax = cmax
        if cmin is None:
            self.cmin = -cmax
        else:
            self.cmin = cmin

    def setSampleTime(self, sample_time):
        """Define a taxa de atualização para o controlador.
        """
        self.sample_time = sample_time
