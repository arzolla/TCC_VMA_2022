 #!/usr/bin/python
#
# Controle baseado no ivPID, com algumas modificações e mais recursos para se utilizar na FIRA.


import time
import numpy as np
from scipy import signal

class Controller:
    """Controlador não-linear
    """

    def __init__(self, K_theta=0.025, K_dx=0.65, K_arctan=0.004):

        # Parâmetros de ganho
        self.K_theta = K_theta
        self.K_dx = K_dx
        self.K_arctan = K_arctan

        # Variaveis
        self.Term_theta = 0.0
        self.Term_dx = 0.0
        self.sample_time = 0.01
        self.last_time = time.time()
        self.last_output = None

        # Limites da saída
        self.cmax = None
        self.cmin = None
        

        # Configuração do filtro
        self.num = None
        self.den = None
        self.zi = None


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

    def __filter(self, input):
        if self.zi is None:
            return input
        output, _ = signal.lfilter(self.num, self.den, input, zi=self.zi)  
        return output

    def update(self, theta, dx, velocidade, current_time=None):

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
        theta = self.__filter(theta)
        dx = self.__filter(dx)

        # Cálculo dos termos
        self.Term_dx = np.arctan(0.001*(self.K_arctan * dx/velocidade)) * self.K_dx
        self.Term_theta = self.Kp_theta * theta


        # Calcula resposta do controle
        output = self.Term_theta + self.Term_dx

        # Limita o output do sistema
        output = self.__saturation(output, self.cmax, self.cmin)

        # Salva os valores atuais para próxima chamada de atualização
        self.last_time = current_time
        self.last_output = output

        return output

    def setFilter(self, n=1, wn=0.02):
        """Define os parâmetros do filtro Butterworth a ser aplicado nas entradas"""
        self.num, self.den = signal.butter(n, wn)
        self.zi = np.zeros(self.num.size-1)

    def setKp(self, Kp_theta, Kp_dx):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp_theta = Kp_theta
        self.Kp_dx = Kp_dx
       

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
