 #!/usr/bin/python
#
# Controle baseado no ivPID, com algumas modificações e mais recursos para se utilizar na FIRA.


import time
import numpy as np

class Control:
    """PI Controller
    """

    def __init__(self, Kp_theta=0.2, Kp_dx=0.0, Ki_dx=0.0):

        self.Kp_theta = Kp_theta
        self.Kp_dx = Kp_dx
        self.Ki_dx = Ki_dx

        self.sample_time = 0.01
        
        #self.current_time = time.time()
        self.last_time = time.time()
        self.last_error_theta = None
        self.last_error_dx = None
        self.last_output = None


        # Limpando variaveis
        self.SetPoint_theta = 0.0
        self.SetPoint_dx = 0.0

        self.PTerm_theta = 0.0
        self.PTerm_dx = 0.0
        self.ITerm_dx = 0.0


        # Limites da saída
        self.cmax = None
        self.cmin = None

        # Windup Guard
        self.windup_max = None
        self.windup_min = None
        self.windup_reset = None
        self.windup_reset_tolerance = 0.001


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

        # Computa o erro
        error_dx = self.SetPoint_dx - dx
        error_theta = self.SetPoint_theta - theta


        # Cálculo do termo Proporcional
        self.PTerm_dx = np.arctan(0.001*(self.Kp_dx * error_dx/velocidade))/1.5
        self.PTerm_theta = self.Kp_theta * error_theta * 0.025

        # Caso metodo de windup reset esteja habilitado e erro menor q tolerância ou erro atravessou o zero
        if (self.windup_reset is True) and ((abs(error_dx) < self.windup_reset_tolerance) or ((error_dx * self.last_error_dx) < 0)):
            self.ITerm_dx = 0
        else:
            # Cálculo do termo Integral
            self.ITerm_dx += self.Ki_dx * error_dx * delta_time


        # Aplica limite de windup
        self.ITerm_dx = self.__saturation(self.ITerm_dx, self.windup_max, self.windup_min)

        # Calcula resposta do controle
        output = self.PTerm_theta + self.PTerm_dx + self.ITerm_dx

        # Limita o output do sistema
        output = self.__saturation(output, self.cmax, self.cmin)

        # Salva os valores atuais para próxima chamada de atualização
        self.last_time = current_time
        self.last_error_dx = error_dx
        self.last_error_theta = error_theta
        self.last_output = output

        return output

    def setSetPoint(self, setpoint_theta, setpoint_dx):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.SetPoint_theta = setpoint_theta
        self.SetPoint_dx = setpoint_dx

    def setKp(self, Kp_theta, Kp_dx):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp_theta = Kp_theta
        self.Kp_dx = Kp_dx

    def setKi(self, Ki_dx):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki_dx = Ki_dx


    def setWindup(self, windup_max=None, windup_min=None, method = None):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        # Sem windup guard
        if method == None and windup_max == None:
            self.windup_max = None
            self.windup_min = None
            self.windup_reset = None
        if method == 'Clamp' or method == None:
            self.windup_max = windup_max
            # Caso não seja inserido windup_min
            if windup_min is None:
                # Windup minimo é menos windup máximo
                self.windup_min = -windup_max
            # Caso contrário 
            else:
                # Seta windup mínimo
                self.windup_min = windup_min
        # Este método zera o termo integral quando o erro está abaixo de determinada tolerância
        if method == 'Reset':
            self.windup_reset = True
            # Input do windup max é tolerância do erro
            if windup_max is not None:
                self.windup_reset_tolerance = windup_max
        # Este método limita o valor máximo e mínimo que o termo integral pode ter
        

    def setOutputLimit(self, cmax, cmin):
        """Define os limites máximo e mínimo da resposta do controle
        """
        self.cmax = cmax
        self.cmin = cmin

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time
