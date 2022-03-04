#!/usr/bin/python
#
# Controle baseado no ivPID, com algumas modificações e mais recursos para se utilizar na FIRA.


import time

class PID:
    """PID Controller
    """

    def __init__(self, Kp=0.2, Ki=0.0, Kd=0.0):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.sample_time = 0.01
        
        #self.current_time = time.time()
        self.last_time = time.time()
        self.last_error = None
        self.last_output = None

        # Limpando variaveis
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.windup_max = None
        self.windup_min = None
        self.windup_reset = None
        self.windup_reset_tolerance = 0.001

        # Limites da saída
        self.cmax = None
        self.cmin = None



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
            

    def update(self, state, current_time=None):

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
        error = self.SetPoint - state

        # Computa variação do erro
        delta_error = error - self.last_error

        # Cálculo do termo Proporcional
        self.PTerm = self.Kp * error

        # Caso metodo de windup reset esteja habilitado e erro menor q tolerância ou erro atravessou o zero
        if (self.windup_reset is True) and ((abs(error) < self.windup_reset_tolerance) or ((error * self.last_error) < 0)):
            self.ITerm = 0
        else:
            # Cálculo do termo Integral
            self.ITerm += self.Ki * error * delta_time

        
        # Calcula termo Diferencial
        self.DTerm = self.Kd * delta_error / delta_time

        # Aplica limite de windup
        self.ITerm = self.__saturation(self.ITerm, self.windup_max, self.windup_min)

        # Calcula resposta do controle
        output = self.PTerm + self.ITerm + self.DTerm

        # Limita o output do sistema
        output = self.__saturation(output, self.cmax, self.cmin)

        # Salva os valores atuais para próxima chamada de atualização
        self.last_time = current_time
        self.last_error = error
        self.last_output = output

        return output

    def setSetPoint(self, setpoint):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.SetPoint = setpoint

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, method='None', windup_max=None, windup_min=None):
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
        if method == 'None':
            self.windup_max = None
            self.windup_min = None
            self.windup_reset = None
        # Este método zera o termo integral quando o erro está abaixo de determinada tolerância
        elif method == 'Reset':
            self.windup_reset = True
            # Input do windup max é tolerância do erro
            if windup_max is not None:
                self.windup_reset_tolerance = windup_max
        # Este método limita o valor máximo e mínimo que o termo integral pode ter
        elif method == 'Clamp':
            self.windup_max = windup_max
            # Caso não seja inserido windup_min
            if windup_min is None:
                # Windup minimo é menos windup máximo
                self.windup_min = -windup_max
            # Caso contrário 
            else:
                # Seta windup mínimo
                self.windup_min = windup_min


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
