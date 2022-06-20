    #!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Modificado por Victor de Mattos Arzolla - 2021
# Baseado no script "multiple_sensors.py" da PythonAPI 
# disponível no repositório oficial do CARLA Simulator 
# <https://github.com/carla-simulator/carla>

import glob
import os
import sys

from sources import * # funções necessárias para rodar a simulação

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse


from image_processing import computer_vision_rgb, control_monitor, SimulationData
import cv2
from controller import Controller



# Função para executar o controle
def control_main(vehicle, control, velocidade, psi, dx):

    #print(left_line, right_line)   


    steering = control.update(psi, dx, velocidade)

    # log_data(steering, 'steering')
    # log_data(psi, 'psi')
    # log_data(dx, 'dx')
    # log_data(dx_filt, 'dx_filt')
    # log_data(psi_filt, 'psi_filt')
    # log_data(time.time(),'time')

    steering_norm = steering*0.018
    vehicle.enable_constant_velocity(carla.Vector3D(velocidade, 0, 0)) # aplicando velocidade constante
    vehicle.apply_control(carla.VehicleControl(steer = round(float(steering_norm), 4))) # aplicando steering
    rodas = (vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)+vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))/2
    #print('psi:', psi,'  -   rodas:',rodas)
    #print('velocidade',vehicle.get_velocity())
    
    #print('steering', steering, theta, dx)
    #print('steering:', vehicle.get_control().steer)           # lendo steering
    #print('posicao', vehicle.get_transform())


def log_data(data, data_name):
    data_f = open(data_name+".txt", "a")
    data_f.write(str(data)+", ")




try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q, K_a, K_s, K_d, K_f, K_z, K_x, K_i, K_p
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()


    try:

        # Getting the world and
        world = client.get_world()
        print(world.get_map().name)
        if 'Town04' not in world.get_map().name :
            client.load_world('Town04')
            client.reload_world()
            print('Mudando para mapa Town04')
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)


        bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        #bp = veiculo_escolhido
        ponto_spawn = carla.Transform(carla.Location(x=388.0, y=-66.211449, z=0.1), carla.Rotation(pitch=-0.007903, yaw=90.600136, roll=0.053928)) # mais proximo ainda
        #ponto_spawn = carla.Transform(carla.Location(x=385.923126, y=-210.901535, z=0.090814), carla.Rotation(pitch=-0.531341, yaw=90.562447, roll=0.008176)) # proximo da curva acentuada (faixa 3)
        #ponto_spawn = carla.Transform(carla.Location(x=-510.374115, y=120.728378, z=0.2), carla.Rotation(pitch=0.713365, yaw=90.380745, roll=0.003147)) # reto (melhor trajeto completo) (faixa 3)
        #ponto_spawn = carla.Transform(carla.Location(x=-325.457489, y=12.516907, z=0.3), carla.Rotation(pitch=-0.763000, yaw=-179.927246, roll=0.002572)) # proximo da intersecção dificil (faixa 2)
        #ponto_spawn = carla.Transform(carla.Location(x=-397.941681, y=12.788073, z=0.1), carla.Rotation(pitch=-0.007445, yaw=179.632889, roll=0.005279)) # apos intersecção (faixa 2)
        #ponto_spawn = random.choice(world.get_map().get_spawn_points())
        print("Spawn do carro: ",ponto_spawn)
       

        
        vehicle = world.spawn_actor(bp, ponto_spawn)
        vehicle_list.append(vehicle)



        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 2], window_size=[args.width, args.height])
        #display_manager = DisplayManager(grid_size=[1, 1], window_size=[720, 720])


        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        RGBCamera = SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1.2, z=1.4), carla.Rotation(pitch=-15, yaw=0)), 
                      vehicle, {'fov' : '30'}, display_pos=[0, 0])

        RGBCamera2 = SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=-10, z=6), carla.Rotation(pitch=-20, yaw=0)), 
                      vehicle, {'fov' : '60'}, display_pos=[0, 1])

        # Segment = SensorManager(world, display_manager, 'Segmentation', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
        #               vehicle, {}, display_pos=[0, 1])
        # SensorManager(world, display_manager, 'Segmentation', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=00)), 
        #               vehicle, {}, display_pos=[0, 2])



        #Simulation loop

        #Configurando controlador
        # steer: 0.01   =>   rodas: 0.5687311887741089
        velocidade = 22
        #wn = 0.5625/velocidade
        control = Controller(K_psi=0.237, K_dx=2.85)
 
        #control.setFilter(n=1, wn=0.8)
        control.setSampleTime(0.033)
        #control.setOutputLimit(0.5, -0.5)


        # classe para gestão dos dados
        data = SimulationData()


        #vehicle.set_autopilot(True)

        call_exit = False
        time_init_sim = timer.time()

        frame = np.zeros((720,720,3))
        rgb_frame = np.zeros((720,720,3))

        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()



            ####################################################
            ####################################################
            ####################################################
            ####################################################
           
           
            # Envia frame para a função de visão computacional
            #seg_frame = Segment.rgb_frame
            rgb_frame = RGBCamera.rgb_frame
            
            #computer_vision(seg_frame, data)
            computer_vision_rgb(rgb_frame, data)
            control_main(vehicle, control, velocidade, data.psi, data.dx) #precisa retornar erro e steering

            data.steering = (vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)+vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))/2
            data.velocidade = velocidade
            data.control_output = control.last_output
            # data.Kp_dx = controlador.Kp_dx
            # data.Ki_dx = controlador.Ki_dx
            print(control.Term_psi, control.Term_arctan)
            #print('aqui',np.shape(frame))


            control_monitor(data)


            #print('left',vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
            #print('right',vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))



            ####################################################
            ####################################################
            ####################################################
            ####################################################

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:

                    '''if event.key == K_a:
                        new_kp =  controlador.Kp_theta+0.0001
                        if new_kp < 0: 
                            controlador.setKp(new_kp)
                            print('Aumentando Kp_theta para:',controlador.Kp)
                    if event.key == K_s: 
                        controlador.setKp(controlador.Kp_theta-0.0001)
                        print('Diminuindo Kp_theta para:',controlador.Kp)
                    if event.key == K_d:
                        new_kd =  controlador.Kp_dx+0.00001
                        if new_kd < 0:
                            controlador.setKd(new_kd)
                            print('Aumentando Kp_dx para:',controlador.Kd)
                    if event.key == K_f: 
                        controlador.setKd(controlador.Kp_dx-0.00001)
                        print('Diminuindo Kp_dx para:',controlador.Kd)'''
                    if event.key == K_z:
                        new_vel =  velocidade-0.5
                        if new_vel > 0:
                            velocidade = new_vel
                            print('Diminuindo velocidade para:',velocidade)
                    if event.key == K_x: 
                        velocidade = velocidade + 0.5
                        print('Aumentando velocidade para:',velocidade)  
                    if event.key == K_p: 
                        while(event.key != K_p): pass                                           

                    if event.key == K_i: 
                        
                        print('Info:')
                        print('Esquerda:', data.left_line)
                        print('Direita:', data.right_line)
                        print('Local do carro:', vehicle.get_transform())

                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                print('Saindo ....')
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)



def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1440x720',
        help='window resolution (default: 1440x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    video_frames  = None

    main()
