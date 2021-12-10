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


from image_processing import image_processing2, image_processing_kmeans, get_mask, show_image_rgb
import cv2

# Função para receber e processar a imagem recebida do simulador
def computer_vision(frame):

    if frame is not None:
        
        #show_image_rgb(frame) # Mostra imagem RGB

        mask = get_mask(frame) # Obtem apenas faixa da imagem segmentada
        
        Erro = image_processing2(mask)
        #image_processing_kmeans(mask)

        cv2.waitKey(1)
        return Erro


# Função para executar o controle
def control_main(vehicle, Erro):
    #print(frame)

    Estado = vehicle.get_control().steer
    if Estado is None:
        Estado = 0
    if Erro is None:
        Erro = 0

    steering = (0.006*Erro - Estado)

    print('Erro = ', Erro)
    print('Steering aplicado = ', steering)
    
    vehicle.enable_constant_velocity(carla.Vector3D(3, 0, 0)) # aplicando velocidade constante
    vehicle.apply_control(carla.VehicleControl(steer = float(steering))) # aplicando steering 
    
    #print('steering:', vehicle.get_control().steer)           # lendo steering
    #print('posicao', vehicle.get_transform())



try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
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
        #if 'Town04' not in world.get_map().name :
        #    client.load_world('Town04')
        #    client.reload_world()
         #   print('Mudando para mapa Town11')
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
        
        ponto_spawn = carla.Transform(carla.Location(x=402.525452, y=-124.737938, z=0.281942), carla.Rotation(pitch=0.000000, yaw=-89.401421, roll=0.000000)) # melhor
        ponto_spawn = carla.Transform(carla.Location(x=-400.416626, y=9.283669, z=0.281942), carla.Rotation(pitch=-2.857300, yaw=179.601227, roll=0.000000)) # faixas tracejadas
        #ponto_spawn = random.choice(world.get_map().get_spawn_points())
        
        print("Spawn do carro: ",ponto_spawn)
       

        
        vehicle = world.spawn_actor(bp, ponto_spawn)
        vehicle_list.append(vehicle)


        vehicle.set_autopilot(True)


        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 2], window_size=[args.width, args.height])


        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        RGBCamera = SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=00)), 
                      vehicle, {}, display_pos=[0, 0])
        Segment = SensorManager(world, display_manager, 'Segmentation', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 1])
        '''SensorManager(world, display_manager, 'Segmentation', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=00)), 
                      vehicle, {}, display_pos=[0, 2])
'''


        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()



            # Envia frame para a função de visão computacional
            frame = Segment.rgb_frame
            Erro = computer_vision(frame)
            control_main(vehicle, Erro)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
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
