    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import numpy as np
import cv2

def get_roi(image):

    height, width = image.shape
    left_img = image[0:height, 0:int(width/2)]
    right_img = image[0:height, int(width/2):width]
    half = np.zeros_like(left_img)
    left_img = np.concatenate((left_img, half), axis=1)
    right_img = np.concatenate((half, right_img), axis=1)

    return left_img, right_img

 

def skeletize_image(img, cross_size = 3):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (cross_size,cross_size))
    while cv2.countNonZero(img) != 0:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    return skel


def hough_transform(image):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 40  # minimal of votes
    #line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    #line_segments = cv2.HoughLines(cropped_edges, rho, angle, min_threshold, np.array([]))
    line_segments =cv2.HoughLines(image, rho, angle, min_threshold, None, 0, 0)

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            rho = rho - 360*np.cos(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(frame, pt1, pt2, line_color, line_width, cv2.LINE_AA)
    #cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_width)


def display_lines_2pts(frame, pt1, pt2, line_color=(0, 255, 0), line_width=2):
    #line_image = np.zeros_like(frame)

    cv2.line(frame, [int(pt1[0]),int(pt1[1])], [int(pt2[0]),int(pt2[1])], line_color, line_width)
    #cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_width)
    #line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def filter_by_angle(lines, sin_max = 0.5):

    ok_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            sin_theta = np.sin(theta)
            #print(sin_theta)
            if sin_max > abs(sin_theta):
                #print(line)
                ok_lines.append(np.array(line))

    lines = np.array(ok_lines)
    return lines

def filter_out_of_roi(lines, low = 360, high = 1080):

    ok_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            base = rho*(1/np.cos(theta)) - 720*np.sin(theta)
            #print('base',base)
            if base > low and base < high:
                #print(line)
                ok_lines.append(np.array(line))

    lines = np.array(ok_lines)
    return lines


def get_average_line(line_list):

    if line_list is not None:
        if len(line_list) != 0:
        # xs = [3, 7, 6]
        # mean = xs[0]
        # n = 1
        #     while n < len(xs):
        #     n += 1
        #     mean = mean*(i-1)/i + xs[i-1]/i
            avg = [np.mean(line_list, axis=0, dtype=np.float32)]
            return avg
    #print('avg', avg)
    return []


def normalize_hough(lines):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if rho < 0:
                rho = (-rho)
                theta = (theta - np.pi)
            line[0] = rho, theta
    return lines

def shift_origin(lines, shift = 360):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if theta > np.pi/2:
                rho = - rho + abs(shift*np.cos(theta))
                theta = theta - np.pi
            else:
                rho = shift*np.cos(theta) + rho # mudança de origem

            line[0] = rho, theta
    return lines

def return_origin(lines):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            rho = rho - 360*np.cos(theta) # mudança de origem
            line[0] = rho, theta
    return lines


class Accumulator:
    def __init__(self, accum_max_size):
        
        # Variáveis para armazenar a média temporal. 
        # São inicializadas com valor de faixa ideal.
        self.left_line_accum = [np.array([[560.        ,   0]], dtype=np.float32)]
        self.right_line_accum = [np.array([[880.       ,   0]], dtype=np.float32)]
        self.accum_max_size = accum_max_size

    def accumulate(self, left_line, right_line):


        # print('antes')
        # print(left_line[0][0], self.left_line_accum[0][0])

        # Caso faixa seja diferente da ultima
        # if left_line[0][0][0] != self.left_line_accum[0][0][0]:

        self.left_line_accum.append(left_line[0])

        # else:
        #     print('sao iguais')
        #     print(left_line[0][0], self.left_line_accum[0][0])

        # if right_line[0][0][0] != self.right_line_accum[0][0][0]:

        self.right_line_accum.append(right_line[0])

        
        # deleta primeiro termo se tiver mais q 'accum_max' linhas
        if len(self.left_line_accum) > self.accum_max_size:
            #print('antes do pop',self.left_line_accum)
            self.left_line_accum.pop(0)
            #print('depois do pop',self.left_line_accum)

        if len(self.right_line_accum) > self.accum_max_size:
            #print('antes do pop',self.left_line_accum)
            self.right_line_accum.pop(0)
            #print('depois do pop',self.left_line_accum)
        #tira a média dos valores na lista do acumulador

        #print('left_accum_avg',self.left_line_accum)
        #print(self.left_line_accum)
        left_accum_avg = get_average_line(self.left_line_accum)

        right_accum_avg = get_average_line(self.right_line_accum)
        
        #print('media', left_accum_avg)
        #print("self.left_line_accum",self.left_line_accum, "type", type(self.left_line_accum))
        #print('lista',self.left_line_accum,'len',len(self.left_line_accum))
        return left_accum_avg, right_accum_avg

class Holder:
    def __init__(self):
        
        # Variáveis para armazenar a faixa atual 
        # São inicializadas com valor de faixa ideal.
        self.left_line = [np.array([[560.        ,   0]], dtype=np.float32)]
        self.right_line = [np.array([[880.       ,   0]], dtype=np.float32)]

    def hold(self, left_line, right_line):


        # Caso faixa não esteja vazia
        # salva a faixa atual nas variaveis instanciadas na classe
        if len(left_line) != 0:
            self.left_line = left_line

        if len(right_line) != 0:
            self.right_line = right_line

        
        
        return self.left_line, self.right_line

class DifferenceFilter:

    def __init__(self, theta_lim = 0.5, rho_lim=200, count_lim=25):
        
        # Valores padrão das linhas
        self.left_antiga = [np.array([[560.        ,   0]], dtype=np.float32)]
        self.right_antiga = [np.array([[880.       ,   0]], dtype=np.float32)]
        
        # contadores para após x linhas ignoradas ele forçar pegar a nova
        self.l_count = 0
        self.r_count = 0

        # thresholds de diferença para excluir a linha nova
        self.theta_lim = theta_lim 
        self.rho_lim = rho_lim 
        self.count_lim = count_lim 
        # para 10 m/s count_lim é 15, para 30 m/s count_lim é 8

    def filter_strange_line(self, left_line, right_line):


        # if left_antiga is None:
        #     left_antiga = left_line


        rho_l, theta_l = left_line[0][0]
        rho_l_a, theta_l_a = self.left_antiga[0][0]

        # Compara a diferença absoluta entre rho e theta da linha antiga e nova
        if (abs(rho_l - rho_l_a) < self.rho_lim and abs(theta_l - theta_l_a) < self.theta_lim) or self.l_count > self.count_lim:   # Se dif rho for menor q rho_lim e dif theta menor q theta_lim
            left_ok = left_line # usa linha nova
            self.left_antiga = left_line # armazena linha nova
            self.l_count = 0 # zera contador sempre que utilizar linha nova
        else: # se for muito diferente da linha antiga
            left_ok = self.left_antiga # Pega a faixa antiga
            self.l_count = self.l_count + 1 # incrementa contador quando utilizar linha antiga
            #print('pegou LEFT antiga, count',self.l_count, left_line, self.left_antiga)


        # if right_antiga is None:
        #     right_antiga = right_line



        rho_r, theta_r = right_line[0][0]
        rho_r_a, theta_r_a = self.right_antiga[0][0]

        # Compara a diferença absoluta entre rho e theta da linha antiga e nova
        if (abs(rho_r - rho_r_a) < self.rho_lim and abs(theta_r - theta_r_a) < self.theta_lim) or self.r_count > self.count_lim:   # Se dif rho for menor q rho_lim e dif theta menor q theta_lim
            right_ok = right_line # usa linha nova
            self.right_antiga = right_line # armazena linha nova
            self.r_count = 0 # zera contador sempre que utilizar linha nova
        else: # se for muito diferente da linha antiga
            right_ok = self.right_antiga # Pega a faixa antiga
            self.r_count = self.r_count + 1  # incrementa contador quando utilizar linha antiga
            #print('pegou RIGHT antiga, count',self.r_count, right_line, self.right_antiga)    


        return left_ok, right_ok

def intersection(line1, line2):

    rho1, theta1 = line1[0][0]
    rho2, theta2 = line2[0][0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]



def compute_error(center_line):
    if len(center_line) != 0:

        rho, psi = center_line[0][0]

        intersec = intersection([[[rho, psi]]],[[[865, 1.57059]]])
        #print(intersec)
        del_x = (intersec[0] - 720)*np.cos(psi)*0.002084 # transformado para metros

        return np.rad2deg(psi), del_x
    return  0, 0


def get_mid_line(left_line, right_line):
    #print('left', left_line[0])
    if left_line  is not None and right_line is not None:
        if len(left_line) != 0 and len(right_line) != 0:

            rho1, theta1 = left_line[0][0]
            rho2, theta2 = right_line[0][0]

            
            psi = (theta1 + theta2)/2 # yaw error
            rho = (rho1 + rho2)/2
            del_x = 0
            intersec = intersection([[[rho, psi]]],[[[865, 1.57059]]])
            #print(intersec)
            del_x = (intersec[0] - 360)*np.cos(psi)*0.002084 # transformado para metros

            return [[[rho, psi]]], np.rad2deg(psi), del_x
    return [[[0, 0]]], 0, 0
    
holder = Holder()

accum_pos = Accumulator(3)

diff = DifferenceFilter(theta_lim = 0.4, rho_lim=175, count_lim=10000)

def image_processing4(rgb_frame):

    ################################################
    #### TRATAMENTO E PROCESSAMENTO DE IMAGENS #####
    ################################################

    gray_img = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    bird_img = bird_eyes(gray_img)

    bird_img_blur = cv2.GaussianBlur(bird_img,(15,15),0)

    img_bin = adaptive_threshold(bird_img_blur, 11, -1)

    skel_img = skeletize_image(img_bin, 3) # esqueletiza a imagem

    ################################################
    ####### ALGORITMO DE VISÃO COMPUTACIONAL #######
    ########### PARA DETECTAR AS FAIXAS ############
    ################################################

    lines_in = hough_transform(skel_img) # todas as linhas detectadas 

    # Desloca origem em 360 pixels no eixo x
    shift_origin(lines_in)

    lines = filter_by_angle(lines_in) # descarta linhas com angulo muito horizontal

    if lines is not None:
        lines_shift = lines.copy()
    else:
        lines_shift = None

    normalize_hough(lines_shift)


    left_lines_shift = filter_out_of_roi(lines_shift, 360, 720-60)
    right_lines_shift = filter_out_of_roi(lines_shift, 720+60, 1080)

    left_line_shift = get_average_line(left_lines_shift)
    right_line_shift = get_average_line(right_lines_shift)


    # em caso de não detectar faixa, mantém a ultima encontrada
    left_line_shift, right_line_shift = holder.hold(left_line_shift, right_line_shift)
    
    # ignora as faixas muito diferentes da anterior
    left_line_shift, right_line_shift = diff.filter_strange_line(left_line_shift, right_line_shift)

    # média temporal das ultimas faixas
    left_line_shift, right_line_shift = accum_pos.accumulate(left_line_shift, right_line_shift)


    ################################################
    ########## OBTÉM CENTRO DA FAIXA PARA ##########
    ########## CALCULAR ERROS DO CONTROLE ##########
    ################################################

    center_line_shift = get_average_line([left_line_shift[0], right_line_shift[0]])
    
    psi, del_x = compute_error(center_line_shift)


    # Volta para origem antiga
    #left_line_shift = return_origin(left_line_shift)
    #right_line_shift = return_origin(right_line_shift)

    #left_lines_shift = return_origin(left_lines_shift)
    #right_lines_shift = return_origin(right_lines_shift)
    
    #center_line_shift = return_origin(center_line_shift)

    ################################################
    ############### Mostrar Imagens ################
    ################################################
    # converte para rgb
    
    # bird_img_rgb = cv2.cvtColor(bird_img,cv2.COLOR_GRAY2RGB)
    # img_bin_rgb = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    # gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    #skel_rgb = cv2.cvtColor(skel_img, cv2.COLOR_GRAY2RGB)
    #display_lines(skel_rgb, lines, line_color = (255,0,255), line_width=1)

    # tl = [60, 113]
    # tr = [660, 113]
    # br = [1065, 270]
    # bl = [-345, 270]

    # display_lines_2pts(gray_img_rgb, tl, tr, line_color = (0,21,200), line_width=1)
    # display_lines_2pts(gray_img_rgb, tr, br, line_color = (0,21,200), line_width=1)
    # display_lines_2pts(gray_img_rgb, br, bl, line_color = (0,21,200), line_width=1)
    # display_lines_2pts(gray_img_rgb, bl, tl, line_color = (0,21,200), line_width=1)

    # # mostra as linhas
    #display_lines(skel_rgb, left_lines_shift, line_color = (0,0,255), line_width=1)
    #display_lines(skel_rgb, right_lines_shift, line_color = (255,0,0), line_width=1)
    # ########## Mostrar as faixas ######
    # display_lines(skel_rgb, left_line_shift, line_color = (0,0,255))
    # display_lines(skel_rgb, right_line_shift, line_color = (255,0,0))
    # display_lines(skel_rgb, center_line_shift)
    # write_on_screen(skel_rgb, ('psi: '+str(round(psi,3))+' degree'), [370, 360], (0,255,0), size = 0.5, thick = 2)

    #     # del_x
    # display_lines_2pts(skel_rgb, [del_x/0.002084 + 360, 720], [360, 720], line_color = (51,251,255), line_width=3)
    # write_on_screen(skel_rgb, ('dx: '+str(round(del_x,3))+' m'), [int(round(del_x,0)) + 360,710], (51,251,255), size = 0.5, thick = 2) 
    #rgb_frame = cv2.resize(rgb_frame, (380,380))
    # gray_img_rgb = cv2.resize(gray_img_rgb, (380,380))
    #gray_img_res = cv2.resize(gray_img, (380,380))
    #bird_img_resize = cv2.resize(bird_img, (380,380))
    #bird_img_blur = cv2.resize(bird_img_blur, (380,380))
    #img_bin = cv2.resize(img_bin, (380,380))
    # skel_rgb = cv2.resize(skel_rgb, (380,380))
    # img_bin_rgb = cv2.resize(img_bin_rgb, (380,380))
    # bird_img_rgb = cv2.resize(bird_img_rgb, (380,380))


    # cv2.imshow('Camera', rgb_frame)
    #cv2.imshow('Imagem grayscale', gray_img)
    # cv2.imshow('Imagem grayscale e ROI', gray_img_rgb)
    # cv2.imshow('Transformacao de Perspectiva', bird_img_resize)
    # cv2.imshow('Imagem apos filtro Gaussiano', bird_img_blur)
    # cv2.imshow('Imagem Binarizada', img_bin)
    # cv2.imshow('Imagem Esqueletizada', skel_img)
    # cv2.imshow('Todas as Linhas', img_bin_rgb)
    # cv2.imshow('Esquerda, Direita e Medias', bird_img_rgb)
    #cv2.imwrite('gray_camera_view.png',gray_img)


    return  bird_img, left_line_shift, right_line_shift, center_line_shift, psi, del_x




def computer_vision_rgb(rgb_frame, data):
    #seg_frame = data.frame
    if rgb_frame is None:
        rgb_frame = np.full((720,720,3),255)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    #frame = np.zeros((720,720,3))
    #show_image_rgb(frame) # Mostra imagem RGB

    data.frame, data.left_line, data.right_line, data.mid_line, data.psi, data.dx = image_processing4(rgb_frame)
    #rec_frame = control_monitor(data)
    #rec_frame = cv2.resize(rec_frame, (380,380))
    #image_processing_kmeans(mask)
    #print('asdasd',left_line, right_line)
    #return rec_frame
# classe para armazenar os dados da simulação e visão
class SimulationData:
    def __init__(   
                    self, frame = None, 
                    left_line = None, 
                    right_line = None, 
                    mid_line = None, 
                    psi = 0,
                    dx = 0, 
                    steering = 0, 
                    control_output = 0,
                    Kp_dx = 0, 
                    Ki_dx = 0, 
                    velocidade = 0
                ):

        self.frame = frame
        self.left_line = left_line
        self.right_line = right_line
        self.mid_line = mid_line
        self.psi = psi
        self.dx = dx
        self.steering = steering
        self.control_output = control_output
        self.Kp_dx = Kp_dx
        self.Ki_dx = Ki_dx
        self.velocidade = velocidade


def control_monitor(data):

    frame = data.frame
    mid_line = data.mid_line


    if frame is None:
        frame = np.zeros((720,720))

    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    
    if not(isinstance(data.left_line, int)):

        # centro da camera (em azul)
        #display_lines_2pts(frame, [360,0], [360,720], line_color = (200,21,21), line_width=1)
        #display_lines_2pts(frame, [0,360], [720,360], line_color = (200,21,21), line_width=1)

        # faixas e centro
        display_lines(frame, data.left_line, line_color = (0,0,255))
        display_lines(frame, data.right_line, line_color = (255,0,0))
        display_lines(frame, data.mid_line)

        write_on_screen(frame, ('psi: '+str(round(data.psi,3))+' degree'), [370, 360], (0,255,0), size = 0.5, thick = 2)

        # del_x
        display_lines_2pts(frame, [data.dx/0.002084 + 360, 720], [360, 720], line_color = (51,251,255), line_width=3)
        write_on_screen(frame, ('dx: '+str(round(data.dx,3))+' m'), [int(round(data.dx,0)) + 360,710], (51,251,255), size = 0.5, thick = 2) 

    # write_on_screen(frame, ('Steering:'+str(round(data.steering,4))), (10,50), (255,255,255))
    # if  data.steering > 0:
    #     write_on_screen(frame, ('Direita'), (500,50), (255,0,0))
    # else:
    #     write_on_screen(frame, ('Esquerda'), (500,50), (0,255,255))
    # write_on_screen(frame, ('Controle:'+str(np.round(data.control_output,4))), (10,100), (255,255,255))
    # # write_on_screen(frame, ('Estado:'+str(data.estado)), (10,150), (255,255,255)) 
    # # write_on_screen(frame, ('Kp:'+str(data.Kp)), (10,200), (50,50,255))  
    # # write_on_screen(frame, ('Kd:'+str(data.Kd)), (10,250), (50,50,255))    
    # # write_on_screen(frame, ('Ki:'+str(data.Ki)), (10,300), (50,50,255))
    # write_on_screen(frame, ('Vel:'+str(data.velocidade)), (10,350), (50,255,50))
      
    cv2.imshow('Lane Monitor', frame)

    #data.frame = frame
    
    cv2.waitKey(1)
    #data.frame = frame
    return frame
    

def write_on_screen(frame, text, pos, color, size = 1, thick = 1):
    cv2.putText(frame, (text), pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, 2)  






def adaptive_threshold(gray_img, block_size = 21, const = 5):

    #cv2.imshow('gray roi eq', gray_img)
    #ret, thresh_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh1 = cv2.adaptiveThreshold(gray_img, 254, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, const)


    #plt.show()
    return thresh_img


def bird_eyes(image):
    # # targeted rectangle on original image which needs to be transformed
    # tl = [60, 113]
    # tr = [660, 113]
    # br = [1065, 270]
    # bl = [-345, 270]

    # corner_points_array = np.float32([tl,tr,br,bl])


    # # # Create an array with the parameters (the dimensions) required to build the matrix
    # imgTl = [0, 0]
    # imgTr = [720, 0]
    # imgBr = [720, 720]
    # imgBl = [0, 720]
    # img_params = np.float32([imgTl,imgTr,imgBr,imgBl])

    # # # Compute and return the transformation matrix
    # matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
    # print(matrix)
    matrix = np.array([ [ 4.23370787e+01,  1.09213483e+02, -1.48813483e+04],
                        [ 0.00000000e+00,  3.80224719e+02, -4.29653933e+04],
                        [ 1.31581988e-19,  3.03370787e-01,  1.00000000e+00],])
    
    img_transformed = cv2.warpPerspective(image,matrix,(720, 720), borderMode=cv2.BORDER_REPLICATE)
    #display_lines_2pts(img_transformed, [360,0], [360,720], line_color = (200,21,21), line_width=1)
    #display_lines_2pts(img_transformed, [0,360], [720,360], line_color = (200,21,21), line_width=1)

    return img_transformed





if __name__ == '__main__':


    path = 'test_img\camera_view.png'
    #path = 'test_img\curva_fov30_left.png'
    #path = 'test_img\curva_fov30_right_brusca.png'

    #path = 'test_img\sombra.png'
    #path = 'test_img\sombra2.png'


    img_BGR = cv2.imread(path, cv2.IMREAD_COLOR)

    # video = cv2.VideoCapture('recordFFV1.avi')

    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    # video_out = cv2.VideoWriter("videos2\\10_result.avi", fourcc, 24, (380,  380), isColor = True)


    data = SimulationData()

    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = video.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     # Our operations on the frame come here
    #     # Display the resulting frame
    #     rec_frame = computer_vision_rgb(frame, data)
    #     cv2.imshow('tresh', rec_frame)
    #     video_out.write(rec_frame)

    #     if cv2.waitKey(1) == ord('q'):
    #         break

    # video_out.release()


    for n in range(1):

        

        computer_vision_rgb(img_BGR, data)

        cv2.waitKey(0)

        cv2.destroyAllWindows()



 
    cv2.waitKey(0)
    cv2.destroyAllWindows()





