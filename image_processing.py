    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_roi(image):

    height, width = image.shape
    half = np.zeros_like
    left_img = image[0:int(height), 0:int(width/2)]
    right_img = image[0:height, int(width/2):width]
    half = np.zeros_like(left_img)
    left_img = np.concatenate((left_img, half), axis=1)
    right_img = np.concatenate((half, right_img), axis=1)

    return left_img, right_img

 

def skeletize_image(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
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
    angle = np.pi / 360  # angular precision in radian, i.e. 1 degree
    min_threshold = 35  # minimal of votes
    #line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    #line_segments = cv2.HoughLines(cropped_edges, rho, angle, min_threshold, np.array([]))
    line_segments =cv2.HoughLines(image, rho, angle, min_threshold, None, 0, 0)

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    if lines is not None:
            for line in lines:
                rho, theta = line[0]
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


def filter_by_angle(lines, deg_max = 20):

    ok_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            deg_var = abs(np.rad2deg(theta))-180
            if deg_max > deg_var:
                ok_lines.append(np.array(line))

    ok_lines = np.array(ok_lines)
    return ok_lines


def get_average_line(line_list):

    avg = [np.mean(line_list, axis=0,dtype=np.float32)]


    print('avg', avg)
    return avg


class Accumulator:
    def __init__(self, accum_max_size):
        
        # Variáveis para armazenar a média temporal. 
        # São inicializadas com valor de faixa ideal.
        self.left_line_accum = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]
        self.right_line_accum = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]
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
        self.left_line = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]
        self.right_line = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]

    def hold(self, left_line, right_line):


        # Caso faixa não esteja vazia
        # salva a faixa atual nas variaveis instanciadas na classe
        if len(left_line) != 0:
            self.left_line = left_line

        if len(right_line) != 0:
            self.right_line = right_line

        
        
        return self.left_line, self.right_line



left_antiga = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]
right_antiga = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]


# contadores para após x linhas ignoradas ele forçar pegar a nova
l_count = 0
r_count = 0

# thresholds de diferença para excluir a linha nova
theta_lim = 0.20
rho_lim = 100
count_lim = 25
# para 10 m/s count_lim é 15, para 30 m/s count_lim é 8

def filter_strange_line(left_line, right_line):

    global left_antiga, right_antiga, l_count, r_count

    # if left_antiga is None:
    #     left_antiga = left_line


    rho_l, theta_l = left_line[0][0]
    rho_l_a, theta_l_a = left_antiga[0][0]

    # Compara a diferença absoluta entre rho e theta da linha antiga e nova
    if (abs(rho_l - rho_l_a) < rho_lim and abs(theta_l - theta_l_a) < theta_lim) or l_count > count_lim:   # Se dif rho for menor q rho_lim e dif theta menor q theta_lim
        left_ok = left_line # usa linha nova
        left_antiga = left_line # armazena linha nova
        l_count = 0 # zera contador sempre que utilizar linha nova
    else: # se for muito diferente da linha antiga
        left_ok = left_antiga # Pega a faixa antiga
        l_count = l_count + 1 # incrementa contador quando utilizar linha antiga
        print('pegou LEFT antiga, count',l_count, left_line, left_antiga)


    # if right_antiga is None:
    #     right_antiga = right_line



    rho_r, theta_r = right_line[0][0]
    rho_r_a, theta_r_a = right_antiga[0][0]

    # Compara a diferença absoluta entre rho e theta da linha antiga e nova
    if (abs(rho_r - rho_r_a) < rho_lim and abs(theta_r - theta_r_a) < theta_lim) or r_count > count_lim:   # Se dif rho for menor q rho_lim e dif theta menor q theta_lim
        right_ok = right_line # usa linha nova
        right_antiga = right_line # armazena linha nova
        r_count = 0 # zera contador sempre que utilizar linha nova
    else: # se for muito diferente da linha antiga
        right_ok = right_antiga # Pega a faixa antiga
        r_count = r_count + 1  # incrementa contador quando utilizar linha antiga
        print('pegou RIGHT antiga, count',r_count, right_line, right_antiga)    


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


def get_mid_line(left_line, right_line):
    #print('left', left_line[0])
    if left_line  is not None and right_line is not None:
        rho1, theta1 = left_line[0][0]
        rho2, theta2 = right_line[0][0]
        print(left_line, right_line)
        
        psi = (theta1 + theta2)/2 # yaw error
        rho = (rho1 + rho2)/2
        del_x = 0


        return [[[rho, psi]]], np.rad2deg(psi) - 180, del_x

    
holder = Holder()
accum_pre = Accumulator(2)
accum_pos = Accumulator(7)



def image_processing4(rgb_frame):

    bird_img = bird_eyes(rgb_frame)

    cv2.imshow('birds', bird_img)

    img_bin = adaptive_threshold(bird_img)

    skel_img = skeletize_image(img_bin) # esqueletiza a imagem

    left_img, right_img = get_roi(skel_img)

    #vis = np.concatenate((left_img, right_img), axis=1)

    cv2.imshow('skel img', skel_img)


    left_lines = hough_transform(left_img) # todas as linhas detectadas 
    right_lines = hough_transform(right_img)

    left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)

    display_lines(left_img, left_lines, line_color = (0,0,255), line_width=1)
    display_lines(right_img, right_lines, line_color = (0,0,255), line_width=1)

    cv2.imshow('left', left_img)
    cv2.imshow('right', right_img)

    left_lines = filter_by_angle(left_lines) # descarta linhas com angulo muito horizontal
    right_lines = filter_by_angle(right_lines) # descarta linhas com angulo muito horizontal


    #left_lines, right_lines  = sort_left_right(lines)


    left_line = get_average_line(left_lines)
    right_line = get_average_line(right_lines)

    #print(left_line, right_line)
    ########## Mostrar as faixas ######
    # converte para rgb
    roi_img_rgb = cv2.cvtColor(skel_img,cv2.COLOR_GRAY2RGB)

    # mostra as linhas
    display_lines(roi_img_rgb, left_lines, line_color = (0,0,255), line_width=1)
    display_lines(roi_img_rgb, right_lines, line_color = (0,0,255), line_width=1)
    display_lines(roi_img_rgb, left_line)
    display_lines(roi_img_rgb, right_line)
 
    cv2.imshow('Hough Lines and Lane', roi_img_rgb)
    ########## Mostrar as faixas ######



    left_line, right_line = holder.hold(left_line, right_line)

    #left_line, right_line = accum_pre.accumulate(left_line, right_line)

    # filtrar antes de pegar a média?
    #left_line, right_line = filter_strange_line(left_line, right_line)

    #left_line, right_line = accum_pos.accumulate(left_line, right_line)



    # encontra os parâmetros
    mid_line, psi, del_x = get_mid_line(left_line, right_line)



    return bird_img, left_line, right_line, mid_line, psi, del_x




def computer_vision_rgb(rgb_frame, data):
    #seg_frame = data.frame
    if rgb_frame is None:
        rgb_frame = np.zeros((720,720,3))
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    #frame = np.zeros((720,720,3))
    #show_image_rgb(frame) # Mostra imagem RGB

    data.frame, data.left_line, data.right_line, data.mid_line, data.psi, data.dx = image_processing4(rgb_frame)
    control_monitor(data)
    #image_processing_kmeans(mask)
    #print('asdasd',left_line, right_line)

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
                    Kp_theta = 0,
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
        self.Kp_theta = Kp_theta
        self.Kp_dx = Kp_dx
        self.Ki_dx = Ki_dx
        self.velocidade = velocidade


def control_monitor(data):

    frame = data.frame
    mid_line = data.mid_line


    if frame is None:
        frame = np.zeros((720,720,3))
    
    
    if not(isinstance(data.left_line, int)):

        # centro da camera (em azul)
    
        display_lines_2pts(frame, [360,0], [360,720], line_color = (200,21,21), line_width=1)
        display_lines_2pts(frame, [0,360], [720,360], line_color = (200,21,21), line_width=1)

        # linhas (em verde)
        display_lines(frame, data.left_line)
        display_lines(frame, data.right_line)
        display_lines(frame, data.mid_line, line_color = (255,0,255))
        # triangulo (em magenta)
        #display_lines_2pts(frame, bisec_pt, intersec, line_color = (255,0,255), line_width=1)
        #display_lines_2pts(frame, [intersec[0],bisec_pt[1]], intersec, line_color = (255,0,255), line_width=1)
        #display_lines_2pts(frame, [360, bisec_pt[1]-1], [intersec[0], bisec_pt[1]-1], line_color = (255,0,255), line_width=1)
        write_on_screen(frame, ('Psi: '+str(round(data.psi,3))+' degree'), [360, 360], (255,0,255), size = 0.5, thick = 2)

        # del_x
        #display_lines_2pts(frame, bisec_pt, [360, bisec_pt[1]], line_color = (51,251,255), line_width=3)
        write_on_screen(frame, ('D_x: '+str(round(data.dx,3))), [40, 20], (51,251,255), size = 0.5, thick = 2) 

    write_on_screen(frame, ('Steering:'+str(round(data.steering,5))), (10,50), (255,255,255))
    if  data.steering > 0:
        write_on_screen(frame, ('Direita'), (500,50), (255,0,0))
    else:
        write_on_screen(frame, ('Esquerda'), (500,50), (0,255,255))
    # write_on_screen(frame, ('Estado:'+str(data.estado)), (10,100), (255,255,255))
    # write_on_screen(frame, ('Estado:'+str(data.estado)), (10,150), (255,255,255)) 
    # write_on_screen(frame, ('Kp:'+str(data.Kp)), (10,200), (50,50,255))  
    # write_on_screen(frame, ('Kd:'+str(data.Kd)), (10,250), (50,50,255))    
    # write_on_screen(frame, ('Ki:'+str(data.Ki)), (10,300), (50,50,255))
    write_on_screen(frame, ('Vel:'+str(data.velocidade)), (10,350), (50,255,50))
      
    cv2.imshow('rgb with lines', frame)

    #data.frame = frame
    
    cv2.waitKey(1)
    #data.frame = frame
    

def write_on_screen(frame, text, pos, color, size = 1, thick = 1):
    cv2.putText(frame, (text), pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, 2)  


def adaptive_threshold(rgb_img):

    #cv2.imshow('rgb image', rgb_img)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('gray image', gray_img)
    gray_img = cv2.GaussianBlur(gray_img,(7,7),0)
    #roi_img_rgb, ROI = get_roi(gray_img, 1)
    #cv2.imshow('roi img rgb', roi_img_rgb)
    #cv2.imshow('gray blurred', gray_img)


    cv2.imshow('roi img', gray_img)

    # gray_img_eq = cv2.equalizeHist(gray_img[ROI])
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    # gray_img_eq = clahe.apply(gray_img[ROI])

    # gray_img[ROI] = gray_img_eq.reshape(-1)



    #cv2.imshow('gray roi eq', gray_img)
    #ret, thresh1 = cv2.threshold(roi_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh1 = cv2.adaptiveThreshold(gray_img, 254, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    thresh_img = cv2.adaptiveThreshold(gray_img, 254, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)
    cv2.imshow('tresh img', thresh_img)



    #plt.show()
    return thresh_img

def bird_eyes(image):
    # targeted rectangle on original image which needs to be transformed
    tl = [145, 80]
    tr = [575, 80]
    br = [720, 175]
    bl = [0, 175]

    corner_points_array = np.float32([tl,tr,br,bl])


    # Create an array with the parameters (the dimensions) required to build the matrix
    imgTl = [0, 0]
    imgTr = [720, 0]
    imgBr = [720, 720]
    imgBl = [0, 720]
    img_params = np.float32([imgTl,imgTr,imgBr,imgBl])

    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
    img_transformed = cv2.warpPerspective(image,matrix,(720, 720))
    #display_lines_2pts(img_transformed, [360,0], [360,720], line_color = (200,21,21), line_width=1)
    #display_lines_2pts(img_transformed, [0,360], [720,360], line_color = (200,21,21), line_width=1)

    return img_transformed


if __name__ == '__main__':

    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_color.png'
    #path = 'static_road_angle.png'
    #path = 'static_road.png'
    #path = 'perfeito.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_left_only.png'
    #path = 'line2.png'
    #path = 'color_curva_suave.png'
    #path = 'color_curva.png'
    #path = 'static_road_color.png'
    #path = 'ideal_fov30.png'
    path = 'curva_fov30_left.png'
    path = 'curva_fov30_right.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\imglank.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\svanish.png'
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_BGR = cv2.imread(path, cv2.IMREAD_COLOR)

    #image_processing(img_gray)
    #cv2.waitKey(0)
    data = SimulationData()
    for n in range(1):

        #image_processing_kmeans(img_gray)
        computer_vision_rgb(img_BGR, data)
        #control_monitor(img_BGR, 1, 2, 1, 3, 4, 5, 6, 7)
        #adaptive_threshold(img_BGR)
        bird_eyes(img_BGR)
        cv2.waitKey(0)

        cv2.destroyAllWindows()





    #img = get_roi(img_gray)
    #img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
    #cv2.imshow('image',img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()





