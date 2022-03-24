    #!/usr/bin/env python

# Feito por Victor de Mattos Arzolla

import numpy as np
import cv2

def get_roi(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, 560),
        (0, 720),
        (720, 720),
        (720, 560),
        (400, 390),
        (320, 390),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def get_roi_half(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, 390),
        (0, 720),
        (720, 720),
        (720, 390),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def draw_vanishing_point(img):
    start_point_1 = (97, 720)
    end_point_1 = (623, 0)

    start_point_2 = (623, 720)
    end_point_2 = (97, 0)
    

    # linhas 505 e 560

    # Green color in BGR
    color = (255, 255, 255)
    
    # Line thickness of 9 px
    thickness = 1


    vp_img = cv2.line(img, start_point_1, end_point_1, color, thickness)
    vp_img = cv2.line(vp_img, start_point_2, end_point_2, color, thickness)

    return vp_img




def get_mask(image):
    roadline_color = (50, 234, 157) # in bgr
    image = np.ascontiguousarray(image, dtype=np.uint8)
    mask = cv2.inRange(image, roadline_color, roadline_color)
    #cv2.line(image, (0, 0), (200, 400), (0, 255, 0), thickness=2)
    #r_chan = image[:, :,2]
    #mask = cv2.inRange(r_chan, 6, 6)
    #print(r_chan)
    #vehicle.andar_para_o_lado
    #cv2.imshow("", mask)
    #cv2.waitKey(1)
    return mask


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
    min_threshold = 30  # minimal of votes
    #line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)
    #line_segments = cv2.HoughLines(cropped_edges, rho, angle, min_threshold, np.array([]))
    line_segments =cv2.HoughLines(image, rho, angle, min_threshold, None, 0, 0)

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                cv2.line(line_image, pt1, pt2, line_color, line_width, cv2.LINE_AA)
    #cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def filter_vertical_lines(lines, sine_limit=0.9):

    ok_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0] 

            if np.sin(theta) < sine_limit:
                ok_lines.append(np.array(line))

    ok_lines = np.array(ok_lines)
    return ok_lines

def get_average_line(line_list):
    rho_sum = 0
    theta_sum = 0
    avg = []
    #print('linelist',line_list, np.shape(line_list), type(line_list))
    if line_list != []:
        for line in line_list:
            rho, theta = line[0]

            rho_sum += rho
            theta_sum += theta
        #print('list', list,' len(list)', len(list) )
        avg.append([[rho_sum/len(line_list), theta_sum/len(line_list)]])

    avg = np.array(avg)
    #print('avg', avg)
    return avg


def sort_left_right(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        rho, theta = line[0]

        if theta < np.pi/2:
            left_lines.append(line)
        else:
            right_lines.append(line)

    left_lines = np.array(left_lines)
    right_lines = np.array(right_lines)
    return left_lines, right_lines



# Variáveis para armazenar a média temporal. 
# São inicializadas com valor de faixa ideal.
right_line_accum = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]
left_line_accum = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]

left_antiga = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]
right_antiga = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]


def accumulator(left_line, right_line):

    global right_line_accum, left_line_accum
    #print('recebido', left_line, right_line)

    # Caso faixa seja vazia
    if left_line != []:
        left_line_accum.append(left_line[0])

    if right_line != []:
        right_line_accum.append(right_line[0])

    
    accum_max_size = 5
    
    # deleta primeiro termo se tiver mais q 5 linhas
    if len(left_line_accum) > accum_max_size:
        #print('antes do pop',left_line_accum)
        left_line_accum.pop(0)
        #print('depois do pop',left_line_accum)

    if len(right_line_accum) > accum_max_size:
        #print('antes do pop',left_line_accum)
        right_line_accum.pop(0)
        #print('depois do pop',left_line_accum)
    #tira a média dos valores na lista do acumulador

    #print('left_accum_avg',left_line_accum)
    #print(left_line_accum)
    left_accum_avg = get_average_line(left_line_accum)

    right_accum_avg = get_average_line(right_line_accum)
    
    #print('media', left_accum_avg)
    #print("left_line_accum",left_line_accum, "type", type(left_line_accum))
    #print('lista',left_line_accum,'len',len(left_line_accum))
    return left_accum_avg, right_accum_avg



def filter_strange_line(left_line, right_line):

    print(left_line)

    global left_antiga, right_antiga

    rho_l, theta_l = left_line[0][0]
    rho_l_a, theta_l_a = left_antiga[0][0]

    # Compara a diferença absoluta entre rho e theta da linha antiga e nova
    if abs(rho_l - rho_l_a) < 10 and abs(theta_l - theta_l_a) < 0.26:   # Se dif rho for menor q 10 e dif theta for pi/12
        left_ok = left_line # usa linha nova
        left_antiga = left_line # armazena linha nova
    else: # se for muito diferente da linha antiga
        left_ok = left_antiga # usa linha antiga

    rho_r, theta_r = right_line[0][0]
    rho_r_a, theta_r_a = right_antiga[0][0]

    # Compara a diferença absoluta entre rho e theta da linha antiga e nova
    if abs(rho_r - rho_r_a) < 10 and abs(theta_r - theta_r_a) < 0.26:   # Se dif rho for menor q 10 e dif theta for pi/12
        right_ok = right_line # usa linha nova
        right_antiga = right_line # armazena linha nova
    else: # se for muito diferente da linha antiga
        right_ok = right_antiga # usa linha antiga
    


    return left_ok, right_ok

def intersection(line1, line2):

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    #x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def get_bisector(left_line, right_line):
    #print('left', left_line[0])
    if left_line  is not None and right_line is not None:
        rho1, theta1 = left_line[0][0]
        rho2, theta2 = right_line[0][0]

        rho = -(rho1*np.sin(theta1) + rho2*np.cos(theta2)) 
        theta = theta1 + theta2 
        return [[[rho, theta]]]

def image_processing4(img_gray):
    roi_img = get_roi(img_gray)

    skel_img = skeletize_image(roi_img) # esqueletiza a imagem


    lines = hough_transform(skel_img) # todas as linhas detectadas 
    #print('liens saida hough',lines)

    lines = filter_vertical_lines(lines) # discarta linhas com angulo muito horizontal
    #print('liens filtrada',lines)
    skel_img_bgr = cv2.cvtColor(skel_img,  cv2.COLOR_GRAY2BGR)

  
    left_lines, right_lines  = sort_left_right(lines)

    #print('lines separadas',left_lines, right_lines)


 
    left_line = get_average_line(left_lines)
    right_line = get_average_line(right_lines)

    #print('liness',left_lines, np.shape(left_lines), type(left_lines))
    #print('right line', left_line, np.shape(left_line), type(left_line))

    #print('pros3 avg',left_line, right_line)



    left_line, right_line = accumulator(left_line, right_line)


    left_line, right_line = filter_strange_line(left_line, right_line)

    #intersecção das duas linhas

    #print('left e right o',left_line, right_line)
    
  
    bisector = get_bisector(left_line,right_line)
    #print('soma',mid_line)
    #mid_line = [[[-360, np.pi]]]

    skel_with_lines = display_lines(skel_img_bgr, lines, line_color = (0,0,255), line_width=1)

    skel_with_lines = display_lines(skel_with_lines, left_line)
    skel_with_lines = display_lines(skel_with_lines, right_line)

    skel_with_lines = display_lines(skel_with_lines, bisector, line_color = (255,0,255), line_width=1)

    cv2.imshow('processing4',skel_with_lines)
    return left_line, right_line, bisector


def control_monitor(frame, left_line, right_line, bisector, estado, steering, Kp, Kd, Ki, velocidade):
    if frame is None:
        frame = np.zeros((720,720,3))
 
    if not(isinstance(left_line, int)):
        frame = display_lines(frame, left_line)
        frame = display_lines(frame, right_line)
        frame = display_lines(frame, bisector, line_color = (255,0,255))

    write_on_screen(frame, ('Steering:'+str(steering)), (10,50), (255,255,255)) 
    write_on_screen(frame, ('Estado:'+str(estado)), (10,100), (255,255,255)) 
    write_on_screen(frame, ('Kp:'+str(Kp)), (10,150), (50,50,255))  
    write_on_screen(frame, ('Kd:'+str(Kd)), (10,200), (50,50,255))    
    write_on_screen(frame, ('Ki:'+str(Ki)), (10,250), (50,50,255))
    write_on_screen(frame, ('Vel:'+str(velocidade)), (10,300), (50,255,50))
      
    cv2.imshow('rgb with lines',frame)

def write_on_screen(frame, text, pos, color):
    cv2.putText(frame, (text), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, 2)  

def show_image_rgb(rgb):
    cv2.imshow('image', rgb)

if __name__ == '__main__':

    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_color.png'
    path = 'static_road.png'
    #path = 'perfeito.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_left_only.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\line2.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\imglank.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\svanish.png'
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_BGR = cv2.imread(path, cv2.IMREAD_COLOR)

    #image_processing(img_gray)
    #cv2.waitKey(0)
       
    for n in range(1):

        #image_processing_kmeans(img_gray)
        image_processing4(img_gray)
        #control_monitor(img_BGR, 1, 2, 1, 3, 4, 5, 6, 7)
        
        cv2.waitKey(0)

        cv2.destroyAllWindows()





    #img = get_roi(img_gray)
    #img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
    #cv2.imshow('image',img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()





