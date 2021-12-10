    #!/usr/bin/env python



import numpy as np

#bgr - rgb
import cv2
import matplotlib.pyplot as plt


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

# acha o centro da linha
def centro_linha(linha):

    maxElement = np.amax(linha)
    print('Max element from Numpy Array : ', maxElement)

    result = np.where(linha == np.amax(linha))
    print('Returned tuple of arrays :', result)
    print('List of Indices of maximum element :', result[0])
    
    left_sum = 0 # soma do indice dos pixels de valor maximo
    right_sum = 0
    len_left = 0
    len_right = 0

    for value in result[0]:
        if value < 360:
            left_sum += value
            len_left += 1
        else:
            right_sum += value
            len_right += 1

    if len_left == 0:
        left_1 =0
    else:
        left_1 = left_sum/len_left

    if len_right == 0:
        right_1 = 0
    else:
        right_1 = right_sum/len_right


    print('Media left', left_1)
    print('Media right', right_1)
    return left_1, right_1

# 
def histo(img_pb, altura_linha):

    linha = img_pb[altura_linha,:]
 

    '''
    plt.figure()
    plt.plot(range(len(linha_1)), linha_1)
    plt.plot(range(len(linha_2)), linha_2)
    plt.show()
    #retorna lista de indices com valor maximo
    '''
    return linha

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    cropped_edges = cv2.Canny(cropped_edges, 200, 400)
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

# main para processar imagem
def image_processing(img_gray):

    
    roi_img = get_roi(img_gray)
    linhas = detect_line_segments(roi_img)
    print(linhas)

    roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    

    vp_img = draw_vanishing_point(np.zeros_like(roi_img_rgb))
    cv2.imshow('image',vp_img)
    print(vp_img.shape)
    #vp_img = get_roi(vp_img)

    y_linha_1 = 560
    y_linha_2 = 505

    index_linha_1 = histo(img_gray, y_linha_1)
    index_linha_2 = histo(img_gray, y_linha_2)

    left_x_linha_1, right_x_linha_1 = centro_linha(index_linha_1)
    left_x_linha_2, right_x_linha_2 = centro_linha(index_linha_2)


    left_p_1 = [int(left_x_linha_1), y_linha_1]
    left_p_2 = [int(left_x_linha_2), y_linha_2]

    right_p_1 = [int(right_x_linha_1), y_linha_1]
    right_p_2 = [int(right_x_linha_2), y_linha_2]

    vp_pontos_img = cv2.circle(vp_img, (left_p_1), radius=6, color=(0, 0, 255), thickness=-1)
    vp_pontos_img = cv2.circle(vp_pontos_img, (left_p_2), radius=6, color=(0, 0, 255), thickness=-1)

    vp_pontos_img = cv2.circle(vp_pontos_img, (right_p_1), radius=6, color=(0, 0, 255), thickness=-1)
    vp_pontos_img = cv2.circle(vp_pontos_img, (right_p_2), radius=6, color=(0, 0, 255), thickness=-1)


    vp_pontos_line_img = cv2.line(vp_pontos_img, left_p_1, left_p_2, color=(255, 0, 255), thickness=3)
    vp_pontos_line_img = cv2.line(vp_pontos_line_img, right_p_1, right_p_2, color=(255, 0, 255), thickness=3)

    #cv2.imshow('image',vp_pontos_line_img)
    

    #cv2.imshow('image',lane_lines_image)
    #cv2.waitKey(1)

#entra imagem segmentada, devolve apenas as faixas
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


def skel(img):
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

def detect_lines(image):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 90  # angular precision in radian, i.e. 1 degree
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
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(line_image, pt1, pt2, line_color, line_width, cv2.LINE_AA)
    #cv2.line(line_image,(x1,y1),(x2,y2),line_color,line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def filter_vertical_lines(lines):

    ok_lines = []
    sine_limit = 0.9
    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            if np.sin(theta) < sine_limit:
                ok_lines.append(line)

    return ok_lines

def get_avg_line_list(line_list):
    rho_sum = 0
    theta_sum = 0
    avg = []
    #print('linelist',line_list)
    if line_list is not None:
        for line in line_list:
            rho, theta = line[0]

            rho_sum += rho
            theta_sum += theta
        #print('list', list,' len(list)', len(list) )
        avg.append(np.array([[rho_sum/len(line_list), theta_sum/len(line_list)]], dtype=np.float32))
    return avg

def average_line(lines):

    left_lines = []
    right_lines = []
    right_avg = []
    left_avg = []
    for line in lines:
        rho, theta = line[0]

        if np.cos(theta) > 0:
            right_lines.append(line)
        else:
            left_lines.append(line)

    if right_lines != []:
        right_avg = get_avg_line_list(right_lines)

    if left_lines != []:
        left_avg = get_avg_line_list(left_lines)


    return  left_avg, right_avg


right_line_antes = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]
left_line_antes = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]

left_line_accum = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]
right_line_accum = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]


def accumulator(left_line, right_line):

    global right_line_antes, left_line_antes
    global right_line_accum, left_line_accum
    #print('recebido', left_line, right_line)


    # Caso faixa seja vazia
    if left_line != []:
        left_line_accum.append(left_line[0])

    if right_line != []:
        right_line_accum.append(right_line[0])


    #print("left line0",left_line, "type", type(left_line))

    #left_line_accum.append(left_line[0]) # adiciona a left line na lista
    #right_line_accum.append(right_line[0])


    #print("left_line_accum",left_line_accum, "type", type(left_line_accum))
    #left_line_accum.append(np.array([[rho_sum/len(left_lines), theta_sum/len(left_lines)]], dtype=np.float32))
    
    accum_max_size = 5
    
    # deleta primeiro termo se tiver mais q 5 linhas
    if len(left_line_accum) > accum_max_size:
        #print('antes do pop',left_line_accum)
        left_line_accum.pop(0)
        #print('depois do pop',left_line_accum)

    if len(right_line_accum) > 5:
        #print('antes do pop',left_line_accum)
        right_line_accum.pop(0)
        #print('depois do pop',left_line_accum)
    #tira a média dos valores na lista do acumulador

    #print('left_accum_avg',left_line_accum)
    #print(left_line_accum)
    left_accum_avg = get_avg_line_list(left_line_accum)

    right_accum_avg = get_avg_line_list(right_line_accum)
    
    #print('media', left_accum_avg)
    #print("left_line_accum",left_line_accum, "type", type(left_line_accum))
    #print('lista',left_line_accum,'len',len(left_line_accum))
    return left_accum_avg, right_accum_avg

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
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



def image_processing2(img_gray):
    roi_img = get_roi(img_gray)
    #cv2.imshow('image', roi_img)
    #cv2.waitKey(0)
    skel_img = skel(roi_img) # esqueletiza a imagem
    #skel_img = skel(img_gray)

    #cv2.imshow('image',skel_img)
    #cv2.waitKey(0)

    lines = detect_lines(skel_img) # todas as linhas detectadas 
    #print('lines:', lines)

    lines = filter_vertical_lines(lines) # discarta linhas com angulo muito horizontal

    #lines = [left_lines, right_lines]
    #print(left_lines, right_lines)
    #print('lines', lines, 'type', type(lines))
    #print('lines1', type(lines[1]))

    left_line, right_line  = average_line(lines) # pega média das linhas da esquerda e direita
 

    left_line, right_line = accumulator(left_line, right_line)

    #print('left', left_line, '\n right', right_line, 'type', type(left_line))

    #print(left_line, right_line)



    #intersecção das duas linhas
    intersec = intersection(left_line[0], right_line[0])
    #print('intersec_x', intersec[0][0])
    Erro = intersec[0][0] - 360
    #print('Erro = ', Erro)
    #single_lines = np.concatenate((left_line, right_line), axis=0)


    #print('left:', right_line)

    skel_img_bgr = cv2.cvtColor(skel_img,  cv2.COLOR_GRAY2BGR)

    skel_with_lines = display_lines(skel_img_bgr, lines, line_color = (0,0,255), line_width=1)

    #cv2.imshow('image', skel_with_lines)
    #cv2.waitKey(0)
    
    skel_with_lines = display_lines(skel_with_lines, left_line)
    skel_with_lines = display_lines(skel_with_lines, right_line)
    #skel_with_lines = display_lines(skel_with_lines, single_lines)

    cv2.imshow('image',skel_with_lines)
    return Erro



if __name__ == '__main__':

    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_color.png'
    path = 'D:\CARLA_0.9.12_win\TCC\static_road.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_left_only.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\line2.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\imglank.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\svanish.png'
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    #image_processing(img_gray)
    #cv2.waitKey(0)
       
    cv2.destroyAllWindows()
    for n in range(10):
        image_processing2(img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #img = get_roi(img_gray)
    #img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
    #cv2.imshow('image',img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()





