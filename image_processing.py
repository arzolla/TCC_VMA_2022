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



def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


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


def hough_transform(image):
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


def filter_vertical_lines(lines, sine_limit=0.9):

    ok_lines = []
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
    if line_list != []:
        for line in line_list:
            rho, theta = line[0]

            rho_sum += rho
            theta_sum += theta
        #print('list', list,' len(list)', len(list) )
        avg.append(np.array([[rho_sum/len(line_list), theta_sum/len(line_list)]], dtype=np.float32))
    return avg

def average_lines(lines):

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


def separe_left_right(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        rho, theta = line[0]

        if np.cos(theta) > 0:
            right_lines.append(line)
        else:
            left_lines.append(line)

    return left_lines, right_lines


def average_lines(lines):

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

    print('pros2',left_lines, right_lines)

    return  left_avg, right_avg



left_line_accum = [np.array([[-81.       ,   2.5132742]], dtype=np.float32)]
right_line_accum = [np.array([[502.        ,   0.62831855]], dtype=np.float32)]


def accumulator(left_line, right_line):

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



def image_processing3(img_gray):
    roi_img = get_roi(img_gray)

    skel_img = skel(roi_img) # esqueletiza a imagem



    lines = hough_transform(skel_img) # todas as linhas detectadas 


    lines = filter_vertical_lines(lines) # discarta linhas com angulo muito horizontal

    left_lines, right_lines = separe_left_right(lines)

    print('pros3',left_lines, right_lines)

    #left_line, right_line  = average_lines(lines) # pega média das linhas da esquerda e direita
 
    left_line = get_avg_line_list(left_lines)
    right_line = get_avg_line_list(right_lines)

    print('pros3 avg',left_line, right_line)

    #left_line, right_line = accumulator(left_line, right_line)

    #intersecção das duas linhas
    intersec = intersection(left_line[0], right_line[0])

    print('3 int',intersec)

    Erro = intersec[0][0] - 360



    skel_img_bgr = cv2.cvtColor(skel_img,  cv2.COLOR_GRAY2BGR)
    skel_with_lines = display_lines(skel_img_bgr, lines, line_color = (0,0,255), line_width=1)



    skel_with_lines = display_lines(skel_with_lines, left_line)
    skel_with_lines = display_lines(skel_with_lines, right_line)

    cv2.imshow('processing3',skel_with_lines)
    return Erro

def image_processing2(img_gray):
    roi_img = get_roi(img_gray)

    skel_img = skel(roi_img) # esqueletiza a imagem



    lines = hough_transform(skel_img) # todas as linhas detectadas 


    lines = filter_vertical_lines(lines) # discarta linhas com angulo muito horizontal


    left_line, right_line  = average_lines(lines) # pega média das linhas da esquerda e direita
 
    print('pros2 avg',left_line, right_line)

    #left_line, right_line = accumulator(left_line, right_line)

    #intersecção das duas linhas
    intersec = intersection(left_line[0], right_line[0])

    print('2 int',intersec)

    Erro = intersec[0][0] - 360


    skel_img_bgr = cv2.cvtColor(skel_img,  cv2.COLOR_GRAY2BGR)
    skel_with_lines = display_lines(skel_img_bgr, lines, line_color = (0,0,255), line_width=1)



    skel_with_lines = display_lines(skel_with_lines, left_line)
    skel_with_lines = display_lines(skel_with_lines, right_line)

    cv2.imshow('processing2',skel_with_lines)
    return Erro

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

lines_antes = []

def image_processing_kmeans(img_gray):
    roi_img = get_roi_half(img_gray)


    skel_img = skel(roi_img) # esqueletiza a imagem


    #skel_img = skel(img_gray) # sem a roi

    lines = hough_transform(skel_img) # todas as linhas detectadas 
    #print('lines:', lines)

    #lines = filter_vertical_lines(lines, 0.99)
    #print(type(lines))

    global lines_antes

    if len(lines) < 4:
        lines = lines_antes
    else:
        lines_antes = lines

    segmented = segment_by_angle_kmeans(lines, k=4)
    #print("\n primeiro \n:", segmented[0])
    #print("\n segundo \n:", segmented[1])

    

    skel_img_bgr = cv2.cvtColor(skel_img,  cv2.COLOR_GRAY2BGR)



    skel_with_lines = skel_img_bgr
    colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255)]

    for n, line_group in enumerate(segmented):


        skel_with_lines = display_lines(skel_with_lines, line_group, colors[n], line_width=1)


    #cv2.imshow('image', skel_with_lines)
    #cv2.waitKey(0)
    
    #skel_with_lines = display_lines(skel_with_lines, left_line)
    #skel_with_lines = display_lines(skel_with_lines, right_line)
    #skel_with_lines = display_lines(skel_with_lines, single_lines)

    cv2.imshow('image',skel_with_lines)
    #return Erro

def show_image_rgb(rgb):
    cv2.imshow('image', rgb)

if __name__ == '__main__':

    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_color.png'
    path = 'static_road.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\static_road_left_only.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\line2.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\imglank.png'
    #path = 'D:\CARLA_0.9.12_win\TCC\svanish.png'
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    #image_processing(img_gray)
    #cv2.waitKey(0)
       
    for n in range(1):
        image_processing2(img_gray)
        image_processing3(img_gray)


        cv2.waitKey(0)

        cv2.destroyAllWindows()





    #img = get_roi(img_gray)
    #img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
    #cv2.imshow('image',img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()





