import cv2
import numpy as np


def output_optical(mp4_vid_fl):
    '''
    src: https://www.geeksforgeeks.org/python/python-opencv-dense-optical-flow/
    '''

    cap = cv2.VideoCapture(mp4_vid_fl)

    ret, first_frame = cap.read()

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)

    mask[..., 1] = 255

    counter = 1

    lst_of_optical_flow = []

    while(cap.isOpened()) and ret == True:
        
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mask[..., 0] = angle * 180 / np.pi / 2
        
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        
        prev_gray = gray
        
        if counter == 1:
            lst_of_optical_flow += [np.copy(rgb), np.copy(rgb)]

        else:
            lst_of_optical_flow.append(np.copy(rgb))

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return lst_of_optical_flow


def process_optical_flow(lst_of_optical_flow):

    blurred_img_lst = []

    for img in lst_of_optical_flow:

        blurred_img = cv2.medianBlur(img, 21)
        grey_img = np.median(blurred_img, axis = -1)
        blurred_img_lst.append(grey_img)

    return blurred_img_lst

def create_optical_flow_output(mp4_fl):
    
    optical_img_lst = output_optical(mp4_fl)

    out_blurred = process_optical_flow(optical_img_lst)

    return out_blurred

def main():
    #output_optical("/Users/johannesbauer/Documents/Coding/SaberPredict/Dataset/clips/1/1_Left.mp4")
    pass

if __name__ == "__main__":
    main()