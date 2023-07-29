import cv2 
import numpy as np

from skimage import io
import matplotlib.pyplot as plt





#funkcija za srednju osvetljenost  
def brightness(img):
    mean_brightness=cv2.mean(img)[0]
    return mean_brightness

def estimate_background_illumination(img):
    kernel=np.ones((5,5), np.uint8)
    opening=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    blur=cv2.medianBlur(closing, 21)
    diff=255-cv2.absdiff(img, blur)
    norm_diff=cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_diff

def image_sharpening(img):
    kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    new_img=cv2.filter2D(img, -1, kernel)
    return new_img

def image_binarization(img, tresh):
    for i, row in enumerate(img):
        for j, pixel in enumerate(img):
            if(j-tresh<tresh/2):
                img[i][j]=0
            else:
                img[i][j]=255
    return img







img=cv2.imread("C:\slike\Tekst_1.png")
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tresh=brightness(img_gray)

end_img=image_binarization(image_sharpening(estimate_background_illumination(img_gray)), tresh)

plt.imshow(end_img, cmap="gray")
plt.show()



