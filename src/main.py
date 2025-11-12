import noise_gen as ng
import cv2
import os

def main() -> None:
    
    path: str = '..\Digital_Image_Processing\data\dog.jpg'
    img = cv2.imread(path)

    if img is None:
        return

    cv2.namedWindow('Source Image')
    cv2.imshow('Source Image', img)

    cv2.waitKey(0)

    noise = ng.add_Gauss_noise(img, 25, 0)

    cv2.namedWindow('Noise Image')
    cv2.imshow('Noise Image', noise)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Hello, world!')


if __name__ == '__main__':
    main()