import cv2
#%matplotlib auto


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def remove_noise(image):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_binary = cv2.threshold(bg, 0, 255, cv2.THRESH_OTSU)[1]

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
    erode = cv2.erode(out_binary, se1, iterations=1)
    final = cv2.morphologyEx(erode, cv2.MORPH_OPEN, se1)

    median_blur = cv2.medianBlur(final, 3)
    return median_blur


def crop_image(median_blur):
    img = median_blur
    upper_boundry = -1
    lower_boundry = img.shape[0]

    for y in img[0:]:
        upper_boundry += 1
        if 0 in y:
            break

    for y1 in reversed(img[0:]):
        lower_boundry += -1
        if 0 in y1:
            break
    
    roi = img[upper_boundry:lower_boundry, :]
    return roi


def resize_image(roi):
    img = cv2.resize(roi, (512, 256), interpolation=cv2.INTER_AREA)
    return img


def process_image(path):
    image = read_image(path)
    cleaned_image = remove_noise(image)
    cropped_image = crop_image(cleaned_image)
    resized_image = resize_image(cropped_image)
    
    return resized_image