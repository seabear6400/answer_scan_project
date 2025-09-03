import cv2

def preprocess(img, width=800, height=1100):
    """크기 정규화 + 노이즈 제거"""
    img = cv2.resize(img, (width, height))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img
