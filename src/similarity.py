import cv2
from skimage.metrics import structural_similarity as ssim

def calc_ssim(img1, img2):
    """SSIM 유사도 계산"""
    score, _ = ssim(img1, img2, full=True)
    return score

def calc_orb_similarity(img1, img2):
    """ORB 특징점 유사도 계산"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0

    good = [m for m in matches if m.distance < 60]
    return len(good) / len(matches)

def combined_similarity(img1, img2):
    """SSIM + ORB 가중 평균"""
    ssim_score = calc_ssim(img1, img2)
    orb_score = calc_orb_similarity(img1, img2)
    return 0.6 * ssim_score + 0.4 * orb_score
