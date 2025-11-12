# Библиотека для генерации искуственного шума

import numpy as np

def add_salt_pepper_noise(img, salt_prob: float, pepper_prob: float):
    if img is None:
        raise ValueError('img is None')
    if salt_prob < 0 or salt_prob > 1:
        raise ValueError('salt_prob is out of range')
    if pepper_prob < 0 or pepper_prob > 1:
        raise ValueError('pepper_prob is out of range')

    noise = np.copy(img) # Copy source image
    salt = np.random.random(img.shape[:2]) < salt_prob

    # Adding salt
    if len(noise.shape) == 3:
        noise[salt] = [255, 255, 255]
    else:
        noise[salt] = 255

    # Adding pepper
    pepper = np.random.random(img.shape[:2]) < pepper_prob
    if len(noise.shape) == 3:
        noise[pepper] = [0, 0, 0]
    else:
        noise[pepper] = 0

    return noise

def add_Gauss_noise(img, sigma: float, a: float):
    if img is None:
        raise ValueError('img is None')
    if sigma <= 0:
        raise ValueError('sigma less than zero')

    mask = np.random.normal(a, sigma, img.shape)
    noise = img.astype(np.float64) + mask
    return np.clip(noise, 0, 255).astype(np.uint8)

def add_impulse_noise(img, noise_prob: float):
    if img is None:
        raise ValueError('img is None')
    if noise_prob < 0 or noise_prob > 1:
        raise ValueError('noise_prob is out of range')

    noise = np.copy(img)
    area = np.random.random(noise.shape[:2]) < noise_prob
    amount = np.sum(area)
    if len(img.shape) == 3:
        rnd_vals = np.random.randint(0, 256, (amount, 3))
    else:
        rnd_vals = np.random.randint(0, 256, amount)

    noise[area] = rnd_vals
    return noise

def add_uniform_noise(img, beg: float, end: float):
    mask = np.random.uniform(beg, end, img.shape)
    noise = img.astype(np.float64) + mask
    return np.clip(noise, 0, 256).astype(np.uint8)