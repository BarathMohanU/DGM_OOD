import numpy as np
import cv2
from numpy.linalg import norm as np_norm
from scipy.special import softmax
from scipy.special import factorial

def rotate(img, angle):
    rotated_img = np.zeros(img.shape)
    if angle == 90:
        rot_func = cv2.cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        rot_func = cv2.cv2.ROTATE_180
    elif angle == 270:
        rot_func = cv2.ROTATE_90_COUNTERCLOCKWISE
    for i in range (img.shape[0]):
        temp = cv2.rotate(img[i,:,:,:], rot_func)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        rotated_img[i,:,:,:] = temp
    return np.round(rotated_img)

def rotate_inv(img, angle):
    rotated_img = np.zeros(img.shape)
    if angle == 90:
        rot_func = cv2.cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        rot_func = cv2.cv2.ROTATE_180
    elif angle == 270:
        rot_func = cv2.ROTATE_90_COUNTERCLOCKWISE
    for i in range (img.shape[0]):
        temp = cv2.rotate(cv2.flip(img[i,:,:,:],1), rot_func)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        rotated_img[i,:,:,:] = temp
    return np.round(rotated_img)

def inv(img):
    inverted_img = np.zeros(img.shape)
    for i in range (img.shape[0]):
        temp = cv2.flip(img[i,:,:,:],1)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=-1)
        inverted_img[i,:,:,:] = temp
    return np.round(inverted_img)

def transform(imgs, angle):
    if angle == 0:
        imgs = rotate(imgs, 90)
    elif angle == 1:
        imgs = rotate(imgs, 180)
    elif angle == 2:
        imgs = rotate(imgs, 270)
    elif angle == 3:
        imgs = inv(imgs)
    elif angle == 4:
        imgs = rotate_inv(imgs, 90)
    elif angle == 5:
        imgs = rotate_inv(imgs, 180)
    elif angle == 6:
        imgs = rotate_inv(imgs, 270)
    return imgs

def dissimilarity(a,b):
    cos = np.zeros(a.shape[0])
    for i in range (a.shape[0]):
        cos[i] = (1 - (np.dot(a[i,:,:,:].flatten(), b[i,:,:,:].flatten())/(np_norm(a[i,:,:,:].flatten())*np_norm(b[i,:,:,:].flatten()))))/2
    return cos

def remove_perf_recon(model, batch):
    mod = model(batch, training=False)
    logits = mod.mixture_distribution.logits.numpy()
    logits = softmax(logits, axis=-1)
    locs = mod.components_distribution.distribution.distribution.mean()
    recon = np.round(np.sum(np.expand_dims(logits,axis=-1)*locs,axis=3) + 0.5)
    recon_error = np.mean(np.abs(batch - recon),axis=-1)
    recon_error[recon_error<=0.5] = 0
    recon_error[recon_error>0.5] = 1
    log_probs = mod.log_prob(batch).numpy()
    log_probs = log_probs*recon_error
    return np.sum(log_probs, axis=(1,2))
    
def derangement(n):
    if n <= 2:
        orders = np.array(list(itertools.permutations(np.arange(n*n))))
        tmp = []
        for i in range (orders.shape[0]):
            count = 0
            for j in range (orders.shape[1]):
                if orders[i,j] == j:
                    count += 1
            if count == 0:
                tmp.append(orders[i,:])
        return np.array(tmp), np.array(tmp).shape[0]
    else:
        shuffles = 20
        orders = np.zeros((shuffles, n*n))
        tmp = np.arange(n*n)
        for i in range (shuffles):
            p = np.random.permutation(n*n)
            orders[i,:] = tmp[p]
        return orders.astype('int32'), orders.shape[0]

def patch_shuffle(imgs, n, orders):
    shap = imgs.shape
    patches = np.zeros((shap[0], int(shap[1]/n), int(shap[2]/n), shap[3], int(n*n)))
    patch_shap = patches.shape
    shuffled = np.zeros((shap[0], shap[1], shap[2], shap[3], orders.shape[0]))
    for i in range (n):
        for j in range (n):
            patches[:,:,:,:,i*n+j] = imgs[:, i*(patch_shap[1]):(i+1)*(patch_shap[1]), j*(patch_shap[2]):(j+1)*(patch_shap[2]), :]
    for k in range (orders.shape[0]):
        for i in range (n):
            for j in range (n):
                shuffled[:, i*(patch_shap[1]):(i+1)*(patch_shap[1]), j*(patch_shap[2]):(j+1)*(patch_shap[2])
                         , :, k] = patches[:,:,:,:,np.squeeze(orders[k,i*n+j])]
    return shuffled