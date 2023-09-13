import numpy as np
from numpy.linalg import norm
from skimage.measure import label, regionprops

def colour_deconvolution(image, stains):

    # same as my MATLAB function, but not as efficient (ca. factor 10)
    
    if stains == 'HE':
        # Color deconvolution of H&E stains
        # RGB OD matrix for hematoxylin, eosin and background
        v1 = np.array([0.644318599239284, 0.716675682651343, 0.266888569576439])
        v2 = np.array([0.092831273979569, 0.954545685888634, 0.283239983269889])
        v3 = np.array([0.635954465292149, 0.000000000000000, 0.771726582459732])

    elif stains == 'KB':
        v1 = np.array([0.953, 0.303, -0.003])
        v2 = np.array([0.004, 0.999, -0.006])
        v3 = np.array([0.290, 0.001, 0.957])

    elif stains == 'LFB NFR':
        v1 = np.array([0.74,0.606,0.267])
        v2 = np.array([0.214,0.851,0.478])
        v3 = np.array([0.26800000,0.57000000,0.7760000])

    elif stains == 'H DAB':
        v1 = np.array([0.6500286, 0.704031, 0.2860126])
        v2 = np.array([0.26814753, 0.57031375, 0.77642715])
        v3 = np.array([0.7110272, 0.42318153, 0.5615672])

    elif stains == 'H DAB 2':
        v1 = np.array([0.7219949, 0.61885273, 0.30942637])
        v2 = np.array([0.38015208, 0.58023214, 0.72028816])
        v3 = np.array([0.57810706, 0.5294827, 0.62083834])

    elif stains == 'H DAB Ventana':
        v1 = np.array([0.7219949, 0.61885273, 0.30942637])
        v2 = np.array([0.48616475, 0.6279628, 0.60770595])
        v3 = np.array([0.49230802, 0.47189406, 0.7314019])
        
    elif stains == 'HE Pigment':
        v1 = np.array([0.56907135, 0.7671424, 0.29605788])
        v2 = np.array([0.29937613, 0.8949957, 0.33069116])
        v3 = np.array([0.41451415, 0.63914895, 0.6478168])
        
    else:
        raise ValueError(f"Colour deconvolution not available for stain: {stains}")
        
    ODmatrix = np.array([v1/norm(v1), v2/norm(v2), v3/norm(v3)])
    
    image = np.array(image)
    im_shape = image.shape
    image = image/255.
    image = -np.log(image+1e-9)
    image = image.reshape(-1,3)  
    image = np.linalg.solve(ODmatrix.T, image.T).T
    image = image.reshape(im_shape)
    image = np.exp(-image)
    image = image*255
    image = np.round(image)
    image[image<0] = 0
    image[image>255] = 255
    image = np.uint8(np.round(image))
    
    return image

def binary_disk(radius):
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2+y**2 <= radius**2
    return disk
    
def bw_filter_area(bw, min_area):
    label_bw = label(bw)
    coords = []
    bw_filtered = np.zeros_like(bw)
    
    regions = regionprops(label_bw)
    for i, r in enumerate(regions):
        if r.area > min_area:
            coords.append(r.coords)
    
    if coords:
        coords = np.concatenate(coords)
        for coord in coords:
            x, y = coord
            bw_filtered[x,y] = True
    
    return bw_filtered