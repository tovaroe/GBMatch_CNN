import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1' #DLL bug in my environment - might be unnecessary for most other people

import openslide
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, maximum_filter
import numpy as np
from skimage.transform import resize

from wsi_utils import *


class WSIHandler():
    def __init__(self, WSI_path):
        self.WSI_path = WSI_path
        self.WSI = openslide.OpenSlide(WSI_path)
        self.vendor = self.WSI.properties[openslide.PROPERTY_NAME_VENDOR]
        assert self.vendor in ['hamamatsu', 'aperio'], f"Unsupported file format. Supported vendors are hamamatsu and aperio, got {self.vendor}"
        self.tissue_mask = None
        self.tissue_mask_tile_generator = None
        
    def read_ndpa(self, magnification=0.15625):
              
        # define function to convert ndpa-values to pixel coordinates
        def nm2pix(nm, OffsetFromSlideCenter, mpp, objectivePower, magnification, image_size):
            nm_coord = nm - OffsetFromSlideCenter
            pixel_coord = nm_coord/(mpp*1000)
            pixel_coord = pixel_coord/(objectivePower/magnification)
            pixel_coord = pixel_coord + np.round(image_size/2)
            pixel_coord = int(np.round(pixel_coord))
            return pixel_coord
        
        # get wsi attributes
        mppX = float(self.WSI.properties['openslide.mpp-x'])
        mppY = float(self.WSI.properties['openslide.mpp-y'])
        objectivePower = int(self.WSI.properties['openslide.objective-power'])
        XOffsetFromSlideCentre = float(self.WSI.properties['hamamatsu.XOffsetFromSlideCentre'])
        YOffsetFromSlideCentre = float(self.WSI.properties['hamamatsu.YOffsetFromSlideCentre'])
        width = int(int(self.WSI.properties['openslide.level[0].width'])*magnification/objectivePower)
        height = int(int(self.WSI.properties['openslide.level[0].height'])*magnification/objectivePower)
        nm2pixX = lambda x: nm2pix(x, XOffsetFromSlideCentre, mppX, objectivePower, magnification, width)
        nm2pixY = lambda y: nm2pix(y, YOffsetFromSlideCentre, mppY, objectivePower, magnification, height)
        ndpa_path = self.WSI_path + '.ndpa'
        
        # assertions
        assert (width, height) in self.WSI.level_dimensions, "Width and/or height do not match level_dimensions."
        assert os.path.exists(ndpa_path), f"NDPA file not found. Given path: {ndpa_path}"
        
        # build & walk the xml annotation tree
        xml_tree = ET.parse(ndpa_path)
        annotations = xml_tree.getroot().findall("./ndpviewstate/annotation")
        polymask = Image.new('L', (width, height), 0)
        colormap = {"#ff0000": 1,
                    "#ffff00": 2,
                    "#00ff00": 3,
                    "#0000ff": 4,
                    "#000000": 5,
                    "#ff00ff": 6,
                    "#00ffff": 7,
                    "#ffffff": 8}


        for annotation in annotations:
            if annotation.attrib['displayname'] == 'AnnotateFreehand':
                color = colormap[annotation.attrib['color']]
                
                points = []
                pointlist = annotation.findall("./pointlist/point")
                if not len(pointlist)>1: continue
                for point in pointlist:
                    x = nm2pixX(int(point.findall("./x")[0].text))
                    y = nm2pixY(int(point.findall("./y")[0].text))
                    points.append((x,y))
                
                ImageDraw.Draw(polymask).polygon(points, outline=color, fill=color)

        mask = np.array(polymask)
        return mask
        
    def read_svs_annotation(self, magnification=0.15625):
        # Reads all annoations layers with name == "", "Layer 1" or "Tumor"
        annotation_path = self.WSI_path.replace('.svs', '.xml')
        assert os.path.exists(annotation_path), f"annotation file not found. Given path: {annotation_path}"
        xml_tree = ET.parse(annotation_path)
        
        objectivePower = int(self.WSI.properties['openslide.objective-power'])
        width = int(int(self.WSI.properties['openslide.level[0].width'])*magnification/objectivePower)
        height = int(int(self.WSI.properties['openslide.level[0].height'])*magnification/objectivePower)
        
        pix_convert = lambda x: int(np.round(x*(magnification/objectivePower)))
        
        polymask = Image.new('L', (width, height), 0)
        
        for annotation in xml_tree.getroot().findall("./Annotation"):
            if annotation.attrib['Name'] in ['', 'Layer 1', 'Tumor']:
                regions = annotation.findall("./Regions/Region")
                for region in regions:
                    if region.attrib['Type'] == '0': # take only freehand annotation            
                        pointlist = region.findall("./Vertices/Vertex")
                        if not len(pointlist) > 1: continue
                        points = []
                        
                        for point in pointlist:
                            x = pix_convert(int(point.attrib['X']))
                            y = pix_convert(int(point.attrib['Y']))
                            points.append((x,y))
                        
                        ImageDraw.Draw(polymask).polygon(points, outline=5, fill=5)# 5 corresponds to tumor here... maybe adapt this to a more sensible value
                        
        mask = np.array(polymask)
        return mask    

    def read_annotation(self, magnification=0.15625):
        if self.vendor == 'hamamatsu':
            return self.read_ndpa(magnification)
        elif self.vendor == 'aperio':
            return self.read_svs_annotation()
        
    def segment_tissue(self, min_area = 100):
        thumbnail = self.get_thumbnail()
        
        cd = colour_deconvolution(thumbnail, 'HE')+1e-9
        
        mask = (cd[:,:,0]/cd[:,:,2] < 0.85) + (cd[:,:,1]/cd[:,:,2] < 0.95)
        
        return mask
        
    def obtain_tissue_mask(self, min_area = 20, annotation_handling='exclude'):
        assert annotation_handling in ['exclude', 'include', 'ignore'], f"annoatation_handling must be in ['exclude', 'include', 'ignore'], got {annotation_handling}"
        
        if annotation_handling in ['exclude', 'ignore']:
            tissue_mask = self.segment_tissue(min_area)
        
        if annotation_handling in ['include', 'exclude']:
            annotation_mask = self.read_annotation()       
            if annotation_handling == 'include':         
                tissue_mask = annotation_mask == 5 # black because of reasons?
            if annotation_handling == 'exclude':
                tissue_mask[annotation_mask > 0] = False
            
        if annotation_handling in ['exclude', 'ignore']:
            tissue_mask = binary_erosion(tissue_mask)
            tissue_mask = bw_filter_area(tissue_mask, min_area)
        
        self.tissue_mask = tissue_mask
        return self.tissue_mask
        
    def load_tissue_mask(self, path=None):
        if path==None:
            path_trunk = '/'.join(self.WSI_path.split('/')[:-1])
            filename = self.WSI_path.split('/')[-1].replace('.ndpi', '.npy')
            path = '/'.join([path_trunk, 'tissue_masks', 'numpy', filename])
        self.tissue_mask = np.load(path)
        return self.tissue_mask
      
    def get_random_tile(self, magnification=20, width=256, height=256):
        ### !!! possibly buggy for some svs files
        assert not self.tissue_mask is None, "Please obtain tissue mask first"
        
        size = (width, height)
        rng = np.random.default_rng()
        downsample = float(self.WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])/magnification
        tm_factor = int(np.round(float(self.WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])/0.15625))
        level = self.WSI.get_best_level_for_downsample(downsample)
        
        coordinates = np.argwhere(self.tissue_mask)
        coords = np.flip(rng.choice(coordinates)*tm_factor) + np.random.choice(int(magnification/0.15625), 2)
                 
        return self.WSI.read_region(coords, level, size).convert('RGB')
        
    def get_thumbnail(self):
        magnification = 0.15625
        # magnification 0.15625 corresponds to downsample = 256 with 40x objective power
        downsample = int(self.WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])/magnification
        
        if self.vendor == 'aperio':
            # workaround for few downsample levels in svs files
            thumbnail_size = np.array(self.WSI.dimensions)/downsample
        elif self.vendor == 'hamamatsu':
            thumbnail_size = self.WSI.level_dimensions[self.WSI.get_best_level_for_downsample(downsample)]
        else:
            raise ValueError("Unsupported file format")
            
        return self.WSI.get_thumbnail(thumbnail_size)
        
    def tile_generator(self, magnification=20, width=256, height=256):
        # yields consecutive tiles and corresponding coords in downsampled tissue_mask
        
        size = (width, height)
        px_conversion_factor = np.int64(magnification / 0.15625) # thumbnail magnification
        downsample_factor = np.int64(np.array(size)/px_conversion_factor)
        downsample = float(self.WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])/magnification
        level = self.WSI.get_best_level_for_downsample(downsample)
        
        # build correct tissue mask
        tissue_mask_downsampled = resize(maximum_filter(self.tissue_mask, size=downsample_factor).astype(np.float32), np.round(np.array(self.tissue_mask.shape)/downsample_factor)) > 0.5
        self.tissue_mask_tile_generator = tissue_mask_downsampled.copy()
        coordinates = np.argwhere(tissue_mask_downsampled) 
        
        # iterate through coordinates & yield tiles
        tedious_factor = int(downsample/(np.round(self.WSI.level_downsamples)[level]))

        for coords_orig in coordinates:
            coords = np.flip(coords_orig*px_conversion_factor*downsample_factor*int(downsample)) # downsample was tedious_factor
            yield (self.WSI.read_region(coords, level, np.array(size)*tedious_factor).convert('RGB').resize(size), coords_orig)

    