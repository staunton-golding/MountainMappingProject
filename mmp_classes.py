import time
import numpy as np
import pandas as pd

from PIL import Image
from pillow_heif import register_heif_opener
from numba import njit

#for opening iphone .heic photos
register_heif_opener()

#get image metadata
class ImageData(object):
    exif_data = None
    image = None
    
    def __init__(self, filename):
        self.image = Image.open(filename)
        self.image.verify()
        self.get_exif()
        self.get_geotagging(self.exif)
        
    def get_exif(self):
        self.exif = self.image.getexif().get_ifd(0x8825)
        
    def get_geotagging(self, exif):
        geo_tagging_info = {}
        if not exif:
            raise ValueError("No EXIF metadata found")
        else:
            gps_keys = ['GPSLatitudeRef', 'GPSLatitude', 'GPSLongitudeRef', 'GPSLongitude']#,
            
            for ii, vv in self.exif.items():
                try:
                    geo_tagging_info[gps_keys[ii-1]] = np.array(vv)
                except IndexError:
                    pass
            
            self.geotag_info = geo_tagging_info
            self.lat, self.long, self.lat_ref, self.long_ref = self.get_lat_lng()

    def convert_to_degrees(self, value):
            #Helper to convert GPS stored in EXIF to degrees in float
            d0 = value[0]
            m0 = value[1]
            s0 = value[2]
            return d0 + (m0 / 60.0) + (s0 / 3600.0)
    
    def get_lat_lng(self):
        #Returns latitude and longitude from the provided exif_data (obtained through get_exif_data above)
        lat = None
        lng = None
        exif_data = self.geotag_info

        gps_lat = self.geotag_info['GPSLatitude']
        gps_lat_ref = self.geotag_info['GPSLatitudeRef']
        gps_long = self.geotag_info['GPSLongitude']
        gps_long_ref = self.geotag_info['GPSLongitudeRef']
        
        if gps_lat.size > 0 and gps_lat_ref and gps_long.size > 0 and gps_long_ref:
            lat = self.convert_to_degrees(gps_lat)
            if gps_lat_ref != "N":                     
                lat = 0 - lat
            lng = self.convert_to_degrees(gps_long)
            if gps_long_ref != "E":
                lng = 0 - lng
                
        lat = np.asarray(lat)
        long = np.asarray(lng)
        return lat, long, gps_lat_ref, gps_long_ref
#get peaks within bounding box of DEM data
class PeaksInImage(object):
    
    def __init__(self, image_data, geo_x, geo_y):
        self.gt = image_data.GetGeoTransform()
        self.geo_x = geo_x
        self.geo_y = geo_y
        self.band1 = image_data.ReadAsArray()
        self.convert_geo_xy()
   
    #convert geo coords of mountains to xy coords of image
    def convert_geo_xy(self):
        gt = self.gt
        geo_x = self.geo_x
        geo_y = self.geo_y
        band1 = self.band1

        #convert geo to pixel
        col = ((geo_x - gt[0]) / gt[1]) 
        row = ((geo_y - gt[3]) / gt[5])
        
        P1 = np.vstack((col, row)).T

        # Define columns
        column_index_x = 0
        column_index_y = 1
        
        # Define the range (inside downloaded data)
        lower_bound_y = 0
        upper_bound_y = band1.shape[0]
        
        lower_bound_x = 0
        upper_bound_x = band1.shape[1]
        
        # checks if mountain peak is in our image
        mask_x = ((P1[:, column_index_y] >= lower_bound_y) & (P1[:, column_index_y] <= upper_bound_y)
                & (P1[:, column_index_x] >= lower_bound_x) & (P1[:, column_index_x] <= upper_bound_x))
        
        # Get the row indices where the mask is True (where mountains are in image)
        row_indices = np.where(mask_x)[0]
        self.row_indices = row_indices
        
        #get points in area of interest
        aoi_P1 = P1[row_indices,:]
        aoi_P1 = np.round(aoi_P1).astype(int)
        self.aoi_P1 = aoi_P1      

#bounding box not always centered around photo coordinates. Get row and column of photo location in DEM
class MidpointIdx(object):
    
    def __init__(self, image_data, lat_orig, long_orig):
        self.gt = image_data.GetGeoTransform()
        self.long_orig = long_orig
        self.lat_orig = lat_orig
        self.band1 = image_data.ReadAsArray()
        self.get_mp()

    def get_mp(self):
        #get idx of where image was taken
        gt = self.gt
        band1 = self.band1
        
        center_long = gt[0] + gt[1] * np.arange(0,band1.shape[0])
        center_long_idx = np.argmin(np.abs(np.abs(center_long) - np.abs(self.long_orig)))
        center_lat = gt[3] + gt[5] * np.arange(0,band1.shape[1])
        center_lat_idx = np.argmin(np.abs(np.abs(center_lat) - np.abs(self.lat_orig)))

        self.midpoint = np.asarray([center_lat_idx, center_long_idx])


#all njit tagged functions are helper functions for below. Instantiated outside of class as njit doesn't play nice with big classes

#fill array containing panoramic image (start with values closest to photo location, work your way out, as to not include peaks / elevations obscured by bigger mountains
@njit
def _fill_im_array(imshow_arr, r_sorted, im_elev, theta_round, color, peak_check_mask, elev_add):
    
    for idx in range(len(r_sorted)):
        theta = theta_round[r_sorted[idx]]
        rad = r_sorted[idx]
        elev = im_elev[r_sorted[idx]]
        
        if imshow_arr[elev, theta] == 0:
            first_nonzero_row = np.nonzero(imshow_arr[:, theta] == 0)[0][0]
            imshow_arr[first_nonzero_row:elev, theta] = color[r_sorted[idx]]
            
        if imshow_arr[elev+elev_add, theta] != 0: #sometimes true peaks are slightly obscured, while the mountain is wholly visible. elev_add solves for this
            peak_check_mask[r_sorted[idx]] = 0
            
    return imshow_arr, peak_check_mask

#calculate distance between two coordinates
@njit
def _haversine(cols_corrected, delta_deg, rows_corrected, lat_orig):
    return ((np.sin(np.deg2rad((cols_corrected * delta_deg / 2))) * np.sin(np.deg2rad((cols_corrected * delta_deg / 2)))) + 
         (np.cos(np.deg2rad((cols_corrected * delta_deg + lat_orig))) * np.cos(np.deg2rad((lat_orig))) * np.sin(np.deg2rad((rows_corrected * delta_deg / 2))) * np.sin(np.deg2rad((rows_corrected * delta_deg / 2)))))

#part of distance calculation
@njit
def _arct(a):
    return(2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

#angle between all DEM pixels and photo location pixel
@njit
def _atan_theta(cols_corrected, rows_corrected):
    return np.arctan2(cols_corrected, rows_corrected)

#distance (pixels) from photo pixel, sorted by closest to furthest
@njit
def _sort_rad(rows_corrected, cols_corrected):
    rad = np.sqrt(rows_corrected**2 + cols_corrected**2)
    r_sorted = np.argsort(rad)
    return rad, r_sorted

#color for pano image point (changes according to distance) 
@njit 
def _color_fill(rad, max_r):
    return 1 - (rad / max_r)

class PolarImage(object):

    def __init__(self, band1, gt, midpoint, lat, aoi_P1, row_indices, elev_add, df_selected):
        
        self.midpoint = midpoint
        self.lat = lat
        self.band1 = band1
        self.gt = gt
        self.delta_deg = np.abs(self.gt[1]) #for use in pipeline that downloads images from dem_getter 1NED - pixels are always square (but this value is sometimes negative depending on orientation)
        self.aoi_P1 = aoi_P1
        self.row_indices = row_indices
        self.elev_add = elev_add

        #convert DEM to adjusted polar coordinates
        self.cart2pol_im(self.band1, self.midpoint, self.delta_deg, self.lat)
        imshow_arr = np.zeros((np.max(self.elev_corrected)+50,np.max(self.theta_round)))
        #Convert mountain peaks to adjusted polar coordinates
        self.unwrap_peak_labels(self.aoi_P1, self.row_indices, self.midpoint, self.band1, self.elev_corrected, df_selected)
        #fill panoramic image (DEM data and peaks)
        start = time.time()
        self.pano_image_arr, self.peak_check_mask = self.fill_im_array(imshow_arr, self.r_sorted, self.elev_corrected, self.theta_round, self.color, self.peak_check_mask, self.elev_add)
        end = time.time()
        print(f"Done with filling pano image aray in {int(np.round(end - start))} seconds")
        #correct for zero column artifacts
        self.artifact_correction_pano_image()
        self.artifact_correction_peaks()
    
    def haversine(self, cols_corrected, delta_deg, rows_corrected, lat_orig):
        return _haversine(cols_corrected, delta_deg, rows_corrected, lat_orig)       
        
    def fill_im_array(self, imshow_arr, r_sorted, im_elev, theta_round, color, peak_check_mask, elev_add):
        return _fill_im_array(imshow_arr, r_sorted, im_elev, theta_round, color, peak_check_mask, elev_add)      
   
    def arct(self, a):
        return _arct(a)
    
    def atan_theta(self, cols_corrected, rows_corrected):
        return _atan_theta(cols_corrected, rows_corrected)
    
    def sort_rad(self, rows_corrected, cols_corrected):
        return _sort_rad(rows_corrected, cols_corrected)
    
    def color_fill(self, rad, max_r):
        return _color_fill(rad, max_r)
    
    def cart2pol_im(self, elev_data, midpoint, delta_deg, lat_orig):
        start = time.time()
        #cols and rows as indices
        cols = np.arange(0, elev_data.shape[0])
        cols = np.hstack((cols,)*elev_data.shape[1])
        
        rows = np.arange(0, elev_data.shape[1]).reshape(-1,1)
        rows = np.tile(rows, elev_data.shape[0]).flatten()
        
        #don't need pixel / idx from where image was taken (radius will be 0, theta will be confused)
        cols_no_mp = np.delete(cols, midpoint[0])
        rows_no_mp = np.delete(rows, midpoint[1])
    
        #cols and rows indices centered around image location
        cols_corrected = cols_no_mp - midpoint[0]
        rows_corrected = rows_no_mp - midpoint[1]
        
        #haversine formula, from https://www.movable-type.co.uk/scripts/latlong.html
        a = self.haversine(cols_corrected, delta_deg, rows_corrected, lat_orig)
        c = self.arct(a)
        R = 6371*1000
        d = c*R
        delt_elev = d**2 / (2*R)
        
        im_arr = elev_data.flatten()
        im_arr_no_mp = np.delete(im_arr, np.ravel_multi_index([midpoint[0], midpoint[1]], elev_data.shape))
    
        #convert x and y into modified polar
        theta = self.atan_theta(cols_corrected, rows_corrected)
        theta_round = np.round((theta+np.pi)*4, decimals = 2)*100 #reduce num theta for more efficient image processing, increasing decimals increases horizontal resolution
        rad, r_sorted = self.sort_rad(rows_corrected, cols_corrected)
        theta_round = theta_round.astype(int)
        elev_corrected = (im_arr_no_mp - delt_elev).astype(int)
        
        max_r  = np.max(rad)
        color = self.color_fill(rad, max_r)
        color[elev_corrected<0] = 0 #to allow for elimination of all zero columns in fill_im_array step
        elev_corrected[elev_corrected<0] = 0
        
        self.theta_round = theta_round
        self.r_sorted = r_sorted
        self.rad = rad
        self.elev_corrected = elev_corrected
        self.color = color
        end = time.time()
        print(f"Done with cart2pol in {int(np.round(end - start))} seconds")
        
    def unwrap_peak_labels(self, aoi_P1, row_indices, midpoint, band1, elevs, df_selected):
        #names of all peaks in bounding box
        start = time.time()
        names_used = df_selected['Feature Name']
        names_used = names_used.iloc[row_indices].reset_index()
        names_used = names_used.iloc[:,1]
        
        xs_p1 = aoi_P1[:,1] #xs
        ys_p1 = aoi_P1[:,0] #ys

        #idx of peaks in flattened array (masked in this way due to multiple naming conventions - some peaks are within the same 30m x 30m box, so can't just make mask overlayed on DEM, need specific theta and elevation of each peak (encoded in flattened array idx).
        idx_peaks = np.ravel_multi_index([xs_p1, ys_p1], band1.shape)
        elevs_aoi = elevs[idx_peaks]
        
        peak_check_mask = np.zeros((len(band1.flatten()),1))
        peak_check_mask[idx_peaks] = 1
        peak_check_mask = peak_check_mask.T.squeeze()
        
        xs_p1_theta = xs_p1 - midpoint[1]
        ys_p1_theta = ys_p1 - midpoint[0]
        
        theta_peak = np.arctan2(ys_p1_theta, xs_p1_theta) ### Y FIRST, CHECK DOCS AGAIN TM
        theta_round_peak = (np.round((theta_peak + np.pi)*4, decimals = 2)*100).astype(int)
        
        name_peak_df = pd.DataFrame({'Elevations': elevs_aoi, 'Theta': theta_round_peak, 'idx peaks': idx_peaks, 'Peak Name': names_used}).reset_index()

        self.name_peak_df = name_peak_df
        self.peak_check_mask = peak_check_mask
        end = time.time()
        print(f"Done with unwrap_peak_labels in {int(np.round(end - start))} seconds")

    def artifact_correction_pano_image(self):
        #theta equisampled from min to max theta with 1,000 steps per rad. Not all theta have values. Need to eliminate these zero-columns
        self.art_cor_idx = np.where(np.all(self.pano_image_arr == 0, axis=0))[0]
        self.art_cor_pano_image = self.pano_image_arr[:,~np.all(self.pano_image_arr == 0, axis=0)]
        del self.pano_image_arr

    def artifact_correction_peaks(self):
        start = time.time()
        #after pano image artifact corrected, need to shift peaks in image to left to account for deleted zero-columns
        
        #idx of all peaks still visible
        idx_of_peaks = np.nonzero(self.peak_check_mask)[0]

        #filter df to have only visible peaks
        all_idx_in_img = self.name_peak_df['idx peaks'].isin(idx_of_peaks)
        mountains_in_image = self.name_peak_df[all_idx_in_img == True]
        
        mt_index_adjust = mountains_in_image.index
        theta_peaks_adjust = mountains_in_image['Theta'].to_numpy()
        elev_peaks_adjust = mountains_in_image['Elevations'].to_numpy()
        names_peaks = mountains_in_image['Peak Name'].reset_index().iloc[:,1]
        
        #sort theta (theta guaranteed << 10,000, big O not a concern)
        theta_sort_idx = np.argsort(theta_peaks_adjust)
        
        #unadjusted, need to hold original values (e.g. theta = 225 with idx to be deleted = 224, 223, 222, 119, 117. Without, this theta would move to 222, but 223 and 224 wouldn't be counted)
        theta_sorted_temp_arr = theta_peaks_adjust[theta_sort_idx]

        #sort all needed variables
        names_sorted = names_peaks[theta_sort_idx]
        mt_index_adjust = mt_index_adjust[theta_sort_idx]
        theta_peaks_adjust_sorted = theta_peaks_adjust[theta_sort_idx]
        elev_peaks_adjust_sorted = elev_peaks_adjust[theta_sort_idx]
        
        #find first theta (from orig sorted array) bigger than deleted column, shift all thetas bigger left by one.
        for jj in range(len(self.art_cor_idx)):
            first_theta_greater = np.nonzero(theta_sorted_temp_arr[:] >= self.art_cor_idx[jj])[0][0]
            theta_peaks_adjust_sorted[first_theta_greater:] -= 1
        end = time.time()
        print(f"Done with artifact correction in {int(np.round(end - start))} seconds")
        self.name_peak_df_art_cor = pd.DataFrame({'Elevations': elev_peaks_adjust_sorted, 'Theta': theta_peaks_adjust_sorted, 'Peak Name': names_sorted, 'Mountain Index': mt_index_adjust})
        del self.name_peak_df, self.peak_check_mask
        