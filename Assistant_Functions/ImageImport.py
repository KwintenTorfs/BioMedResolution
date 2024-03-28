import os
import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.errors import InvalidDicomError
from skimage.morphology import binary_erosion
from scipy import ndimage

# According to Ahmad (2021) -> take soft tissue upper to be 170, not 100 HU
# Upper bound of lung tissue is taken as -600 =>
#     https://books.google.be/books?id=IJwZHPrDQYUC&pg=PA379&redir_esc=y#v=onepage&q&f=false
tissue_hounsfield_units = {'soft': [0, 170], 'bone': [300, np.infty], 'fat': [-300, 0], 'lung': [-np.infty, -600],
                           'soft_qaelum': [-300, 300]}
# Threshold list as used in the original Resolution paper by Sanders
sanders_threshold_list = [-475, -400, -200, -190, -175, -165, -155, -150]
# Anam (2017) Truncation correction Factor - TP is not in % but from 0-1
anam_truncation_correction = 1.15
# I did a small-scale study on the effect of thresholds for masking.
# SEE: C:\Users\ktorfs5\KU Leuven\PhD\Projects\Patient Image Quality\Minor\Speed up Masking\Small Project - Body Masking
# From there, I concluded to use only -400 HU and -200 HU as thresholds
kwinten_threshold_list = [-400, -200]


def radial_distance_xy(vector1, vector2):
    # Calculates distance on xy plane between two vectors in 3D
    z1, y1, x1 = vector1
    z2, y2, x2 = vector2
    return np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)


def conversion_factor(dw, phantom):
    a, b = ssde_coefficients[phantom]
    return a * np.exp(-b * dw)


def process_kernel(kernel):
    # This method deals with iterative kernels that are given as lists e.g; ['Br40', '3']
    #       and returns one single 'Br40-3'. We use a dash '-' as seperator
    separator = '-'
    if type(kernel) == MultiValue:
        kernel = separator.join(list(kernel))
    return kernel


def remove_truncation(image):
    image = image.astype(float)
    dimensions = image.shape
    radius_circle = dimensions[0] / 2
    corner_size = np.floor(0.75 * radius_circle * (1 - 1 / np.sqrt(2)))
    radius_corner = int(np.floor((corner_size - 1) / 2))
    corner_x_max, corner_y_max = dimensions[0] - radius_corner - 1, dimensions[1] - radius_corner - 1
    image_minimum = np.nanmin(image)
    corner_truncated = []
    for corner_mask_center_x in [radius_corner, corner_x_max]:
        for corner_mask_center_y in [radius_corner, corner_y_max]:
            corner = image[corner_mask_center_x - radius_corner:corner_mask_center_x + radius_corner + 1,
                           corner_mask_center_y - radius_corner:corner_mask_center_y + radius_corner + 1]
            corner_max = np.nanmax(corner)
            corner_truncated.append(corner_max == image_minimum)
    circular_truncation = all(corner_truncated)
    if circular_truncation:
        positions = np.arange(-radius_circle + 0.5, radius_circle, 1)
        nx, ny = np.meshgrid(positions, positions)
        distance_to_center = np.sqrt(nx ** 2 + ny ** 2)
        outside_truncation = distance_to_center > radius_circle
        image[outside_truncation] = np.nan
    return image


def body_segmentation(image, threshold_list):
    # Step 1: the image (in HU) is processed with multiple thresholds (in HU). Pixels that
    #       pass all thresholds will have value = No. thresholds
    threshold_image = np.zeros(image.shape, dtype=np.uint8)
    for threshold in threshold_list:
        threshold_image += np.array(ndimage.binary_fill_holes(image > threshold))
    nb_thresholds = len(threshold_list)
    # If pixel passes all thresholds, it is part of the binary image
    binary_image = np.array(threshold_image == nb_thresholds, dtype=np.uint8)
    # Next each pixel is numbered: background = 0, structure 1 = 1, structure 2 = 2, ...
    labels, label_nb = ndimage.label(binary_image)
    # Then count the amount of pixels per structure. Don't include background
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0
    # Body segmentation will assume that largest structure is body and thus use it for mask
    mask = np.array(labels == label_count.argmax(), dtype=np.uint8)
    mask = np.multiply(mask, 1, dtype=np.uint8)
    segment = mask.astype(np.float32)
    segment[segment == 0] = np.nan
    body = np.array(image * segment, dtype=np.float32)
    return mask, body


def tissue_fractions(image):
    # This function will calculate the fractions of different tissues in an image
    total_pixels = len(image[np.logical_not(np.isnan(image))])
    hu_soft = tissue_hounsfield_units['soft']
    hu_fat = tissue_hounsfield_units['fat']
    hu_bone = tissue_hounsfield_units['bone']
    hu_lung = tissue_hounsfield_units['lung']
    soft = len(image[np.logical_and(hu_soft[1] > image, image >= hu_soft[0])]) / total_pixels
    fat = len(image[np.logical_and(hu_fat[1] > image, image >= hu_fat[0])]) / total_pixels
    bone = len(image[np.logical_and(hu_bone[1] > image, image >= hu_bone[0])]) / total_pixels
    lung = len(image[np.logical_and(hu_lung[1] > image, image >= hu_lung[0])]) / total_pixels
    return lung, fat, soft, bone


def normalize_image(image, center, width):
    img_min = center - width // 2
    img_max = center + width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


def fraction_truncation(image, mask):
    contour = mask.copy()
    fov_contour = np.logical_not(np.isnan(image))
    fov_erosion = binary_erosion(fov_contour)
    except_edges_fov = np.logical_xor(fov_contour, fov_erosion)
    fov_contour[1:-1, 1:-1] = except_edges_fov[1:-1, 1:-1]
    erosion = binary_erosion(mask)
    except_edges = np.logical_xor(mask, erosion)
    contour[1:-1, 1:-1] = except_edges[1:-1, 1:-1]

    truncated_pixels = np.logical_and(fov_contour, contour).sum()
    non_truncated_pixels = contour.sum() - truncated_pixels

    truncated_length = truncated_pixels - 1
    non_truncated_length = non_truncated_pixels - 1

    truncated_fraction = truncated_length / (non_truncated_length + truncated_length)
    return truncated_fraction, contour, fov_contour


def wed_truncation_correction(wed, truncation_percentage, scaling_parameter=anam_truncation_correction):
    truncation_correction = np.exp(scaling_parameter * truncation_percentage ** 3)
    corrected_wed = truncation_correction * wed
    return corrected_wed, truncation_correction


# In a previous version, only files that ended on .dcm were processed. However, due to the nature of UZ Leuven
#       PACS images, this is no longer a requirement
allowed_filetypes = ['.dcm']

# This selects the window of HU that we use. Standard is 2000 so that the image values range from -1000 to 1000 HU
normalization_window = 2000
normalization_center = 0

# Coefficients to calculate the f conversion factor for the SSDE as seen in AAPM report in 2014
ssde_coefficients = {'Body': [3.704369, 0.03671937], 'Head': [1.874799, 0.03871313]}


class Image:
    def __init__(self, directory, file, process=True, thresholds=None):

        if thresholds is None:
            thresholds = sanders_threshold_list
        self.Study_ID = None
        self.Procedure = None
        self.StudyDate = None
        self.SoftwareVersion = None
        self.Protocol = None
        self.PatientSex = None
        self.BodyPart = None
        self.thresholds = thresholds
        self.WED, self.f, self.fov_contour, self.body_contour, self.WED_correction_factor = None, None, None, None, None
        self.truncated_fraction, self.WED_uncorrected, self.area, self.ctdi_phantom = None, None, None, None
        self.average_hu, self.totalCollimation, self.singleCollimation = None, None, None
        self.SeriesDescription, self.DataCollectionDiameter = None, None
        self.manufacturer, self.model, self.station, self.procedure, self.SliceNumber = None, None, None, None, None
        self.kernel, self.SliceThickness, self.channels, self.time, self.study_id = None, None, None, None, None
        self.mAs, self.mA, self.kVp, self.CTDI_vol, self.SSDE, self.Pitch = np.nan, None, None, None, None, None
        self.dicom, self.array, self.slope, self.intercept = None, None, None, None
        self.hu, self.body, self.PixelSize, self.mask, self.raw_hu = None, None, None, None, None
        self.slice_thickness, self.slice_location, self.ReconstructionTargetCenter = None, None, None
        self.ReconstructionDiameter, self.DataCollectionCenter, self.global_noise = None, None, None
        self.lung, self.fat, self.soft, self.bone = None, None, None, None
        self.file = file
        self.path = os.path.join(directory, file)
        self.filename = os.path.basename(self.path)
        self.valid = True

        if self.valid:
            self._get_dicom_file()
            if process:
                self.set_array()
                self.set_basic_dicom_info()
                self.set_secondary_dicom_info()
                self.mask_and_body_segmentation()
                self.set_tissue_fractions()
                self.calculate_ssde()

    def _transform_to_hu(self, array):
        return array * self.slope + self.intercept

    def _get_dicom_file(self):
        if self.valid:
            try:
                self.dicom = pydicom.dcmread(self.path)
            except (AttributeError, InvalidDicomError, PermissionError):
                self.valid = False

    def set_basic_dicom_info(self):
        if self.valid:
            try:
                self.kVp = int(np.round(self.dicom.KVP, 0))
            except (AttributeError, TypeError):
                self.kVp = None
            try:
                self.PixelSize = self.dicom.PixelSpacing[0]
            except AttributeError:
                self.PixelSize = None
            try:
                self.SliceThickness = self.dicom.SliceThickness
            except AttributeError:
                self.SliceThickness = None
            try:
                self.mAs = int(np.round(self.dicom.Exposure, 0))
            except (AttributeError, TypeError):
                self.mAs = np.nan
            try:
                self.Pitch = self.dicom.SpiralPitchFactor
            except (AttributeError, TypeError):
                self.Pitch = None
            try:
                self.kernel = process_kernel(self.dicom.ConvolutionKernel)
            except AttributeError:
                self.kernel = None
            try:
                self.CTDI_vol = np.round(self.dicom.CTDIvol, 2)
            except (AttributeError, TypeError):
                self.CTDI_vol = None
            try:
                self.manufacturer = self.dicom.Manufacturer
            except AttributeError:
                self.manufacturer = None
            try:
                self.model = self.dicom.ManufacturerModelName
            except AttributeError:
                self.model = None
            try:
                self.SliceNumber = int(self.dicom.InstanceNumber)
            except (AttributeError, TypeError):
                self.SliceNumber = None

    def set_secondary_dicom_info(self):
        if self.valid:
            try:
                self.SeriesDescription = self.dicom.SeriesDescription
            except AttributeError:
                self.SeriesDescription = None
            try:
                self.DataCollectionDiameter = self.dicom.DataCollectionDiameter
            except AttributeError:
                self.DataCollectionDiameter = None
            try:
                self.study_id = self.dicom.AccessionNumber
            except AttributeError:
                self.study_id = None
            try:
                self.time = self.dicom.ExposureTime
            except AttributeError:
                self.time = None
            try:
                self.procedure = self.dicom.RequestedProcedureDescription
            except AttributeError:
                self.procedure = None
            try:
                self.singleCollimation = self.dicom.SingleCollimationWidth
            except AttributeError:
                self.singleCollimation = None
            try:
                self.station = self.dicom.StationName
            except AttributeError:
                self.station = None
            try:
                self.mA = self.dicom.XRayTubeCurrent
            except AttributeError:
                self.mA = None
            try:
                self.totalCollimation = self.dicom.TotalCollimationWidth
            except AttributeError:
                self.totalCollimation = None
            try:
                self.channels = int(np.round(self.totalCollimation / self.singleCollimation, 0))
            except (AttributeError, TypeError):
                self.channels = None
            try:
                self.slice_location = self.dicom.SliceLocation
            except AttributeError:
                self.slice_location = None
            try:
                self.ReconstructionDiameter = self.dicom.ReconstructionDiameter
            except AttributeError:
                self.ReconstructionDiameter = None
            try:
                self.StudyDate = self.dicom.StudyDate
            except AttributeError:
                self.StudyDate = None
            try:
                self.SoftwareVersion = self.dicom.SoftwareVersions
            except AttributeError:
                self.SoftwareVersion = None
            try:
                self.Protocol = self.dicom.ProtocolName
            except AttributeError:
                self.Protocol = None
            try:
                self.PatientSex = self.dicom.PatientSex
            except AttributeError:
                self.PatientSex = None
            try:
                self.BodyPart = self.dicom.BodyPartExamined
            except AttributeError:
                self.BodyPart = None
            try:
                self.Procedure = self.dicom.RequestedProcedureDescription
            except AttributeError:
                self.Procedure = None
            try:
                self.Study_ID = self.dicom.AccessionNumber
            except AttributeError:
                self.Study_ID = None
            try:
                self.DataCollectionCenter = np.array(self.dicom.DataCollectionCenterPatient)
            except AttributeError:
                self.DataCollectionCenter = None
            try:
                self.ReconstructionTargetCenter = np.array(self.dicom.ReconstructionTargetCenterPatient)
            except AttributeError:
                self.ReconstructionTargetCenter = None

    def set_array(self):
        if self.valid:
            try:
                self.array = self.dicom.pixel_array.astype(float)
                self.array = remove_truncation(self.array)
                self.slope = self.dicom.RescaleSlope
                self.intercept = self.dicom.RescaleIntercept
                self.raw_hu = self._transform_to_hu(self.array)
            except (AttributeError, InvalidDicomError, PermissionError):
                self.valid = False

    def mask_and_body_segmentation(self):
        if self.valid:
            try:
                self.mask, self.body = body_segmentation(self.raw_hu, self.thresholds)
            except (AttributeError, PermissionError):
                self.valid = None

    def set_tissue_fractions(self):
        try:
            self.lung, self.fat, self.soft, self.bone = tissue_fractions(self.body)
        except (AttributeError, TypeError):
            self.lung, self.fat, self.soft, self.bone = None, None, None, None

    def calculate_ssde(self):
        try:
            ctdi_phantom_code = self.dicom.CTDIPhantomTypeCodeSequence[0].CodeValue
        except AttributeError:
            ctdi_phantom_code = None
        if ctdi_phantom_code == 113691:
            self.ctdi_phantom = 'Body'
        elif ctdi_phantom_code == 113690:
            self.ctdi_phantom = 'Head'
        else:
            self.ctdi_phantom = 'Body'

        try:
            self.average_hu = np.nanmean(self.body)
            self.area = np.sum(self.mask) * (self.PixelSize / 10) ** 2  # Cross-sectional area of patient in cm²
            self.WED_uncorrected = 2 * np.sqrt((1 + self.average_hu / 1000) * self.area / np.pi)  # water equivalent
            # diameter in cm
        except (AttributeError, TypeError):
            self.average_hu, self.area, self.WED_uncorrected = None, None, np.nan

        try:
            self.truncated_fraction, self.body_contour, self.fov_contour = fraction_truncation(self.raw_hu, self.mask)
        except (AttributeError, TypeError):
            self.truncated_fraction, self.body_contour, self.fov_contour = None, None, None
        try:
            self.WED, self.WED_correction_factor = \
                wed_truncation_correction(self.WED_uncorrected, self.truncated_fraction)
        except (AttributeError, TypeError):
            self.WED, self.WED_correction_factor = self.WED_uncorrected, None
        try:
            self.f = conversion_factor(self.WED, self.ctdi_phantom)  # ssde conversion factor in cm-1
            self.SSDE = self.f * self.CTDI_vol
        except (AttributeError, TypeError):
            self.f, self.SSDE = None, None

    def set_global_noise(self, global_noise):
        self.global_noise = global_noise

    def set_slice_number(self, new_slice_number):
        self.SliceNumber = new_slice_number


class ImageStack:

    def __init__(self, directory, process=True, thresholds=None):
        if thresholds is None:
            thresholds = sanders_threshold_list
        self.thresholds = thresholds
        self.WED_std = None
        self.WED_avg = None
        self.MatrixCenter, self.DataCollectionCenter, self.ReconstructionTargetCenter = None, None, None
        self.dx, self.dy, self.dz, self.dimensions, self.nb_slices = None, None, None, None, None
        self.slices, self.NewSliceNumbers, self.SliceNumbers = None, None, None
        self.ISOCENTER, self.totalCollimation, self.singleCollimation = None, None, None
        self.manufacturer, self.model, self.station, self.channels = None, None, None, None
        self.procedure, self.PatientSex, self.BodyPart, self.Protocol, self.Study_ID = None, None, None, None, None
        self.CTDI_vol_avg, self.CTDI_vol_std, self.SSDE_avg, self.SSDE_std = None, None, None, None
        self.mAsStack, self.mAs_avg, self.mAs_std, self.time = None, None, None, None
        self.kernel, self.kVp, self.PixelSize, self.Pitch, self.SliceThickness = None, None, None, None, None
        self.SoftwareVersion, self.ReconstructionDiameter, self.StudyDate = None, None, None
        self.mask_stack, self.raw_hu_stack, self.SSDEStack, self.WEDStack, self.CTDIStack = None, None, None, None, None
        self.valid = True
        self.directory = directory
        self._check_validness()
        self._get_slices(process)
        if process:
            self.set_arrays()
            self.set_basic_dicom_info()
            self.set_secondary_dicom_info()
            self.mask_and_body_segmentation()
            self.set_tissue_fractions()
            self.calculate_all_ssde()
            self.sort_slices()
            self.set_stacks()
            self.get_dicom_info()

    def _check_validness(self):
        self.valid = True
        self._check_directory()

    def _check_directory(self):
        self.valid = os.path.isdir(self.directory)

    def _get_slices(self, process):
        if self.valid:
            try:
                # self.slices = [Image(self.directory, file, process=process) for file in os.listdir(self.directory)]
                slices = []
                for file in os.listdir(self.directory):
                    slab = Image(self.directory, file, process=process, thresholds=self.thresholds)
                    if slab.valid:
                        slices.append(slab)
                self.slices = slices
            except (AttributeError, TypeError):
                self.slices = None
                self.valid = False

    def sort_slices(self):
        if self.valid:
            try:
                self.slices = [s for s in self.slices if s.slice_location is not None]
                self.slices.sort(key=lambda s: s.SliceNumber)
                self.slices = np.array(self.slices)
                self.SliceNumbers = np.array([s.SliceNumber for s in self.slices])
                for s in self.slices:
                    s.set_slice_number(s.SliceNumber - np.min(self.SliceNumbers) + 1)
                self.NewSliceNumbers = self.SliceNumbers - np.min(self.SliceNumbers) + 1
            except (AttributeError, TypeError):
                self.slices = None
                self.valid = False

    def set_basic_dicom_info(self):
        if self.valid:
            for slab in self.slices:
                slab.set_basic_dicom_info()

    def set_secondary_dicom_info(self):
        if self.valid:
            for slab in self.slices:
                slab.set_secondary_dicom_info()

    def mask_and_body_segmentation(self):
        if self.valid:
            for slab in self.slices:
                slab.mask_and_body_segmentation()

    def set_arrays(self):
        if self.valid:
            for slab in self.slices:
                slab.set_array()

    def set_tissue_fractions(self):
        if self.valid:
            for slab in self.slices:
                slab.set_tissue_fractions()

    def calculate_all_ssde(self):
        if self.valid:
            mas = np.array([slab.mAs for slab in self.slices])
            try:
                self.mAsStack = mas
                self.mAs_avg = np.round(np.nanmean(mas), 3)
                self.mAs_std = np.round(np.nanstd(mas), 3)
            except (AttributeError, TypeError):
                self.mAs_std, self.mAs_avg, self.mAsStack = None, None, None

            ctdi = np.array([slab.CTDI_vol for slab in self.slices])
            try:
                self.CTDIStack = ctdi
                self.CTDI_vol_avg = np.round(np.nanmean(ctdi), 3)
                self.CTDI_vol_std = np.round(np.nanstd(ctdi), 3)
            except (AttributeError, TypeError):
                self.CTDI_vol_avg, self.CTDI_vol_std, self.CTDIStack = None, None, None
            
            wed = np.array([slab.WED for slab in self.slices])
            try:
                self.WEDStack = wed
                self.WED_avg = np.round(np.nanmean(wed), 3)
                self.WED_std = np.round(np.nanstd(wed), 3)
            except (AttributeError, TypeError):
                self.WEDStack, self.WED_std, self.WED_avg = None, None, None

            ssde = np.array([slab.SSDE for slab in self.slices])
            try:
                self.SSDEStack = ssde
                self.SSDE_avg = np.round(np.nanmean(ssde), 3)
                self.SSDE_std = np.round(np.nanstd(ssde), 3)
            except (AttributeError, TypeError):
                self.SSDEStack, self.SSDE_avg, self.SSDE_std = None, None, None

    def set_stacks(self):
        if self.valid:
            try:
                self.raw_hu_stack = np.array([s.raw_hu for s in self.slices])
                self.mask_stack = np.array([s.mask for s in self.slices])
            except (AttributeError, IndexError):
                self.raw_hu_stack = None
                self.mask_stack = None

    def _find_max_radial_distance(self):
        left_top, right_top, left_bot, right_bot = [0, 0, 0], [0, 0, self.dx], [0, self.dy, 0], [0, self.dy, self.dx]
        dlt = radial_distance_xy(left_top, self.ISOCENTER)
        dlb = radial_distance_xy(left_bot, self.ISOCENTER)
        drt = radial_distance_xy(right_top, self.ISOCENTER)
        drb = radial_distance_xy(right_bot, self.ISOCENTER)
        self.max_radial_distance = max(dlt, dlb, drt, drb)

    def get_dicom_info(self):
        if self.valid:
            dicom_zero = self.slices[0]
            try:
                self.SliceThickness = dicom_zero.SliceThickness
            except AttributeError:
                self.SliceThickness = None
            try:
                self.PixelSize = dicom_zero.PixelSize
            except AttributeError:
                self.PixelSize = None
            try:
                self.ReconstructionDiameter = dicom_zero.ReconstructionDiameter
            except AttributeError:
                self.ReconstructionDiameter = None
            try:
                self.Pitch = dicom_zero.Pitch
            except AttributeError:
                self.Pitch = None
            try:
                self.StudyDate = dicom_zero.StudyDate
            except AttributeError:
                self.StudyDate = None
            try:
                self.SoftwareVersion = dicom_zero.SoftwareVersion
            except AttributeError:
                self.SoftwareVersion = None
            try:
                self.Protocol = dicom_zero.Protocol
            except AttributeError:
                self.Protocol = None
            try:
                self.PatientSex = dicom_zero.PatientSex
            except AttributeError:
                self.PatientSex = None
            try:
                self.BodyPart = dicom_zero.BodyPart
            except AttributeError:
                self.BodyPart = None

            try:
                self.kVp = dicom_zero.kVp
            except (AttributeError, TypeError):
                self.kVp = None
            try:
                self.kernel = dicom_zero.kernel
            except AttributeError:
                self.kernel = None
            try:
                self.Study_ID = dicom_zero.Study_ID
            except AttributeError:
                self.Study_ID = None
            try:
                self.time = dicom_zero.time
            except AttributeError:
                self.time = None
            try:
                self.manufacturer = dicom_zero.manufacturer
            except AttributeError:
                self.manufacturer = None
            try:
                self.model = dicom_zero.model
            except AttributeError:
                self.model = None
            try:
                self.procedure = dicom_zero.procedure
            except AttributeError:
                self.procedure = None
            try:
                self.singleCollimation = dicom_zero.singleCollimation
            except AttributeError:
                self.singleCollimation = None
            try:
                self.station = dicom_zero.station
            except AttributeError:
                self.station = None
            try:
                self.totalCollimation = dicom_zero.totalCollimation
            except AttributeError:
                self.totalCollimation = None
            try:
                self.channels = int(np.round(self.totalCollimation / self.singleCollimation, 0))
            except (AttributeError, TypeError):
                self.channels = None
            try:
                self.nb_slices = len(self.slices)
                self.dimensions = np.array([self.SliceThickness, self.PixelSize, self.PixelSize])
                self.dz, self.dy, self.dx = np.multiply(self.raw_hu_stack.shape, self.dimensions)
                self.MatrixCenter = np.array([self.dz / 2, self.dy / 2, self.dx / 2])
            except (AttributeError, IndexError, TypeError):
                pass
            try:
                self.ReconstructionTargetCenter = dicom_zero.ReconstructionTargetCenter
                self.DataCollectionCenter = dicom_zero.DataCollectionCenter
                self.ISOCENTER = self.MatrixCenter - self.ReconstructionTargetCenter + self.DataCollectionCenter
            except (AttributeError, ValueError, TypeError):
                self.ReconstructionTargetCenter = self.MatrixCenter
                self.DataCollectionCenter = self.MatrixCenter
                self.ISOCENTER = self.MatrixCenter
            try:
                self._find_max_radial_distance()
            except (AttributeError, TypeError):
                pass
