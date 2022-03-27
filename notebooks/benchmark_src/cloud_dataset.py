import numpy as np
import pandas as pd
import rasterio
import torch
from typing import Optional, List
import albumentations as A

class CloudDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """
    
    def __init__(
        self,
        x_paths: pd.DataFrame, # : syntax specifies datatype for each parameter for CloudDataset objects
        bands: List[str],
        y_paths: Optional[pd.DataFrame] = None,
        transforms: Optional[list] = None,
    ):
        """
        Instantiate the CloudDataset class.

        Args:
            x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands
            bands (list[str]): list of the bands included in the data
            y_paths (pd.DataFrame, optional): a dataframe with a for each chip and columns for chip_id
                and the path to the label TIF with ground truth cloud cover
            transforms (list, optional): list of transforms to apply to the feature data (eg augmentations)
        """
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms
        self.bands = bands

    def __len__(self):
        return len(self.data)
    
    # Similar to pop, helps iterate through dataset
    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1) # Creates 512x512x4 ndarray

        # Apply data augmentations, if provided
        if self.transforms:
            # Subset each band's image to get 4 arrays that are each 512x512
            band_02_arr = x_arr[:, :, 0]
            band_03_arr = x_arr[:, :, 1]
            band_04_arr = x_arr[:, :, 2]
            band_08_arr = x_arr[:, :, 3]
            
            # Apply transform to each band's image
            band_02_arr_transformed = self.transforms(image=band_02_arr)["image"]
            band_03_arr_transformed = self.transforms(image=band_03_arr)["image"]
            band_04_arr_transformed = self.transforms(image=band_04_arr)["image"]
            band_08_arr_transformed = self.transforms(image=band_08_arr)["image"]
            
            # Recombine transformed images back into 512x512x4 ndarray
            x_arr = np.dstack((band_02_arr_transformed,
                               band_03_arr_transformed,
                               band_04_arr_transformed,
                               band_08_arr_transformed))

        # re-orders array to match expected format needed for model    
        x_arr = np.transpose(x_arr, [2, 0, 1]) 
        
        # Prepare dictionary for item
        item = {"chip_id": img.chip_id, "chip": x_arr}
        
        # Spatial transforms are valid transforms to apply to label (unlike pixel transforms)
        spatial_transforms = [A.augmentations.geometric.transforms.Affine,
                    A.augmentations.crops.transforms.CenterCrop,
                    A.augmentations.transforms.CoarseDropout,
                    A.augmentations.crops.transforms.Crop,
                    A.augmentations.crops.transforms.CropAndPad,
                    A.augmentations.crops.transforms.CropNonEmptyMaskIfExists,
                    A.augmentations.geometric.transforms.ElasticTransform,
                    A.augmentations.transforms.Flip,
                    A.augmentations.transforms.GridDistortion,
                    A.augmentations.transforms.GridDropout,
                    A.augmentations.transforms.HorizontalFlip,
                    A.augmentations.transforms.Lambda,
                    A.augmentations.geometric.resize.LongestMaxSize,
                    A.augmentations.transforms.MaskDropout,
                    A.augmentations.transforms.NoOp,
                    A.augmentations.transforms.OpticalDistortion,
                    A.augmentations.transforms.PadIfNeeded,
                    A.augmentations.geometric.transforms.Perspective,
                    A.augmentations.geometric.transforms.PiecewiseAffine,
                    # A.augmentations.transforms.PixelDropout, # doesn't match docs for some reason
                    A.augmentations.crops.transforms.RandomCrop,
                    A.augmentations.crops.transforms.RandomCropNearBBox,
                    A.augmentations.transforms.RandomGridShuffle,
                    A.augmentations.crops.transforms.RandomResizedCrop,
                    A.augmentations.geometric.rotate.RandomRotate90,
                    A.augmentations.geometric.resize.RandomScale,
                    A.augmentations.crops.transforms.RandomSizedBBoxSafeCrop,
                    A.augmentations.crops.transforms.RandomSizedCrop,
                    A.augmentations.geometric.resize.Resize,
                    A.augmentations.geometric.rotate.Rotate,
                    A.augmentations.geometric.rotate.SafeRotate,
                    A.augmentations.geometric.transforms.ShiftScaleRotate,
                    A.augmentations.geometric.resize.SmallestMaxSize,
                    A.augmentations.transforms.Transpose,
                    A.augmentations.transforms.VerticalFlip]

        # Load label if available
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1).astype("float32")
            
            # Apply data augmentations to the label - ONLY SPATIAL TRANSFORMS CAN BE APPLIED TO LABEL
            if self.transforms:
                
                # Create list of valid spatial transforms from list of transforms applied to train images
                valid_label_transforms = [transform for transform in self.transforms if type(transform) in spatial_transforms]
                
                # Apply only valid transforms to the label
                self.transforms = A.Compose(valid_label_transforms)
                y_arr = self.transforms(image=y_arr)["image"]
                print(np.unique(y_arr))
            item["label"] = y_arr

        return item
