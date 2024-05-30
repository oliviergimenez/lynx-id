import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Probabilities ands weights of applying transformations
p_XXX corresponds to the probability of applying a transformation or a selection of transformation. Usually they are set to 0.5 to not have all the possible groups of transformations applied.
0.5 might already be to strong of a factor but training models and testing transforms will determine it.
w_XXX corresponds to weights for transformations. They are used to determine the probability of the selection of a transformation in a group of transformations.
By default, all weights are put to 1 to have equal chances between transformations and allow "comprehensive changes".
"""
p_fromfloat = 1 #always to allow the use of the transformations which needs uint8
p_longestmaxsize=1 # To resize all images to the same size
p_pad = 1 # To resize all images to the same square size
p_hflip = 0 #0 because could cause confusion between "mirrored lynx"
p_crop = 0.2 #low because it's a heavy transformation that creates difficult data
p_ToFloat = 1 #Set to 1 to retrieve the original format in floats

# Image quality
p_image_quality = 0.5
w_iq_blur = 1
w_iq_downscale = 1
w_iq_gaussianblur = 1
w_iq_motionblur = 1
w_iq_sharpen = 1
w_iq_noop = 1

# Obstructions
p_obstructions = 0.5
w_obs_griddropout = 1
w_obs_fog = 1
w_obs_sunflare = 1
w_obs_coarsedropout = 1
w_obs_noop = 1

# Color changes
p_color = 0.5
w_color_gray = 1
w_color_hsv = 1 
w_color_rgb = 1
w_color_tonecurve = 1
w_color_brightnesscontrast = 1
w_color_gamma = 1
w_color_noop = 1

# Geometric changes
p_geometric = 0.5
w_geometric_griddistortion = 1
w_geometric_opticaldistortion = 1
w_geometric_perspective = 1
w_geometric_noop =1

# Noise
p_noise = 0.5
w_noise_multiplicative = 1
w_noise_noop = 1    

#    A.FromFloat(dtype='uint8', max_value=None, always_apply=False, p=p_fromfloat),

transforms = A.Compose([
    A.LongestMaxSize(max_size=224, p=p_longestmaxsize),  # Resize the longest side to 224
    A.PadIfNeeded(min_height=224, min_width=224, p=p_pad),  # Pad to make the image 224x224
    ToTensorV2(),
])

augments = A.Compose([
    A.RandomSizedCrop (min_max_height=(512,1024), height=1024, width=1024, w2h_ratio=1.0, interpolation=1, always_apply=False, p=p_crop),
    A.OneOf([
        #A.GaussNoise(var_limit=(20, 200), mean=0, per_channel=False, always_apply=False, p=1),
        #A.ISONoise(color_shift=(0.02, 0.1), intensity=(0.3, 0.7), always_apply=False, p=1),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=True, always_apply=False, p=w_noise_multiplicative),
        A.NoOp(p=w_noise_noop),
    ], p=p_noise),
    A.OneOf([
        A.Blur(blur_limit=(3,7), always_apply=False, p=w_iq_blur),
        A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=0, always_apply=False, p=w_iq_downscale),
        A.GaussianBlur(blur_limit=(7, 7), sigma_limit=0, always_apply=False, p=w_iq_gaussianblur),
        #A.MedianBlur(blur_limit=(3, 7), always_apply=False, p=1),
        A.MotionBlur(blur_limit=(3, 11), always_apply=False, p=w_iq_motionblur),
        A.Sharpen(alpha=(0.3, 1), lightness=(0.5, 1.0), always_apply=False, p=w_iq_sharpen),
        A.NoOp(w_iq_noop) 
    ], p=p_image_quality),
    A.OneOf([
        A.GridDropout(ratio=0.4, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=True, fill_value=0, mask_fill_value=None, always_apply=False, p=w_obs_griddropout),
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.40, always_apply=False, p=w_obs_fog),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=3, num_flare_circles_upper=6, src_radius=50, src_color=(255, 255, 255), always_apply=False, p=w_obs_sunflare),
        A.CoarseDropout(max_holes=3, max_height=75, max_width=75, min_holes=2, min_height=30, min_width=30, fill_value=0, mask_fill_value=None, always_apply=False, p=w_obs_coarsedropout),
        A.NoOp(p=w_obs_noop)
    ], p=p_obstructions),
    A.OneOf([
        A.ToGray(p=w_color_gray),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, always_apply=False, p=w_color_hsv),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=w_color_rgb), 
        A.RandomToneCurve(scale=0.3, always_apply=False, p=w_color_tonecurve),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=w_color_brightnesscontrast),
        A.RandomGamma(gamma_limit=(50, 150), eps=None, always_apply=False, p=w_color_gamma),
        A.NoOp(p=w_color_noop),
    ], p=p_color),
    A.OneOf([
        #A.Affine(scale=1.0, translate_percent=0, translate_px=None, rotate=(-30,30), shear=(-45,45), interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=True, always_apply=False, keep_ratio=True, p=1),
        A.GridDistortion(num_steps=7, distort_limit=0.4, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=w_geometric_griddistortion),
        A.OpticalDistortion(distort_limit=0.4, shift_limit=0.4, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=w_geometric_opticaldistortion),
        A.Perspective(scale=(0.05, 0.2), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=w_geometric_perspective),
        #A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=False, keypoints_threshold=0.01, p=1),
        #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=1),
        A.NoOp(p=w_geometric_noop),
    ], p=p_geometric),
    A.ToFloat(max_value=255, always_apply=True, p=p_ToFloat)
])

# Define the transformations using Albumentations
image_size = 896
transforms_dinov2 = A.Compose([
    A.LongestMaxSize(max_size=image_size, p=p_longestmaxsize),  # Resize the longest side to image_size   
    A.PadIfNeeded(min_height=image_size, min_width=image_size, p=p_pad),  # Pad to make the image image_size*image_size    
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])



augments_dinov2 = A.Compose([
    A.LongestMaxSize(max_size=image_size, p=p_longestmaxsize),  # Resize the longest side to image_size   
    A.PadIfNeeded(min_height=image_size, min_width=image_size, p=p_pad),  # Pad to make the image image_size*image_size    
    # Noise block    
    A.OneOf([
        A.GaussNoise(var_limit=(10, 100), mean=0, per_channel=True, p=1),  # Adjusted for more subtle and realistic noise impact
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),  # Lowered parameters to simulate more accurate digital camera noise
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=True, p=w_noise_multiplicative),
        A.NoOp(p=w_noise_noop),
    ], p=p_noise),
    # Blur/Sharp block    
    A.OneOf([
        A.Blur(blur_limit=(5, 10), always_apply=False, p=w_iq_blur),
        A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=0, always_apply=False, p=w_iq_downscale),
        A.MotionBlur(blur_limit=(5, 15), always_apply=False, p=w_iq_motionblur),
        A.Sharpen(alpha=(0.5, 1), lightness=(0.75, 1.5), always_apply=False, p=w_iq_sharpen),
        A.NoOp(w_iq_noop) 
    ], p=p_image_quality),

    # Visual Distortion Augmentation Block
    A.OneOf([
        A.GridDropout(ratio=0.4, unit_size_min=50, unit_size_max=100, holes_number_x=5, holes_number_y=5, shift_x=0, shift_y=0, random_offset=True, fill_value=0, mask_fill_value=None, always_apply=False, p=w_obs_griddropout),
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.40, always_apply=False, p=w_obs_fog),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=3, num_flare_circles_upper=6, src_radius=50, src_color=(255, 255, 255), always_apply=False, p=w_obs_sunflare),
        A.CoarseDropout(max_holes=3, max_height=75, max_width=75, min_holes=2, min_height=30, min_width=30, fill_value=0, mask_fill_value=None, always_apply=False, p=w_obs_coarsedropout),
        A.NoOp(p=w_obs_noop)
    ], p=p_obstructions),

    # Color Augmentation Block
    A.OneOf([
        A.ToGray(p=w_color_gray),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=w_color_hsv),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, always_apply=False, p=w_color_rgb),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=w_color_brightnesscontrast),
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=w_color_gamma),
        A.NoOp(p=w_color_noop),
    ], p=p_color),

    # Geometric Augmentation Block
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=w_geometric_griddistortion),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=w_geometric_opticaldistortion),
        A.Perspective(scale=(0.05, 0.15), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=w_geometric_perspective),
        A.NoOp(p=w_geometric_noop),
    ], p=p_geometric),
    #A.ToFloat(max_value=255, always_apply=True, p=p_ToFloat)
    #ToTensorV2(),    
])

