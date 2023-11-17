#!/bin/bash

# Prefix for `rename_file` function parameters
BASE_DIR="/gpfsscratch/rech/ads/commun/datasets/extracted/Deep learning lynx - data/0_dataset_raw"

# Definition of the rename function
rename_file() {
    current_name="${BASE_DIR}$1"
    new_name="${BASE_DIR}$2"
    
    mv -v "$current_name" "$new_name"
}

# Renaming images that can be realised
rename_file "/0_dataset_Marie_OCS/OCS_Arcos/Arcos_NA_OCS_NA_2021-05-13-NA_7.JPG" "/0_dataset_Marie_OCS/OCS_Arcos/Arcos_OCS_NA_2021-05-13_NA_7.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-093=2217=2218/F39-093=2217=2218_OFB_OCELLES_2021-05-07-Les-Planches-En-Montagne_37.jpg" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-093=2217=2218/F39-093=2217=2218_OFB_OCELLES_2021-05-07_Les-Planches-En-Montagne_37.jpg"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-007/F01-007_OFB_OCELLES_2013-03-08-Valfin-sur-Valouse_13.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-007/F01-007_OFB_OCELLES_2013-03-08_Valfin-sur-Valouse_13.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-007/F01-007_OFB_OCELLES_2013-03-08-Valfin-sur-Valouse_12.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-007/F01-007_OFB_OCELLES_2013-03-08_Valfin-sur-Valouse_12.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_1353=L306/1353=L306_OFB_OCELLES_2019-04-20-Les-Pontets_2.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_1353=L306/1353=L306_OFB_OCELLES_2019-04-20_Les-Pontets_2.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-29-Mittlach_27.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-29_Mittlach_27.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-20-Mittlach_26.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-20_Mittlach_26.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-20-Mittlach_25.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-20_Mittlach_25.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-29-Mittlach_28.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-06-29_Mittlach_28.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-05-06-Mittlach_24.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-05-06_Mittlach_24.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-05-06-Mittlach_23.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_ARCOS=629/ARCOS=629_OFB_OCELLES_2020-05-06_Mittlach_23.JPG"
rename_file "/0_dataset_Marie_OFB_spots/OFB_SPOTS_987/987_OFB_SPOTS_2022-03-02-NA_1.jpg" "/0_dataset_Marie_OFB_spots/OFB_SPOTS_987/987_OFB_SPOTS_2022-03-02_NA_1.jpg"
rename_file "/0_dataset_Marie_OCS/OCS_B310/B310_OCS_NA_2012-12-19_2.jpg" "/0_dataset_Marie_OCS/OCS_B310/B310_OCS_NA_2012-12-19_NA_2.jpg"
rename_file "/0_dataset_Marie_OCS/OCS_B310/B310_OCS_NA_2012-12-23_1.jpg" "/0_dataset_Marie_OCS/OCS_B310/B310_OCS_NA_2012-12-23_NA_1.jpg"

# Images that are too complicated to rename, a `broken` prefix is added so that they are not used in the dataset.
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/1556_FD_2021-07-04-21_100_1-FDC39-Mont-sur-Monnet (2).jpg" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/broken_1556_FD_2021-07-04-21_100_1-FDC39-Mont-sur-Monnet (2).jpg"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/300_2021-05-02-100_1-FDC39-Mont-sur-Monnet.jpg" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/broken_300_2021-05-02-100_1-FDC39-Mont-sur-Monnet.jpg"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/1556_FD_2021-05-26-21_100_1-FDC39-Mont-sur-Monnet.jpg" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/broken_1556_FD_2021-05-26-21_100_1-FDC39-Mont-sur-Monnet.jpg"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/1556_FG__2021-07-04-21_100_1-FDC39-Mont-sur-Monnet (1).jpg" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-095=1556/broken_1556_FG__2021-07-04-21_100_1-FDC39-Mont-sur-Monnet (1).jpg"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-038/Villard_F39-038_OFB_OCELLES_2015-03-22_Villard-Saint-Sauveur_4322_03_15.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F39-038/broken_Villard_F39-038_OFB_OCELLES_2015-03-22_Villard-Saint-Sauveur_4322_03_15.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-047=922/FDC01_point_36.2_evosges_le_col_2018_09_01_flanc_droit.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-047=922/broken_FDC01_point_36.2_evosges_le_col_2018_09_01_flanc_droit.JPG"
rename_file "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-047=922/FDC01_point_15.2_corlier_montlier_2019_02_10_flanc_droit_erreur date-F01_047.JPG" "/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F01-047=922/broken_FDC01_point_15.2_corlier_montlier_2019_02_10_flanc_droit_erreur date-F01_047.JPG"

# Note:
# There is also a problem with 2 images linked to truncated images
# /gpfsscratch/rech/ads/commun/datasets/extracted/Deep learning lynx - data/0_dataset_raw/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_1376/1376_OFB_OCELLES_2020-06-15_NA_23.png
# /gpfsscratch/rech/ads/commun/datasets/extracted/Deep learning lynx - data/0_dataset_raw/0_dataset_Marie_OFB_ocelles/OFB_OCELLES_F25-067=1376/F25-067=1376_OFB_OCELLES_2020-06-15_Orchamps-Vennes_29.png