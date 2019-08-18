## Evaluation - Characterisation - Utilities

### compare_segmentations

Allows either to compare two segmentation or pairs of segmentation and reference as specified per folder.

Output summary as a csv file listing for each pair of reference / segmentation image the measures of interest

List of measures includes:

-  Volume of each, 
- Number of true positives, true negatives, false positives, false negatives
- Outline error divided into false positives and negatives
- Detection error divided in false positives and negatives
- Dice score
- Jaccard score
- Hausdorff distance 
- Average distance
- Volume difference

If specified, with the -save_map option, the maps of different types of error is saved as well



### compute_ROI_statistics

Allows the computation of mask based statistics over an image from distribution features to texture features

Returns a csv file with all the required measures for all the files listed with pairs of mask and image

Measures include:

- Distribution features
  - Mean, min, max, std
  - 1st, 5th, 25th, 50th, 75th, 95th, 99th percentile
  - Skewness, Kurtosis
- Shape features
  - Volume, surface, surface to area volume, compactness
  - Border (external, internal)
- Texture features 
  - Haralick features

### parcellation_translation

Enable the translation between different parcellation labelling. 

Notably available GIF (Cardoso et al 2015) to Freesurfer labelling

With the option mask_mapping enables the relabeling over a mask

With the subcommand split_label, enables the split of a given label according to a direction based on ellipsoid decomposition

### parcellation_applications

Allows the parsing of xml files (specifically GIF output) into a database with all labels and hierarchically organised aggregates

With the subcommand seg_aggregate allows for the aggregation of multiple region to create a coarser segmentation from a parcellation files. Useful to get whole lobes or specific sublobes.

### process_orientation

Gather small utility function to reorient nifti files, transform from flirt to niftyreg affine matrix (similarly to nifty reg) and from niftyreg to flirt affine matrix.

Check anisotropy, and acquisition of provided images.

