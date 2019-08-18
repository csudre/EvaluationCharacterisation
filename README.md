## Evaluation Characterisation Utilities

A pot-pourri of utilities related to the processing and analysis of segmentation, parcellation files, reorientation, translation between common parcellation atlases

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

Different subcommand application:

- **parsing:** Allows the parsing of xml files (specifically GIF output) into a database with all labels and hierarchically organised aggregates
- **seg_aggregate:** Allows for the aggregation of multiple region to create a coarser segmentation from a parcellation files. Useful to get whole lobes or specific sublobes.
- **database_fromparc:** When no xml file is available but a list of nifti parcellation files, create the database using the hierarchy file.

### process_orientation

Gather small utility function to reorient nifti files, transform from flirt to niftyreg affine matrix (similarly to nifty reg) and from niftyreg to flirt affine matrix.

Different subcommands are

- **checks:** Check anisotropy, and acquisition of provided images.
- **flirtnr:** Performs the transformation of an affine matrix from FLIRT into NiftyReg format. Requires the floating image, reference image and the matrix file to modify
- **nrflirt:** Performs the transformation of an affine matrix from niftyreg into flirt affine matrix. Requires the floating image, reference image and the matrix file to modify
- **reor:** Reorient the supplied nifti image into the desired orientation


