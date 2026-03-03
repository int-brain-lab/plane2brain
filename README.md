# plane2brain
_plane2brain_ is a tool to map image locations from 2-photon imaging experiments to coordinates in brain atlases (such as the Allen Mouse Brain CCF). The core use case of the tool is to convert from pixels to ML,AP,DV coordinates, and thus assigning brain regions to imaged pixels.

## how it works
_plane2brain_ works by combinding two coordinate systems:
- the coordinate system of the imaging plane (2d)
- a coordinate system of the atlas (3d)

The coordinate system of the imaging plane is inserted in to the 3d with it's corresponding tilt and rotation, and pixel locations in the imaging plane are then projected along the optical axis until the hit the atlas surface. Optionally, surface coordinates are further resolved into brain coordinate by projecting inwards from the surface point along the local brain surface normal by the depth of the imaged pixel (if known, for example by a reference stack)

All operations work on the basis of coordinates, hence this tool works (and is intended to be used) on the outputs of suite2p or other segmentation algorithms, and can thus be used to predict which brain region an imaged soma is belonging to.

## when to use
this only works under the following assumptions:

1) there is a point in any of the images that has a known location in the atlas coordinate system

this can be any previously mapped point or mark, a known location of a fiducual mark, a patterns in blood vasculature with known location, or if the center of the craniotomy was planned out with known ML, AP coordinates (and the surgery was carried out with sufficient precision).
For example: during the imaging session, the imaging objective is positioned so that its optical axis intersects with the center of a circular cranial window, and the center of the cranial window is known relative to bregma.

2) during the imaging experiment, the normal vector of the brain surface and the optical axis of the imaging objective are collinear OR the angle between them is known.

In the ideal experiment, a) the cranial window is implanted perfectly tangential to the brains surface, and b) the optical axis of the imaging objective and the normal of the glass window are collinear, minimizing reflecive losses and hence maximizing the coupling efficiency of laser excitation light into the brain. Either this is the case, or the deviation from this ideal scenario is quantified.
For example: if a reference stack with planes at different depths is taken at the imaged location, the difference between the brain normal and the optical axis (the 'tilt') can be inferred from the imaging data.

if such a reference stack is aquired as a part of an imaging session, this data can be used to correct for both session to session variability (such as daily positioning errors) and inter-individual variability (such as imperfectly implanted cranial windows)

### additional requirements
 - there is a way to express the coordinates of each image in a common reference frame, such as defined by the galvo angles. If you are using scanimage, this data is automatically written into the metadata files. If you are using a different system, please reach out.

## how to use
### installation


