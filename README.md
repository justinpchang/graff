# graff

STATUS: PROOF OF CONCEPT

hand-draw tag registration and recognition

## algorithm

Given an image of a tag and a library of registered tag feature vectors, determine the most closely matching tag to the given image.

- Convert image to white on black binary image [algorithm#binarize]
- Clean image (erode then dilate) [algorithm#clean]
- Center image (get bounding box) [algorithm#center]
- Generate feature vector (not needed?)
- For each registered tag [algorithm#minimize]
  - Calculate Hausdorff distance between image and tag [algorithm#hausdorff]
- Return nearest tag