# graff

STATUS: PROOF OF CONCEPT

hand-draw tag registration and recognition

## algorithm

Given an image of a tag and a library of registered tag feature vectors, determine the most closely matching tag to the given image.

- Convert image to white on black binary image
- Clean image (erode then dilate)
- Center image (get bounding box)
- Generate feature vector
- For each registered tag
  - Calculate Hausdorff distance between image and tag
- Return nearest tag