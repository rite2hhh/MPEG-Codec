# MPEG-Codec
MPEG Codec implemented in MATLAB for a class project. Implemented the Encoder and Decoder using helper functions. Each helper function defined to perform a specific task for the Encoder or Decoder.


MPEG Encoder:
-------------
An MPEG encoder has various blocks. It takes in an uncompressed video sequence (generally in RGB color space) as its input, converts it to YCbCr color space. In this color space, we have a Luma component (Y), and two Chroma Components (Cb & Cr). For any image, maximum information is retained in the Luma component, so we can leave this luma component out of the compression equation/ perform minimal compression and get a high definition image. Since we have the Chroma components isolated, we can apply high compression to them to save bandwidth, and still recover a frame similar to the original uncompressed frame at the output.
Most encoders have the following blocks to convert a raw uncompressed video sequence to MPEG, some with slight modifications for better performance or video quality.

(a) RGB - YCbCr

(b) Pass (for I-Frames)/ Subtract(for P-/B-Frames)

(b) Discrete Cosine Transform (DCT)

(c) Quantization

(d) Entropy/ Source Coding

(e) Multiplexer

(f) Source Buffer

Files included in this repository and their use loosely align with each of these blocks:
1. README.m - Run this in MATLAB to invoke entire project
2. Encoder.m - MPEG Encoder Blocks
3. Decoder.m - MPEG Decoder Blocks
4. exhaustiveSearchAlgorithm.m - helper function required for Motion Compensation
5. differentialCoding.m - helper function for difference frame between succesive frames
6. zigZag.m - helper function (Run Length Encoding)
7. inverseZigZag.m
8. inverseDifferentialCoding.m
9. inverseDCT.m
10. inbuildDecoder.m
