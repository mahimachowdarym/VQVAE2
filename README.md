# Vector-Quantized Variational Autoencoder 2 (VQ-VAE-2)



This repository contains an implementation of the Vector-Quantized Variational Autoencoder (VQ-VAE-2) in PyTorch. This model builds on the original VQ-VAE, adding multiple levels of quantized latent variables to capture rich hierarchical features of the data.

* Custom VQ layers for hierarchical quantization.
* Mechanisms for updating and reviving unused codebook entries.
* Efficient distance computation between embeddings and inputs.
* Encoders that reduce input dimensionality and apply vector quantization.
* Decoders that reconstruct data from the quantized representations.
* Mechanisms to handle "dead" entries in the codebook, ensuring efficient use of the quantization space.

Encoder
Encoders reduce the dimensionality of the input data and apply vector quantization. We provide several encoder architectures, including:

QuarterEncoder: Reduces dimensions by a factor of 4.
HalfEncoder: Reduces dimensions by a factor of 2.
Decoder
Decoders reconstruct the input data from the quantized representations. Available decoders include:

QuarterDecoder: Upsamples by a factor of 4.
HalfDecoder: Upsamples by a factor of 2.
HalfQuarterDecoder: Combines upsampling by factors of 2 and 4.

## PixelCNN Implementation
PixelCNN: This class represents a stack of PixelConv layers.

Methods:
__init__: Initializes the PixelCNN with a variable number of layers.
forward: Applies the layers sequentially, assuming the first layer is a PixelConvA and the rest are PixelConvB.
PixelConv: An abstract base class for PixelCNN layers.

Attributes:
depth_in: Number of input filters.
depth_out: Number of output filters.
cond_depth: Depth of the conditioning channels.
horizontal, vertical: Receptive fields of the horizontal and vertical stacks.
Methods:
__init__: Initializes the PixelConv with the specified attributes.
_init_directional_convs: Abstract method to initialize the directional convolutions.
_run_stacks: Runs the vertical and horizontal stacks.
_run_padded_vertical: Abstract method to run the vertical stack with padding.
_run_padded_horizontal: Abstract method to run the horizontal stack with padding.
_compute_cond_bias: Computes the conditional bias if conditioning is provided.
PixelConvA: The first layer in a PixelCNN.

Methods:
__init__: Initializes the PixelConvA with the specified attributes.
forward: Applies the layer to the images, producing latents.
_init_directional_convs: Initializes the vertical and horizontal convolutions.
_run_padded_vertical: Runs the vertical stack with padding.
_run_padded_horizontal: Runs the horizontal stack with padding.
PixelConvB: Any layer except the first in a PixelCNN.

Methods:
__init__: Initializes the PixelConvB with the specified attributes.
forward: Applies the layer to the outputs of previous vertical and horizontal stacks.
_init_directional_convs: Initializes the vertical and horizontal convolutions.
_run_padded_vertical: Runs the vertical stack with padding.
_run_padded_horizontal: Runs the horizontal stack with padding.



