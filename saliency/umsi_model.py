#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
umsi_model.py — UMSI++ (Unified Model of Saliency and Importance ++)
=====================================================================
Faithful TF2 / Keras 3 port of the UMSI++ architecture from:
  Jiang et al., "UEyes: Understanding Visual Saliency across User
  Interface Types", CHI 2023.

Original code: https://github.com/YueJiang-nj/UEyes-CHI2023
Original env:  TF 1.14 + Keras 2.3.1 + CUDA 9.0 + Python 3.7

This port runs on:
  TF 2.16 + Keras 3.10 + Apple Silicon (M4 Mac, CPU)

Architecture summary
--------------------
Input(256×256×3, BGR, VGG-mean-subtracted)
  → Custom Xception backbone (stride-modified blocks 4, 13, exit)
    Output: 32×32×2048
  → ASPP branch (dilation 6, 12, 18) + 1×1 conv  →  concat  → 32×32×1024
  → Classification branch (Conv→GAP→Dense→Softmax 6-class)
    + tiled dense embedding fused via concatenation      → 32×32×1280
  → Decoder (Conv-Dropout-UpSample chain)
    → 512×512×1 heatmap
Outputs: [heatmap_512×512, classification_6]

Weights
-------
The pretrained weights file ``umsi++.hdf5`` is downloaded from:
  https://userinterfaces.aalto.fi/ueyeschi23/model_weights.zip
and placed in ``saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5``.

Usage
-----
    from saliency.umsi_model import UMSIPlus
    model = UMSIPlus("saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5")
    heatmap = model.predict_saliency("path/to/screenshot.png")
"""

from __future__ import annotations

import os

# ── Suppress TF noise ──────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# ── TF / Keras 3 imports ──────────────────────────────────────────────────
import tensorflow as tf
import keras
from keras import layers, Model

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Constants (matching original sal_imp_utilities.py)
# ============================================================================
SHAPE_R: int = 256          # model input height
SHAPE_C: int = 256          # model input width
SHAPE_R_OUT: int = 512      # model output height
SHAPE_C_OUT: int = 512      # model output width

# VGG mean values (BGR channel order, ImageNet stats)
VGG_MEAN_B: float = 103.939
VGG_MEAN_G: float = 116.779
VGG_MEAN_R: float = 123.68

# ============================================================================
# Custom Xception Backbone (stride-modified for saliency)
# ============================================================================

def _build_custom_xception(img_input: tf.Tensor) -> tf.Tensor:
    """Build the custom Xception from UEyes UMSI++.

    Key modifications vs. standard Keras Xception:
      - Block 4: residual conv stride (1,1) instead of (2,2);
                 MaxPool stride (1,1) instead of (2,2)
      - Block 13 (exit): residual conv stride (1,1); MaxPool stride (1,1)
    This keeps spatial resolution at 32×32 for 256×256 input (instead of 8×8).

    Returns the final feature tensor (None, 32, 32, 2048).
    """
    # ── Entry flow ──────────────────────────────────────────────────────
    # Block 1 — two standard convolutions with stride-2 first conv
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,
                       name='block1_conv1')(img_input)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    # Block 2 — SeparableConv + MaxPool (stride 2)
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same',
                              use_bias=False, name='conv2d_1')(x)
    residual = layers.BatchNormalization(name='batch_normalization_1')(residual)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                                name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                                name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                             name='block2_pool')(x)
    x = layers.add([x, residual])

    # Block 3 — SeparableConv + MaxPool (stride 2)
    residual = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same',
                              use_bias=False, name='conv2d_2')(x)
    residual = layers.BatchNormalization(name='batch_normalization_2')(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False,
                                name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False,
                                name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                             name='block3_pool')(x)
    x = layers.add([x, residual])

    # ── Block 4 — MODIFIED: stride (1,1) to preserve spatial res ───────
    residual = layers.Conv2D(728, (1, 1), strides=(1, 1), padding='same',
                              use_bias=False, name='conv2d_3')(x)
    residual = layers.BatchNormalization(name='batch_normalization_3')(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                             name='block4_pool')(x)
    x = layers.add([x, residual])

    # ── Middle flow — Blocks 5-12 (8 repeated blocks) ──────────────────
    for i in range(8):
        residual = x
        prefix = f'block{i + 5}'
        x = layers.Activation('relu', name=f'{prefix}_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                    name=f'{prefix}_sepconv1')(x)
        x = layers.BatchNormalization(name=f'{prefix}_sepconv1_bn')(x)
        x = layers.Activation('relu', name=f'{prefix}_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                    name=f'{prefix}_sepconv2')(x)
        x = layers.BatchNormalization(name=f'{prefix}_sepconv2_bn')(x)
        x = layers.Activation('relu', name=f'{prefix}_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                    name=f'{prefix}_sepconv3')(x)
        x = layers.BatchNormalization(name=f'{prefix}_sepconv3_bn')(x)
        x = layers.add([x, residual])

    # ── Exit flow — Block 13 — MODIFIED: stride (1,1) ─────────────────
    residual = layers.Conv2D(1024, (1, 1), strides=(1, 1), padding='same',
                              use_bias=False, name='conv2d_4')(x)
    residual = layers.BatchNormalization(name='batch_normalization_4')(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False,
                                name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False,
                                name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                             name='block13_pool')(x)
    x = layers.add([x, residual])

    # ── Block 14 — final SeparableConvs ────────────────────────────────
    x = layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False,
                                name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False,
                                name='block14_sepconv2')(x)
    x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    return x


# ============================================================================
# UMSI++ Full Model
# ============================================================================

def build_umsi_model(input_shape: Tuple[int, int, int] = (SHAPE_R, SHAPE_C, 3),
                     verbose: bool = False) -> Model:
    """Build the UMSI++ architecture (saliency + classification).

    Architecture (Jiang et al., CHI 2023, Figure 2):
      Xception(custom) →  ASPP(d=6,12,18) + Classification(6-class)
        → Concatenate → Decoder → 512×512 heatmap

    Key architectural references:
      Xception backbone: Chollet, F. (2017). Xception: Deep learning with
        depthwise separable convolutions. CVPR 2017, 1251–1258.
      ASPP (Atrous Spatial Pyramid Pooling): Chen, L.-C., Papandreou, G.,
        Kokkinos, I., Murphy, K., & Yuille, A.L. (2017). DeepLab: Semantic
        image segmentation with deep convolutional nets, atrous convolution,
        and fully connected CRFs. IEEE TPAMI, 40(4), 834–848.
      UMSI++ training + design-class head: Jiang, Y. et al. (2023). UEyes:
        Understanding visual saliency across user interface types. CHI 2023.
        https://doi.org/10.1145/3544548.3581096

    Returns:
        keras.Model with two outputs:
          [0] out_heatmap  — shape (batch, 512, 512, 1)
          [1] out_classif  — shape (batch, 6)
    """
    inp = layers.Input(shape=input_shape, name='input_1')

    # ── Xception backbone (as sub-model to match original structure) ───
    xception_out = _build_custom_xception(inp)
    # Wrap in sub-model (matches original: xception is a Model inside UMSI)
    xception = Model(inp, xception_out, name='xception')
    backbone_feat = xception.output   # (None, 32, 32, 2048)

    # ── ASPP branch ────────────────────────────────────────────────────
    # Atrous Spatial Pyramid Pooling captures multi-scale context by applying
    # parallel dilated convolutions with rates 6, 12, 18.
    # Dilation rates from DeepLab (Chen et al., 2017, Table 1) and adopted
    # unchanged in UMSI++ (Jiang et al., 2023).
    c0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                        name='aspp_csep0')(backbone_feat)
    # Dilated depthwise + pointwise (rate=6)
    c6 = layers.DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding='same',
                                 use_bias=False,
                                 name='aspp_csepd6_depthwise')(backbone_feat)
    # Dilated depthwise + pointwise (rate=12)
    c12 = layers.DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding='same',
                                  use_bias=False,
                                  name='aspp_csepd12_depthwise')(backbone_feat)
    # Dilated depthwise + pointwise (rate=18)
    c18 = layers.DepthwiseConv2D((3, 3), dilation_rate=(18, 18), padding='same',
                                  use_bias=False,
                                  name='aspp_csepd18_depthwise')(backbone_feat)

    c6 = layers.BatchNormalization(name='aspp_csepd6_depthwise_BN')(c6)
    c12 = layers.BatchNormalization(name='aspp_csepd12_depthwise_BN')(c12)
    c18 = layers.BatchNormalization(name='aspp_csepd18_depthwise_BN')(c18)
    c6 = layers.Activation('relu', name='activation_2')(c6)
    c12 = layers.Activation('relu', name='activation_4')(c12)
    c18 = layers.Activation('relu', name='activation_6')(c18)
    c6 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                        name='aspp_csepd6_pointwise')(c6)
    c12 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                         name='aspp_csepd12_pointwise')(c12)
    c18 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                         name='aspp_csepd18_pointwise')(c18)

    c0 = layers.BatchNormalization(name='aspp0_BN')(c0)
    c6 = layers.BatchNormalization(name='aspp_csepd6_pointwise_BN')(c6)
    c12 = layers.BatchNormalization(name='aspp_csepd12_pointwise_BN')(c12)
    c18 = layers.BatchNormalization(name='aspp_csepd18_pointwise_BN')(c18)

    c0 = layers.Activation('relu', name='aspp0_activation')(c0)
    c6 = layers.Activation('relu', name='activation_3')(c6)
    c12 = layers.Activation('relu', name='activation_5')(c12)
    c18 = layers.Activation('relu', name='activation_7')(c18)

    concat_aspp = layers.Concatenate(name='concatenate_1')([c0, c6, c12, c18])
    # → shape: (batch, 32, 32, 1024)

    # ── Classification branch ─────────────────────────────────────────
    cl = layers.Conv2D(256, (3, 3), strides=(3, 3), padding='same',
                        use_bias=False, name='global_conv')(backbone_feat)
    cl = layers.BatchNormalization(name='global_BN')(cl)
    cl = layers.Activation('relu', name='activation_1')(cl)
    cl = layers.Dropout(0.3, name='dropout_1')(cl)
    cl = layers.GlobalAveragePooling2D(
                        name='global_average_pooling2d_1')(cl)
    cl = layers.Dense(256, name='global_dense')(cl)
    classif_feat = layers.Dropout(0.3, name='dropout_2')(cl)
    out_classif = layers.Dense(6, activation='softmax',
                                name='out_classif')(classif_feat)

    # Fusion: tile the 256-d classification embedding to 32×32 spatial
    fusion = layers.Dense(256, name='dense_fusion')(classif_feat)

    def tile_to_spatial(x):
        """Reshape (batch, 256) → (batch, 32, 32, 256) by tiling."""
        x = tf.reshape(x, (tf.shape(x)[0], 1, 1, 256))
        # tile to 32×32 via concatenation (matches original implementation)
        con = [x for _ in range(32)]
        con = tf.concat(con, axis=1)           # → (batch, 32, 1, 256)
        con = tf.concat([con for _ in range(32)], axis=2)  # → (batch, 32, 32, 256)
        return con

    fusion_tiled = layers.Lambda(tile_to_spatial, name='lambda_1')(fusion)
    # → shape: (batch, 32, 32, 256)

    # Merge ASPP and classification
    concat_all = layers.Concatenate(name='concatenate_2')(
                                     [concat_aspp, fusion_tiled])
    # → shape: (batch, 32, 32, 1280)

    # ── Decoder ────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                       name='concat_projection')(concat_all)
    x = layers.BatchNormalization(name='concat_projection_BN')(x)
    x = layers.Activation('relu', name='activation_8')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)

    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False,
                       name='dec_c1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False,
                       name='dec_c2')(x)
    x = layers.Dropout(0.3, name='dec_dp1')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                              name='dec_ups1')(x)
    # → (batch, 64, 64, 256)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                       name='dec_c3')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                       name='dec_c4')(x)
    x = layers.Dropout(0.3, name='dec_dp2')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                              name='dec_ups2')(x)
    # → (batch, 128, 128, 128)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                       name='dec_c5')(x)
    x = layers.Dropout(0.3, name='dec_dp3')(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear',
                              name='dec_ups3')(x)
    # → (batch, 512, 512, 64)

    out_heatmap = layers.Conv2D(1, (1, 1), padding='same', use_bias=False,
                                 name='dec_c_cout')(x)
    # → (batch, 512, 512, 1)

    # ── Assemble model ────────────────────────────────────────────────
    model = Model(inputs=inp, outputs=[out_heatmap, out_classif],
                   name='umsi_plus_plus')

    if verbose:
        model.summary()

    return model


# ============================================================================
# Preprocessing & Postprocessing (matching original sal_imp_utilities.py)
# ============================================================================

def preprocess_image(image_path: str,
                     shape_r: int = SHAPE_R,
                     shape_c: int = SHAPE_C) -> np.ndarray:
    """Load and preprocess a single image for UMSI++ inference.

    Steps:
      1. Load as BGR (OpenCV default)
      2. Aspect-ratio-preserving resize + zero-padding to (shape_r, shape_c)
      3. VGG mean subtraction (BGR channels)

    Args:
        image_path: Path to the input image.
        shape_r: Target height (default 256).
        shape_c: Target width (default 256).

    Returns:
        np.ndarray of shape (1, shape_r, shape_c, 3) — preprocessed, float32.
    """
    # Read as BGR
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Aspect-ratio-preserving resize with zero-padding
    padded = _padding(img_bgr, shape_r, shape_c)

    # To float32 and VGG mean subtraction
    img = padded.astype(np.float32)
    img[..., 0] -= VGG_MEAN_B   # Blue
    img[..., 1] -= VGG_MEAN_G   # Green
    img[..., 2] -= VGG_MEAN_R   # Red

    return np.expand_dims(img, axis=0)   # add batch dimension


def _padding(img: np.ndarray, shape_r: int, shape_c: int,
             channels: int = 3) -> np.ndarray:
    """Aspect-ratio-preserving resize + zero-padding.

    Matches the original ``padding()`` function from sal_imp_utilities.py.
    """
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        offset = (img_padded.shape[1] - new_cols) // 2
        img_padded[:, offset:offset + new_cols] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        offset = (img_padded.shape[0] - new_rows) // 2
        img_padded[offset:offset + new_rows, :] = img

    return img_padded


def postprocess_saliency(pred: np.ndarray,
                         original_h: int,
                         original_w: int) -> np.ndarray:
    """Resize the 512×512 prediction back to the original image dimensions.

    Reverses the padding that was applied during preprocessing, then resizes
    the unpadded prediction to the original image size.

    Args:
        pred: Raw model output, shape (512, 512) or (512, 512, 1).
        original_h: Original image height in pixels.
        original_w: Original image width in pixels.

    Returns:
        Saliency heatmap of shape (original_h, original_w), float32,
        normalized to [0, 1].
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    pred_shape = pred.shape
    rows_rate = original_h / pred_shape[0]
    cols_rate = original_w / pred_shape[1]

    if rows_rate > cols_rate:
        new_cols = (pred_shape[1] * original_h) // pred_shape[0]
        pred = cv2.resize(pred, (new_cols, original_h))
        offset = (pred.shape[1] - original_w) // 2
        img = pred[:, offset:offset + original_w]
    else:
        new_rows = (pred_shape[0] * original_w) // pred_shape[1]
        pred = cv2.resize(pred, (original_w, new_rows))
        offset = (pred.shape[0] - original_h) // 2
        img = pred[offset:offset + original_h, :]

    # Normalize to [0, 1]
    vmax = img.max()
    if vmax > 0:
        img = img / vmax

    return img.astype(np.float32)


# ============================================================================
# High-Level Inference Wrapper
# ============================================================================

class UMSIPlus:
    """High-level wrapper for UMSI++ saliency prediction.

    Usage:
        model = UMSIPlus("saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5")
        heatmap = model.predict_saliency("screenshot.png")
    """

    # Design-type labels (6-class head from Jiang et al., CHI 2023, §3.2).
    # The model was trained on UEyes (1,980 screenshots, 62 participants)
    # across these six UI/image categories.
    DESIGN_CLASSES = [
        "poster",
        "infographic",
        "mobile_ui",
        "desktop_ui",
        "web_page",
        "natural_image",
    ]

    def __init__(self, weights_path: Union[str, Path],
                 verbose: bool = False):
        """Initialize the UMSI++ model and load pretrained weights.

        Args:
            weights_path: Path to the ``umsi++.hdf5`` weight file.
            verbose: If True, print model summary.
        """
        self.model = build_umsi_model(verbose=verbose)
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"UMSI++ weights not found at: {weights_path}\n"
                "Download from: https://userinterfaces.aalto.fi/ueyeschi23/"
                "model_weights.zip"
            )
        # Load the pretrained weights.
        # We load positionally (skip_mismatch=False) so that ANY architecture
        # mismatch fails loudly. A silent skip_mismatch=True fallback is
        # deliberately NOT used: it would leave mismatched layers randomly
        # initialised while still serving predictions labelled "UMSI++", which
        # would invalidate every downstream saliency result without warning.
        try:
            self.model.load_weights(str(weights_path), skip_mismatch=False)
            print(f"[UMSI++] Weights loaded (positional): {weights_path}")
        except Exception as e:
            raise RuntimeError(
                f"UMSI++ weights at {weights_path} do not match the model "
                f"architecture (positional load failed: {e!r}). Refusing to "
                "fall back to a partial load, which would silently random-"
                "initialise mismatched layers. Verify the weight file is the "
                "UEyes-trained UMSI++ checkpoint."
            ) from e

    def predict_saliency(self, image_path: Union[str, Path],
                         return_classif: bool = False
                         ) -> Union[np.ndarray,
                                    Tuple[np.ndarray, np.ndarray]]:
        """Predict saliency heatmap for a single image.

        Args:
            image_path: Path to input image (PNG, JPG, etc.).
            return_classif: If True, also return the 6-class classification.

        Returns:
            heatmap: Saliency map of shape (H, W), float32 in [0, 1],
                     at the original image resolution.
            classif: (optional) 6-class probability vector, shape (6,).
        """
        image_path = str(image_path)

        # Read original dimensions
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        orig_h, orig_w = img_orig.shape[:2]

        # Preprocess
        img_batch = preprocess_image(image_path)

        # Predict
        preds = self.model.predict(img_batch, verbose=0)
        raw_heatmap = preds[0][0]   # (512, 512, 1)
        classif = preds[1][0]       # (6,)

        # Postprocess
        heatmap = postprocess_saliency(raw_heatmap, orig_h, orig_w)

        if return_classif:
            return heatmap, classif
        return heatmap

    def predict_batch(self, image_paths: list,
                      batch_size: int = 4) -> list:
        """Predict saliency for a batch of images.

        Args:
            image_paths: List of image paths.
            batch_size: How many to process at once.

        Returns:
            List of (heatmap, classif) tuples.
        """
        results = []
        for path in image_paths:
            heatmap, classif = self.predict_saliency(
                path, return_classif=True
            )
            results.append((heatmap, classif))
        return results


# ============================================================================
# CLI test
# ============================================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="UMSI++ Saliency Prediction")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--weights", default="saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5",
                        help="Path to UMSI++ weights")
    parser.add_argument("--output", default=None,
                        help="Path to save the saliency heatmap")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = UMSIPlus(args.weights, verbose=args.verbose)
    heatmap, classif = model.predict_saliency(args.image, return_classif=True)

    # Print classification
    for i, (cls, prob) in enumerate(zip(UMSIPlus.DESIGN_CLASSES, classif)):
        print(f"  {cls}: {prob:.4f}")

    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Save if requested
    if args.output:
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        cv2.imwrite(args.output, heatmap_uint8)
        print(f"Saved: {args.output}")
