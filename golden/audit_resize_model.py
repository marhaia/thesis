"""AUDIT-ONLY UMSI++ variant builder (modern TF2.16 / Keras 3 side).

This module is used EXCLUSIVELY by the resize causality experiment. It does NOT
modify any production file: it imports the frozen production Xception backbone
(saliency.umsi_model._build_custom_xception) read-only and reconstructs the
ASPP + classification + decoder graph with byte-for-byte identical layer names
and hyper-parameters, so the authoritative checkpoint's weights transfer 1:1 by
name. The ONLY thing that can differ is which bilinear resize operator each
decoder upsampling stage uses:

  * production/modern : keras UpSampling2D(interpolation='bilinear')
                        (TF2/Keras3 half-pixel resize semantics)
  * audit "legacy"    : tf.raw_ops.ResizeBilinear(align_corners=False,
                        half_pixel_centers=False)  (TF1.14/Keras2.3.1 semantics)

align_corners=True is deliberately never used.

Weights are copied by name from a freshly loaded production model, so every
Conv/BatchNorm/Dense weight is identical across variants and the resize
operator is provably the only changed computation.
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers, Model

# Read-only imports from the FROZEN production module (no edits).
from saliency.umsi_model import (
    _build_custom_xception, build_umsi_model, UMSIPlus,
    SHAPE_R, SHAPE_C,
)


class LegacyResizeBilinear(layers.Layer):
    """Audit-only bilinear upsample reproducing TF1.14/Keras2.3.1 semantics via
    tf.raw_ops.ResizeBilinear(align_corners=False, half_pixel_centers=False)."""

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = tuple(int(f) for f in factor)

    def call(self, inputs):
        shp = tf.shape(inputs)
        out_h = shp[1] * self.factor[0]
        out_w = shp[2] * self.factor[1]
        size = tf.stack([out_h, out_w])
        return tf.raw_ops.ResizeBilinear(
            images=inputs, size=size,
            align_corners=False, half_pixel_centers=False)

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        return (b,
                None if h is None else h * self.factor[0],
                None if w is None else w * self.factor[1],
                c)

    def get_config(self):
        cfg = super().get_config()
        cfg["factor"] = self.factor
        return cfg


def _ups(x, factor, name, legacy_set):
    """Emit either the production keras UpSampling2D or the audit legacy
    resize, keeping the SAME layer name and factor."""
    if name in legacy_set:
        return LegacyResizeBilinear(factor, name=name)(x)
    return layers.UpSampling2D(size=factor, interpolation='bilinear',
                               name=name)(x)


def build_umsi_variant(legacy_resize_layers=()):
    """Reconstruct the UMSI++ graph with the chosen decoder resize layers
    replaced by the audit legacy operator. Layer names/params are identical to
    saliency.umsi_model.build_umsi_model except for the selected resize ops."""
    legacy_set = set(legacy_resize_layers)

    inp = layers.Input(shape=(SHAPE_R, SHAPE_C, 3), name='input_1')

    # Frozen production backbone (read-only import), wrapped identically.
    xception_out = _build_custom_xception(inp)
    xception = Model(inp, xception_out, name='xception')
    backbone_feat = xception.output

    # ── ASPP branch (verbatim names/params from production) ──
    c0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False,
                       name='aspp_csep0')(backbone_feat)
    c6 = layers.DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding='same',
                                use_bias=False,
                                name='aspp_csepd6_depthwise')(backbone_feat)
    c12 = layers.DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding='same',
                                 use_bias=False,
                                 name='aspp_csepd12_depthwise')(backbone_feat)
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

    # ── Classification branch ──
    cl = layers.Conv2D(256, (3, 3), strides=(3, 3), padding='same',
                       use_bias=False, name='global_conv')(backbone_feat)
    cl = layers.BatchNormalization(name='global_BN')(cl)
    cl = layers.Activation('relu', name='activation_1')(cl)
    cl = layers.Dropout(0.3, name='dropout_1')(cl)
    cl = layers.GlobalAveragePooling2D(name='global_average_pooling2d_1')(cl)
    cl = layers.Dense(256, name='global_dense')(cl)
    classif_feat = layers.Dropout(0.3, name='dropout_2')(cl)
    out_classif = layers.Dense(6, activation='softmax',
                               name='out_classif')(classif_feat)

    fusion = layers.Dense(256, name='dense_fusion')(classif_feat)

    def tile_to_spatial(x):
        x = tf.reshape(x, (tf.shape(x)[0], 1, 1, 256))
        con = [x for _ in range(32)]
        con = tf.concat(con, axis=1)
        con = tf.concat([con for _ in range(32)], axis=2)
        return con

    fusion_tiled = layers.Lambda(tile_to_spatial, name='lambda_1')(fusion)
    concat_all = layers.Concatenate(name='concatenate_2')(
        [concat_aspp, fusion_tiled])

    # ── Decoder (resize ops selectable) ──
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
    x = _ups(x, (2, 2), 'dec_ups1', legacy_set)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                      name='dec_c3')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                      name='dec_c4')(x)
    x = layers.Dropout(0.3, name='dec_dp2')(x)
    x = _ups(x, (2, 2), 'dec_ups2', legacy_set)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                      name='dec_c5')(x)
    x = layers.Dropout(0.3, name='dec_dp3')(x)
    x = _ups(x, (4, 4), 'dec_ups3', legacy_set)

    out_heatmap = layers.Conv2D(1, (1, 1), padding='same', use_bias=False,
                                name='dec_c_cout')(x)

    return Model(inputs=inp, outputs=[out_heatmap, out_classif],
                 name='umsi_plus_plus_audit')


def copy_weights_by_name(dst, src):
    """Copy weights layer-by-layer by name from src (loaded production model)
    to dst (variant). Returns (copied, param_count). Raises on any weighted
    layer whose name/shape does not match, so a silent mismatch is impossible."""
    src_by_name = {l.name: l for l in src.layers}
    copied = 0
    params = 0
    for lyr in dst.layers:
        w = lyr.get_weights()
        if not w:
            continue
        if lyr.name not in src_by_name:
            raise RuntimeError("variant layer %r missing from production model"
                               % lyr.name)
        sw = src_by_name[lyr.name].get_weights()
        if len(sw) != len(w) or any(a.shape != b.shape
                                    for a, b in zip(sw, w)):
            raise RuntimeError("weight shape mismatch at layer %r" % lyr.name)
        lyr.set_weights(sw)
        copied += 1
        params += int(sum(a.size for a in sw))
    return copied, params


def load_production_model(ckpt):
    """Build + strictly load the frozen production model (authoritative
    weights source)."""
    return UMSIPlus(ckpt).model


def build_variant_with_weights(ckpt, legacy_resize_layers, src_model=None):
    """Build a variant and populate it with the authoritative weights.

    If src_model is provided (an already loaded production model) it is reused
    as the weight source; otherwise one is built + loaded."""
    if src_model is None:
        src_model = load_production_model(ckpt)
    variant = build_umsi_variant(legacy_resize_layers)
    copied, params = copy_weights_by_name(variant, src_model)
    return variant, copied, params
