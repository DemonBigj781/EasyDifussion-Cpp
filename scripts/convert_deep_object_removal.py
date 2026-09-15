#!/usr/bin/env python3
"""Convert VPanjeta/Deep-Object-Removal's TF1 checkpoint to portable ONNX.

The conversion environment needs TensorFlow 2.x with compat.v1 plus tf2onnx.
Neither dependency is needed by the Easy Diffusion runtime after conversion.
"""

import argparse
from pathlib import Path


def build_graph(tf):
    tf.compat.v1.disable_eager_execution()
    source = tf.compat.v1.placeholder(tf.float32, [1, 400, 400, 3], name="input")

    def conv(bottom, shape, stride, name):
        with tf.compat.v1.variable_scope(name):
            weight = tf.compat.v1.get_variable("W", shape=shape)
            bias = tf.compat.v1.get_variable("b", shape=shape[-1])
            return tf.nn.bias_add(
                tf.nn.conv2d(bottom, weight, [1, stride, stride, 1], padding="SAME"),
                bias,
            )

    def deconv(bottom, shape, target, name):
        with tf.compat.v1.variable_scope(name):
            weight = tf.compat.v1.get_variable("W", shape=shape)
            bias = tf.compat.v1.get_variable("b", shape=shape[-2])
            return tf.nn.bias_add(
                tf.nn.conv2d_transpose(bottom, weight, target.shape.as_list(), [1, 2, 2, 1], padding="SAME"),
                bias,
            )

    with tf.compat.v1.variable_scope("GEN"):
        conv1_1 = tf.nn.elu(conv(source, [3, 3, 3, 32], 1, "conv1_1"))
        conv1_2 = tf.nn.elu(conv(conv1_1, [3, 3, 32, 32], 1, "conv1_2"))
        conv1_stride = conv(conv1_2, [3, 3, 32, 32], 2, "conv1_stride")

        conv2_1 = tf.nn.elu(conv(conv1_stride, [3, 3, 32, 64], 1, "conv2_1"))
        conv2_2 = tf.nn.elu(conv(conv2_1, [3, 3, 64, 64], 1, "conv2_2"))
        conv2_stride = conv(conv2_2, [3, 3, 64, 64], 2, "conv2_stride")

        conv3_1 = tf.nn.elu(conv(conv2_stride, [3, 3, 64, 128], 1, "conv3_1"))
        conv3_2 = tf.nn.elu(conv(conv3_1, [3, 3, 128, 128], 1, "conv3_2"))
        conv3_3 = tf.nn.elu(conv(conv3_2, [3, 3, 128, 128], 1, "conv3_3"))
        conv3_4 = tf.nn.elu(conv(conv3_3, [3, 3, 128, 128], 1, "conv3_4"))
        conv3_stride = conv(conv3_4, [3, 3, 128, 128], 2, "conv3_stride")
        conv4_stride = tf.nn.elu(conv(conv3_stride, [3, 3, 128, 128], 2, "conv4_stride"))
        conv5_stride = tf.nn.elu(conv(conv4_stride, [3, 3, 128, 128], 2, "conv5_stride"))
        conv6_stride = tf.nn.elu(conv(conv5_stride, [3, 3, 128, 128], 2, "conv6_stride"))

        deconv5 = tf.nn.elu(deconv(conv6_stride, [3, 3, 128, 128], conv5_stride, "deconv5_fs"))
        skip5 = tf.concat([deconv5, conv5_stride], 3)
        deconv4 = tf.nn.elu(deconv(skip5, [3, 3, 128, 256], conv4_stride, "deconv4_fs"))
        skip4 = tf.concat([deconv4, conv4_stride], 3)
        deconv3 = tf.nn.elu(deconv(skip4, [3, 3, 128, 256], conv3_stride, "deconv3_fs"))
        skip3 = tf.concat([deconv3, conv3_stride], 3)
        deconv2 = tf.nn.elu(deconv(skip3, [3, 3, 64, 256], conv2_stride, "deconv2_fs"))
        skip2 = tf.concat([deconv2, conv2_stride], 3)
        deconv1 = tf.nn.elu(deconv(skip2, [3, 3, 32, 128], conv1_stride, "deconv1_fs"))
        skip1 = tf.concat([deconv1, conv1_stride], 3)
        reconstruction = deconv(skip1, [3, 3, 3, 64], source, "recon")

    return source, tf.identity(reconstruction, name="output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import tensorflow as tf
    import tf2onnx

    graph = tf.Graph()
    with graph.as_default():
        build_graph(tf)
        saver = tf.compat.v1.train.Saver()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tf.compat.v1.Session(graph=graph) as session:
        saver.restore(session, str(args.checkpoint))
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            session,
            graph.as_graph_def(),
            ["output"],
        )

    model, _ = tf2onnx.convert.from_graph_def(
        frozen,
        input_names=["input:0"],
        output_names=["output:0"],
        opset=args.opset,
        output_path=str(args.output),
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {len(model.graph.node)} nodes)")


if __name__ == "__main__":
    main()
