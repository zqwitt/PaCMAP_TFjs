import * as tf from "@tensorflow/tfjs-node";

export function euclideanDistance(a, b) {
  return tf.tidy(() => a.expandDims(1).sub(b).square().sum(-1).sqrt());
}
