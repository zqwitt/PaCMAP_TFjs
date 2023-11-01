import * as tf from "@tensorflow/tfjs-node";

export function normalizedDistance(distances) {
  return tf.tidy(() => {
    const k = 7;

    const topKNearestDistances = tf.topk(distances.neg(), k, true).values;
    const fourthToSixthNearestDistances = topKNearestDistances.slice([0, 4], [-1, -1]);
    const averageFourthToSixthNearestDistances = fourthToSixthNearestDistances
      .neg()
      .sum(-1)
      .div(tf.scalar(3))
      .expandDims(1);
    const sigma = averageFourthToSixthNearestDistances.matMul(averageFourthToSixthNearestDistances, false, true);

    return distances.square().div(sigma);
  });
}
