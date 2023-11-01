import * as tf from "@tensorflow/tfjs-node";
import ProgressBar from "progress";

import { euclideanDistance } from "./distances/euclidean.js";
import { normalizedDistance } from "./distances/normalized.js";

class PaCMAP {
  constructor({
    nDimensions = 2,
    numNeighbourPairs = 10,
    ratioMidNearPairs = 0.5,
    ratioFurtherPairs = 2.0,
    learningRate = 1,
    numIterations = 450,
  } = {}) {
    this.nDimensions = nDimensions;
    this.numNeighbourPairs = numNeighbourPairs;
    this.ratioMidNearPairs = ratioMidNearPairs;
    this.ratioFurtherPairs = ratioFurtherPairs;
    this.learningRate = learningRate;
    this.numIterations = numIterations;

    if (this.nDimensions < 2) throw new Error("The number of projection dimensions must be at least 2.");
    if (this.learningRate <= 0) throw new Error("The learning rate must be larger than 0.");
  }

  decideNumPairs() {
    if (this.numNeighbourPairs == undefined) {
      if (this.N <= 10000) {
        this.numNeighbourPairs = 10;
      } else {
        this.numNeighbourPairs = Math.round(10 + 15 * (Math.log10(this.N) - 4));
      }
    }
    this.numMidNearPairs = Math.round(this.numNeighbourPairs * this.ratioMidNearPairs);
    this.numFurtherPairs = Math.round(this.numNeighbourPairs * this.ratioFurtherPairs);
    if (this.numNeighbourPairs < 1) throw new Error("The number of nearest neighbors can't be less than 1");
    if (this.numMidNearPairs < 1) throw new Error("The number of mid-near pairs can't be less than 1");
    if (this.numFurtherPairs < 1) throw new Error("The number of further pairs can't be less than 1");
  }

  findNeighbourPairs() {
    return tf.tidy(() => {
      //Normalize the distances using (||x_i-x_j||^2)/σ_ij
      const normalizedDistances = normalizedDistance(this.distances);
      //Find the nearest neigbours
      const neighbourPairs = tf
        .topk(normalizedDistances.neg(), this.numNeighbourPairs + 1, true)
        .indices.slice([0, 1], [-1, -1]);
      return neighbourPairs;
    });
  }

  findMidNearPairs() {
    return tf.tidy(() => {
      const observationIndices = tf
        .range(0, this.N, 1, "int32")
        .reshape([this.N, 1])
        .expandDims(2)
        .expandDims(3)
        .tile([1, 5, 6, 1]);
      let samples = [];
      for (let i = 0; i < this.N; i++) {
        const indices = tf.rand(
          [5, 6, 1],
          () => {
            let keep = false;
            while (keep == false) {
              let random = Math.floor(Math.random() * this.N);
              if (random != i) {
                keep = true;
                return random;
              }
            }
          },
          "int32"
        );
        samples.push(indices);
      }
      const randomIndices = tf.stack(samples);
      const combinedIndices = tf.concat([observationIndices, randomIndices], 3);
      //Gather 6 random observations
      const distanceSamples = tf.gatherND(this.distances, combinedIndices);
      //Find 2nd nearest neighbours
      const topTwoDistances = tf.topk(distanceSamples.neg(), 2).indices.slice([0, 0, 1]);
      const observationIndicesForTopPairs = tf
        .range(0, this.N, 1, "int32")
        .reshape([this.N, 1])
        .expandDims()
        .expandDims(2)
        .tile([1, 1, 5, 1]);
      const sampleIndicesForTopPairs = tf
        .range(0, 5, 1, "int32")
        .reshape([5, 1])
        .expandDims()
        .expandDims()
        .tile([1, this.N, 1, 1]);
      const combinedIndicesForTopPairs = tf.concat(
        [observationIndicesForTopPairs, sampleIndicesForTopPairs, topTwoDistances.expandDims()],
        3
      );
      //Gather 2nd nearest sample indices
      const midNearPairs = tf.gatherND(randomIndices.squeeze(), combinedIndicesForTopPairs).squeeze();
      return midNearPairs;
    });
  }

  findFurtherPairs() {
    return tf.tidy(() => {
      let samples = [];
      for (let i = 0; i < this.N; i++) {
        const neighbourPairs = tf.gather(this.neighbourPairs, [i]).squeeze();
        const observations = tf.rand(
          [this.numFurtherPairs],
          () => {
            let keep = false;
            const pairs = neighbourPairs.dataSync();
            while (keep == false) {
              let random = Math.floor(Math.random() * this.N);
              if (!pairs.includes(random) && random != i) {
                keep = true;
                return random;
              }
            }
          },
          "int32"
        );
        samples.push(observations);
      }
      const furtherPairs = tf.stack(samples);
      return furtherPairs;
    });
  }

  lossNeighbourPairs() {
    const J = tf.gather(this.Y, this.neighbourPairs);
    const numerator = euclideanDistance(this.Y, J).square().add(tf.scalar(1));
    const denominator = tf.scalar(10).add(numerator);
    const loss = numerator.div(denominator).sum();
    return loss;
  }

  lossMidNearPairs() {
    const J = tf.gather(this.Y, this.midNearPairs);
    const numerator = euclideanDistance(this.Y, J).square().add(tf.scalar(1));
    const denominator = tf.scalar(10000).add(numerator);
    const loss = numerator.div(denominator).sum();
    return loss;
  }

  lossFurtherPairs() {
    const J = tf.gather(this.Y, this.furtherPairs);
    const numerator = tf.scalar(1);
    const denominator = tf.scalar(1).add(euclideanDistance(this.Y, J).square().add(tf.scalar(1)));
    const loss = numerator.div(denominator).sum();

    return loss;
  }

  findPairs() {
    this.distances = euclideanDistance(this.X, this.X);
    this.neighbourPairs = this.findNeighbourPairs();
    this.midNearPairs = this.findMidNearPairs();
    this.furtherPairs = this.findFurtherPairs();
  }

  fit(X, init = "random") {
    tf.tidy(() => {
      this.X = X;
      this.N = X.shape[0];
      this.decideNumPairs();
      this.findPairs();
      if (init === "random") this.Y = tf.variable(tf.randomNormal([this.N, this.nDimensions]));
      const optimizer = tf.train.adagrad(this.learningRate);

      let bar = new ProgressBar(":bar :current / :total", { total: this.numIterations, width: 50 });
      for (let i = 0; i < this.numIterations; i++) {
        let weightNeighbourPairs = 2;
        let weightMidNearPairs = 1000 * (1 - (i - 1) / 100) + 3 * ((i - 1) / 100);
        let weightFurtherPairs = 1;
        if (i >= 101) {
          weightNeighbourPairs = 3;
          weightMidNearPairs = 3;
          weightFurtherPairs = 1;
        } else if (i >= 201) {
          weightNeighbourPairs = 1;
          weightMidNearPairs = 0;
          weightFurtherPairs = 1;
        }
        optimizer.minimize(() => {
          let np = tf.scalar(weightNeighbourPairs).mul(this.lossNeighbourPairs());
          let mp = tf.scalar(weightMidNearPairs).mul(this.lossMidNearPairs());
          let fp = tf.scalar(weightFurtherPairs).mul(this.lossFurtherPairs());

          return np.add(mp).add(fp);
        });
        bar.tick();
      }
    });
  }
}

export default PaCMAP;
