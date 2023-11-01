import PaCMAP from "pacmap_tfjs";
import * as tf from "@tensorflow/tfjs-node";

import MAMMOTH from "./mammoth_3d.js"; // example 3D Mammoth data from PAIR-code - (https://github.com/PAIR-code/understanding-umap/blob/master/raw_data/mammoth_3d.json)

tf.tidy(() => {
  const pacmap = new PaCMAP();
  const X = tf.tensor(MAMMOTH);
  pacmap.fit(X);
  console.log(pacmap.Y.arraySync());
});
