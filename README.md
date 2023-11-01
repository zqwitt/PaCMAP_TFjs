# PaCMAP_TFjs - PaCMAP (Pairwise Controlled Manifold Approximation) with TensorFlow.js

PaCMAP_TFjs is a JavaScript implementation of PaCMAP using Tensorflow.js to allow for speedy euclidean distance operations, top-K neighbours calculations, and adagrad optimizations.

## PaCMAP

PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction method that can be used for visualization, preserving both local and global structure of the data in original space. PaCMAP optimizes the low dimensional embedding using three kinds of pairs of points: neighbour pairs ```neighbourPairs```, mid-near pairs ```midNearPairs```, and further pairs ```furtherPairs```.

The algorithm used in PaCMAP_TFjs is based on [Yingfan Wang et al., "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization"](http://jmlr.org/papers/v22/20-1061.html) Published in the Journal of Machine Learning Research, Volume 22, Number 201, 2021, Pages 1-73.  

It is heavily inspired by the [Python implementation](https://github.com/YingfanWang/PaCMAP) by Yingfan Wang but is also missing most of the functionality.
PacMAP_TFjs is:
- Always non-deterministic
- Only initializes Y from a random gaussian distribution
- Does not have the ability to add new points after fitting
- Cannot take intermittent snapshots
- Can only calculate Euclidean Distances
- etc...

## Installation

To use PaCMAP_TFjs in your project, you'll need to install it as a dependency. You can do this using npm or yarn:
```
npm install pacmap_tfjs
```
or 
```
yarn add pacmap_tfjs
```

## Usage

Here's an example of how to use PaCMAP_TFjs to fit your data:

```javascript
import PaCMAP from "pacmap_tfjs";
import * as tf from "@tensorflow/tfjs-node";

import MAMMOTH from "./mammoth_3d.js"; // example 3D Mammoth data from PAIR-code - (https://github.com/PAIR-code/understanding-umap/blob/master/raw_data/mammoth_3d.json)

tf.tidy(() => {
    const pacmap = new PaCMAP();
    const X = tf.tensor(MAMMOTH);
    pacmap.fit(X);
    console.log(pacmap.Y.arraySync())
});

```
<table>
    <tr>
        <td>
            <img src="https://github.com/zqwitt/PaCMAP_TFjs/blob/main/pacmap/example/mammoth-3d.png" alt="Mammoth 3D" width="300"/>
        </td>
        <td>
            <img src="https://github.com/zqwitt/PaCMAP_TFjs/blob/main/pacmap/example/mammoth-2d.png" alt="Mammoth 2D" width="300"/>
        </td>
    </tr>
</table>

### Constructor Options

You can configure PaCMAP_TFjs by passing an options object to the constructor:

    nDimensions (default: 2): The number of dimensions in the low-dimensional embedding.
    numNeighbourPairs (default: 10): The number of nearest neighbor pairs to consider.
    ratioMidNearPairs (default: 0.5): The ratio of mid-near pairs to consider.
    ratioFurtherPairs (default: 2.0): The ratio of further pairs to consider.
    learningRate (default: 1): The learning rate for optimization.
    numIterations (default: 450): The number of optimization iterations.


### Methods

    fit(X): Fits Y to the given high-dimensional data X.

## License

You can find more details in the LICENSE file.

## Contributing

If you want to contribute to the project or report issues, please check the CONTRIBUTING guidelines.

## Acknowledgments

The algorithm used in PaCMAP_TFjs is based on [Yingfan Wang et al., "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization"](http://jmlr.org/papers/v22/20-1061.html) Published in the Journal of Machine Learning Research, Volume 22, Number 201, 2021, Pages 1-73.  

It is heavily inspired by the [Python implementation](https://github.com/YingfanWang/PaCMAP) by Yingfan Wang but is also missing most of the functionality.

## Credits

This project was developed by Zachary Witt.
Inspired and based on the work of Yingfan Wang et al.,

## Disclaimer

This project is currently a work-in-progress. Use at your own risk.

This project is not affiliated with TensorFlow.js or any other TensorFlow project.

Please feel free to reach out if you have any questions or need further assistance.


