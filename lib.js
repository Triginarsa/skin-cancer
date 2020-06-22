import * as tf from '@tensorflow/tfjs';
import {LABELS} from './dict';

// load model
const MODEL_PATH = 'https://keen-kare-d88423.netlify.app/model.json';

// set image size to 224
export const IMAGE_SIZE = 224;

let model;
export const loadModel = async () => {
  // await tf.setBackend('wasm');
  console.log('loading model....');
  model = await tf.loadLayersModel(MODEL_PATH, {});

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  // Show model layers and summary
  model.summary();
};

export const clearModelMemory = () => {
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
};

/**
 * Given an image element, makes a prediction through model returning the
 * probabilities of the top classes.
 */
export async function predict(imgElement) {

    // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  //change image
  const logits = tf.tidy(() => {
    // Normalize the image from [0, 225] to [INPUT_MIN, INPUT_MAX]
    const normalizationConstant = 1.0 / 255.0;
    // resize image into 224 x 224
    let img = tf.browser.fromPixels(imgElement, 3)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE], false)
      .expandDims(0)
      .toFloat()
      .mul(normalizationConstant)

    startTime2 = performance.now();

    // Make a prediction through model
    return model.predict(img);
  });

  // Convert logits to probabilities and class names.
  const classes = await logits.data();
  // checking prediction each classes
  console.log('Predictions: ', classes);
  // Max probability value
  const labelName = indexOfMax(classes);
  console.log('Predictions: ', labelName);
  // sign probability value
  const probability = classes[labelName];
  // sign class name
  const className = LABELS[labelName];

  // count time
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log()
  console.log(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Return the best probability and name label
  return `Diagnosis : ${className} (${Math.floor(probability * 100)}%)`;
}


export function indexOfMax(arr) {
  if (arr.length === 0) {
      return -1;
  }

  var max = arr[0];
  var maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }

  return maxIndex;
}

export function resizeImage(imgElement){
  const canvas = document.createElement('canvas');
  canvas.height=IMAGE_SIZE;
  canvas.width=IMAGE_SIZE;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height); // clear canvas
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  const img = document.createElement('img');
  img.src = canvas.toDataURL("image/jpeg");;
  img.width = IMAGE_SIZE;
  img.height = IMAGE_SIZE;

  return img;
}

window.loadModel = loadModel;
window.clearModelMemory = clearModelMemory;
window.predict = predict;
window.resizeImage = resizeImage;