/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import {version} from '@tensorflow/tfjs-backend-wasm/dist/version';
import {IMAGE_SIZE,loadModel, predict} from './lib';

// tfjsWasm.setWasmPath(
//     `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
//         version}/dist/tfjs-backend-wasm.wasm`);

// UI

function showResults(imgElement, label) {
  const canvas = document.createElement('canvas');
  canvas.height=IMAGE_SIZE;
  canvas.width=IMAGE_SIZE;

  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  
  //@TODO: This is still not working
  // const overlayContainer = document.createElement('img');
  // const overlay = gradCamOverlay(imgElement,canvas);
  // imgContainer.appendChild(canvas);
  
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  const row = document.createElement('div');
  row.className = 'row';

  const classElement = document.createElement('div');
  classElement.className = 'cell';
  classElement.innerText = label;
  row.appendChild(classElement);
  probsContainer.appendChild(row);
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild
  );
}

const filesElement = document.getElementById('files');
if(filesElement){
  filesElement.addEventListener('change', evt => {
    let files = evt.target.files;
    // Display thumbnails & issue call to predict each image.
    for (let i = 0, f; f = files[i]; i++) {
      // Only process image files (skip non image files)
      if (!f.type.match('image.*')) {
        continue;
      }
      let reader = new FileReader();
      reader.onload = e => {
        // Fill the image & call predict.
        let img = document.createElement('img');
        img.src = e.target.result;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => predict(img).then((predicted) => {
          showResults(img, predicted);
        });
      };

      // Read in the image file as a data URL.
      reader.readAsDataURL(f);
    }
  });
}

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

status('Loading model...');

async function main() {
  loadModel().then(()=>{
    status('');

    // Make a prediction through the locally image.
    const imgElement = document.getElementById('img');
    if (imgElement.complete && imgElement.naturalHeight !== 0) {
      predict(imgElement).then((predicted) => {
        showResults(imgElement, predicted);
      });
      imgElement.style.display = '';
    } else {
      imgElement.onload = () => {
        predict(imgElement).then((predicted) => {
          showResults(imgElement, predicted);
        });
        imgElement.style.display = '';
      }
    }

    document.getElementById('file-container').style.display = '';
  });
}

tfjsWasm.setWasmPath(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@2.0.0/dist/tfjs-backend-wasm.wasm`);
tf.setBackend('wasm').then(() => main());