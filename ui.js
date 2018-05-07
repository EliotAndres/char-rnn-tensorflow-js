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

export function status(statusText) {
  console.log(statusText);
  document.getElementById('status').innerHTML = statusText.replace(/(\r\n|\n|\r)/g,"<br/>");
}

export function showMetadata(sentimentMetadataJSON) {
  document.getElementById('modelType').textContent =
      sentimentMetadataJSON['model_type'];
  document.getElementById('vocabularySize').textContent =
      sentimentMetadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      sentimentMetadataJSON['max_len'];
}

function doPredict(predict) {
  const reviewText = document.getElementById('review-text');
  const result = predict(reviewText.value);
  status(
      'Inference result (0 - negative; 1 - positive): ' + result.score +
      ' (elapsed: ' + result.elapsed + ' ms)');
}

export function setReviewText(text, predict) {
  const reviewText = document.getElementById('review-text');
  reviewText.value = text;
}

function setPredictFunction(predict) {
  const reviewText = document.getElementById('review-text');
  reviewText.addEventListener('input', () => doPredict(predict));
}

export function disableLoadModelButtons() {
  document.getElementById('load-pretrained-remote').style.display = 'none';
  document.getElementById('load-pretrained-local').style.display = 'none';
}
