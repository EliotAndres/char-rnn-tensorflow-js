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
import * as loader from './loader';
import * as ui from './ui';
import SJS  from 'discrete-sampling'

const LOCAL_URLS = {
    model: './resources/model.json',
    metadata: './resources/metadata.json'
};

class SentimentPredictor {
    /**
     * Initializes the Sentiment demo.
     */
    async init(urls) {
        this.urls = urls;
        this.model = await loader.loadHostedPretrainedModel(urls.model);
        this.maxLen = 10;
        this.charIndices = {'\n': 0,' ': 1,'!': 2,'"': 3,'#': 4,'$': 5,'%': 6,'&': 7,"'": 8,'(': 9,')': 10,'*': 11,'+': 12,',': 13,'-': 14,'.': 15,'/': 16,'0': 17,'1': 18,'2': 19,'3': 20,'4': 21,'5': 22,'6': 23,'7': 24,'8': 25,'9': 26,':': 27,'=': 28,'>': 29,'?': 30,'@': 31,'A': 32,'B': 33,'C': 34,'D': 35,'E': 36,'F': 37,'G': 38,'H': 39,'I': 40,'J': 41,'K': 42,'L': 43,'M': 44,'N': 45,'O': 46,'P': 47,'Q': 48,'R': 49,'S': 50,'T': 51,'U': 52,'V': 53,'W': 54,'X': 55,'Y': 56,'Z': 57,'[': 58,'\\': 59,']': 60,'^': 61,'_': 62,'a': 63,'b': 64,'c': 65,'d': 66,'e': 67,'f': 68,'g': 69,'h': 70,'i': 71,'j': 72,'k': 73,'l': 74,'m': 75,'n': 76,'o': 77,'p': 78,'q': 79,'r': 80,'s': 81,'t': 82,'u': 83,'v': 84,'w': 85,'x': 86,'y': 87,'z': 88,'{': 89,'|': 90,'}': 91,'~': 92,'\x81': 93,'\x82': 94,'\x83': 95,'\x84': 96,'\x85': 97,'\x86': 98,'\x87': 99,'\x88': 100,'\x89': 101,'\x8a': 102,'\x8b': 103,'\x8c': 104,'\x8d': 105,'\x8e': 106,'\x8f': 107,'\x90': 108,'\x91': 109,'\x92': 110,'\x93': 111,'\x94': 112,'\x95': 113,'\x96': 114,'\x97': 115,'\x98': 116,'\x99': 117,'\x9a': 118,'\x9b': 119,'\x9c': 120,'\x9f': 121,'¡': 122,'¥': 123,'§': 124,'¨': 125,'©': 126,'ª': 127,'«': 128,'¯': 129,'µ': 130,'¾': 131,'¿': 132,'Ê': 133,'Ë': 134,'Ì': 135,'Ï': 136,'Ð': 137,'Ñ': 138,'Ô': 139,'Õ': 140,'ç': 141,'è': 142,'ê': 143,'ì': 144,'î': 145,'ï': 146,'ò': 147,'õ': 148,'û': 149};
        this.vocabularySize = 150;

        this.indiceChars = this.swap(this.charIndices)
        return this;
    }

    swap(json){
        var ret = {};
        for(var key in json){
            ret[json[key]] = key;
        }
        return ret;
    }

    argMax(array) {
        return Array.from(array).map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }

    sample(preds, temperature=1.0) {
        const predsTemperature = preds.map((prob) => Math.exp(Math.log(prob) / temperature) )
        const mult = SJS.Multinomial(1, predsTemperature);
        return this.argMax(mult.draw());
    }

    generateText() {
        let text = '';
        for (let i = 0; i < this.maxLen; ++i) {
            const index = Math.floor(Math.random() * this.vocabularySize) + 1
            text += this.indiceChars[index]
        }
        console.log('generated', text)
        return text
    }

    textToTensor(text) {
        const inputText = text.toLowerCase();
        // Look up word indices.
        const inputBuffer = tf.buffer([1, this.maxLen, 150], 'float32');
        for (let i = 0; i < inputText.length; ++i) {
            const char = inputText[i];
            inputBuffer.set(1, 0, i, this.charIndices[char]);
        }
        return inputBuffer.toTensor();
    }

    predict(text) {

        ui.status('Running inference');
        const beginMs = performance.now();

        let sentence = text;
        let generated = ''
        for (let i = 0; i < 100; ++i) {
            let input = this.textToTensor(sentence)

            const predictOut = this.model.predict(input);
            const preds = predictOut.dataSync();
            const nextIndex = this.sample(preds, 0.5);
            const nextChar = this.indiceChars[nextIndex];
            sentence = sentence.slice(1) + nextChar;
            generated += nextChar;
        }

        const endMs = performance.now();
        console.log(generated)
        return {elapsed: (endMs - beginMs), generated};
    }
};


/**
 * Loads the pretrained model and metadata, and registers the predict
 * function with the UI.
 */
async function setupSentiment() {

    if (await loader.urlExists(LOCAL_URLS.model)) {
        ui.status('Model available: ' + LOCAL_URLS.model);
        const button = document.getElementById('load-pretrained-local');
        const generateButton = document.getElementById('generate');

        let predictor = null
        button.addEventListener('click', async () => {
            predictor = await new SentimentPredictor().init(LOCAL_URLS);

            const seed = predictor.generateText();
            const result = predictor.predict(seed)
            ui.status('Generated in ' + (result.elapsed/1000).toFixed(2) + ' Result: ' + result.generated)
            generateButton.style.display = 'inline-block';
        });

        generateButton.addEventListener('click', async () => {
            const seed = predictor.generateText();
            const result = predictor.predict(seed)
            ui.status('Generated in ' + (result.elapsed/1000).toFixed(2) + ' Result: ' + result.generated)
        });
        button.style.display = 'inline-block';

    }

    ui.status('Standing by.');
}

setupSentiment();
