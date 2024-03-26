import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import yolo_process from './utils/yolov8_process';
import drawBoundingBox from './utils/drawBoundingBox';
import saveCanvasImage from './utils/saveImage';
import toggleCamera from './utils/toggleCamera';
import { useRef, useState } from 'react'

function App() {
  const fileInputRef = useRef();
  const [saveButtonDisabled, setSaveButtonDisabled] = useState(true);
  const [buttonDisabled, setButtonDisabled] = useState(true);
  const [model, setModel] = useState(null);
  const [info, setInfo] = useState('Please Load model.');
  const [infoColor, setInfoColor] = useState('black');
  const [selectedModel, setSelectedModel] = useState('cocoSsd');

  // load model
  const loadModel = async () => {
    setInfo('Loading model...');
    setInfoColor('red');
    let loadedModel;
    switch (selectedModel) { 
      case 'cocoSsd':
        loadedModel = await cocoSsd.load();
        break;
      case 'yolov8n':
        loadedModel = await tf.loadGraphModel('../public/yolov8n_web_model/model.json');
        break;
    }
    setModel(loadedModel);
    setInfo('Model loaded.');
    setInfoColor('green');
    setButtonDisabled(false);
  }

  // handle file upload
  const handleFileUpload = async event => {
    const file = event.target.files[0];
    const imgel = document.getElementById('input-img');
    imgel.src = URL.createObjectURL(file);
    await imgel.decode();

    let predictions; 
    switch (selectedModel) {
      case 'cocoSsd':
        predictions = await model.detect(imgel);
        drawBoundingBox(predictions, imgel, imgel.width, imgel.height);
        setSaveButtonDisabled(false);
        break;
      case 'yolov8n': {
        predictions = await yolo_process(model, imgel);
        drawBoundingBox(predictions, imgel, imgel.width, imgel.height);
        setSaveButtonDisabled(false);
        break;
      }
    }
    console.log(predictions);
  };

  return (
    <>
      <h2>📷Object Detection</h2>
      <div id='info-container'>
        <p>
          model: <span id='model-info'>{selectedModel}</span>
        </p>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
          <option value="cocoSsd">Coco-ssd</option>
          <option value="yolov8n">yolov8n</option>
        </select>
      </div>
      <canvas id='objdetect-canvas'></canvas>
      <img id='input-img' src="" hidden />
      <video id="input-camera" autoPlay hidden></video>
      <div id='btn-container'>
        <button id='openimg-btn' className='btn' onClick={() => fileInputRef.current.click()} disabled={buttonDisabled}>
          Open Image
          <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/jpeg, image/png" style={{display: 'none'}} />
        </button>
        <button id='saveimg-btn' className='btn' onClick={saveCanvasImage} disabled={saveButtonDisabled}>Save Image</button>
        <button id='opencam-btn' className='btn' onClick={() => toggleCamera(model, selectedModel)} disabled={buttonDisabled}>Open Webcam</button>
      </div>
      <button id='load-btn' className='btn' onClick={loadModel}>Load model</button>
      <p id='info' style={{color: infoColor}}>{info}</p>
    </>
  )
}

export default App