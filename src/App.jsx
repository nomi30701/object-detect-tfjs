import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import drawBoundingBox from './utils/drawBoundingBox';
import saveCanvasImage from './utils/saveImage';
import toggleCamera from './utils/toggleCamera';
import { useRef, useState } from 'react'

function App() {
  const fileInputRef = useRef();
  const [saveButtonDisabled, setSaveButtonDisabled] = useState(true);
  const [model, setModel] = useState(null);
  const [info, setInfo] = useState('Please Load model.');
  const [infoColor, setInfoColor] = useState('black');
  const [buttonDisabled, setButtonDisabled] = useState(true);

  const loadModel = async () => {
    setInfo('Loading model...');
    setInfoColor('red');
    const loadedModel = await cocoSsd.load();
    setModel(loadedModel);
    setInfo('Model loaded.');
    setInfoColor('green');
    setButtonDisabled(false);
  }

  const handleFileUpload = async event => {
    const file = event.target.files[0];
    const imgel = document.getElementById('input-img');
    imgel.src = URL.createObjectURL(file);
    await imgel.decode();
    const predictions = await model.detect(imgel);
    drawBoundingBox(predictions, imgel, imgel.width, imgel.height);
    setSaveButtonDisabled(false);
  };

  return (
    <>
      <h2>📷Object Detection</h2>
      <p>model: <span style={{backgroundColor: 'black', color: 'lightgreen', borderRadius: '10px'}}>Coco-ssd</span></p>
      <canvas id='objdetect-canvas'></canvas>
      <img id='input-img' src="" hidden />
      <video id="input-camera" autoPlay hidden></video>
      <div id='btn-container'>
        <button id='openimg-btn' className='btn' onClick={() => fileInputRef.current.click()} disabled={buttonDisabled}>
          Open Image
          <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/jpeg, image/png" style={{display: 'none'}} />
        </button>
        <button id='saveimg-btn' className='btn' onClick={saveCanvasImage} disabled={saveButtonDisabled}>Save Image</button>
        <button id='opencam-btn' className='btn' onClick={() => toggleCamera(model)} disabled={buttonDisabled}>Open Webcam</button>
      </div>
      <button id='load-btn' className='btn' onClick={loadModel}>Load model</button>
      <p id='info' style={{color: infoColor}}>{info}</p>
    </>
  )
}

export default App