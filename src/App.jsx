import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import Info_container from './Components/Info_container';
import Canvas_continer from './Components/Canvas_container';
import Button_container from './Components/Button_container';
import yolo_process from './utils/yolov8_process';
import drawBoundingBox from './utils/drawBoundingBox';

import { useRef, useState, useCallback } from 'react'

// TODO: new branch for <canvas> be a mask in <image> and <video>. if good for video fps.
function App() {
  const [selectedModel, setSelectedModel] = useState('cocoSsd');
  const [buttonDisabled, setButtonDisabled] = useState(true);
  const [saveButtonDisabled, setSaveButtonDisabled] = useState(true);
  const [model, setModel] = useState(null);
  const [info, setInfo] = useState('Please Load model.');
  const [infoColor, setInfoColor] = useState('black');

  const loadModel = useCallback(async () => {
    setInfo('Loading model...');
    setInfoColor('red');
    let loadedModel;
    switch (selectedModel) { 
      case 'cocoSsd':
        loadedModel = await cocoSsd.load();
        break;
      case 'yolov8n':
        loadedModel = await tf.loadGraphModel(
          `${window.location.href}/yolov8n_web_model/model.json`
        );
        loadedModel.execute(tf.ones(loadedModel.inputs[0].shape));
        break;
    }
    setModel(loadedModel);
    setInfo('Model loaded.');
    setInfoColor('green');
    setButtonDisabled(false);
  }, [selectedModel]);

  const handleFileUpload = useCallback(async event => {
    const file = event.target.files[0];
    const imgel = document.getElementById('input-img');
    imgel.src = URL.createObjectURL(file);
    await imgel.decode();

    let predictions; 
    switch (selectedModel) {
      case 'cocoSsd':
        predictions = await model.detect(imgel);
        break;
      case 'yolov8n':
        predictions = await yolo_process(model, imgel.width, imgel.height, imgel);
        break;
    }
    drawBoundingBox(predictions, imgel, imgel.width, imgel.height);
    setSaveButtonDisabled(false);
  }, [model, selectedModel]);
  
  return (
    <>
      <h2>📷Object Detection</h2>
      <Info_container selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      <Canvas_continer />
      <Button_container loadModel={loadModel} buttonDisabled={buttonDisabled} 
                        handleFileUpload={handleFileUpload} saveButtonDisabled={saveButtonDisabled} 
                        model={model} selectedModel={selectedModel} />
      <p id='info' style={{color: infoColor}}>{info}</p>
    </>
  )
}

export default App