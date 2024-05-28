import React from 'react';
import saveCanvasImage from '../utils/saveImage';
import toggleCamera from '../utils/toggleCamera';

function Button_container(props) {
  const fileInputRef = React.useRef();
  return (
    <>
      <div id='btn-container'>
        <button id='openimg-btn' className='btn' disabled={props.buttonDisabled} onClick={() => fileInputRef.current.click()}>
          Open Image
          <input type="file" ref={fileInputRef} onChange={props.handleFileUpload} accept="image/jpeg, image/png" style={{display: 'none'}} />
        </button>
        <button id='saveimg-btn' className='btn' onClick={saveCanvasImage} disabled={props.saveButtonDisabled}>Save Image</button>
        <button id='opencam-btn' className='btn' onClick={() => toggleCamera(props.model, props.selectedModel)} disabled={props.buttonDisabled}>Open Webcam</button>
      </div>
      <button id='load-btn' className='btn' onClick={props.loadModel}>Load model</button>
    </>
  )
}
export default Button_container