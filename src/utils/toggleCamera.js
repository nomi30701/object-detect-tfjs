import drawBoundingBox from './drawBoundingBox.js';
import yolo_process from './yolov8_process.js';

let isCameraOn = false;
let stream = null;
let animationId = null;
let video;
let opencamera_btn;
let canvas;
let context;

window.onload = function() {
    video = document.getElementById('input-camera');
    opencamera_btn = document.getElementById('opencam-btn');
    canvas = document.getElementById('objdetect-canvas');
    context = canvas.getContext('2d');
}
const modelFunctions = {
    'cocoSsd': (model) => model.detect.bind(model),
    'yolov8n': (model, videoWidth, videoHeight) => yolo_process.bind(null, model, videoWidth, videoHeight)
};

const toggleCamera = async (model, model_name) => {
    // Check camera support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.log('getUserMedia is not supported');
        return;
    }

    try {
        if (!isCameraOn) { // If the camera is not on
            stream = await navigator.mediaDevices.getUserMedia({ // rear camera
                video: { facingMode: 'environment' } 
            });
            video.srcObject = stream; // set the video source to the stream
            isCameraOn = true;
            opencamera_btn.textContent = 'Close Webcam';

            // Wait for the video to start playing
            await new Promise((resolve) => video.onplaying = resolve);

            // Run prediction and draw bounding box on each frame
            let predictFunction = modelFunctions[model_name](model, video.videoWidth, video.videoHeight);
            const predictAndDraw = async () => {
                const predictions = await predictFunction(video);
                drawBoundingBox(predictions, video, video.videoWidth, video.videoHeight);
                animationId = requestAnimationFrame(predictAndDraw);
            };
            predictAndDraw();
            
        } else { // If the camera is on
            cancelAnimationFrame(animationId); // cencel the animation frame
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            opencamera_btn.textContent = 'Open Webcam';
            video.srcObject = null;
            isCameraOn = false;

            // Clear the canvas
            context.clearRect(0, 0, canvas.width, canvas.height);
        }
    } catch (err) {
        console.log('An error occurred: ' + err);
    }
}
export default toggleCamera;