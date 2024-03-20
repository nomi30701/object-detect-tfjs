import drawBoundingBox from './drawBoundingBox.js';

let isCameraOn = false;
let stream = null;
const toggleCamera = async (model) => {
    const video = document.getElementById('input-camera');
    const opencamera_btn = document.getElementById('opencam-btn');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.log('getUserMedia is not supported');
        return;
    }

    if (!isCameraOn) { // If the camera is not on
        try { // try to get the camera stream
            stream = await navigator.mediaDevices.getUserMedia({ // back camera
                video: { facingMode: 'environment' } 
            });
            video.srcObject = stream; // set the video source to the stream
            isCameraOn = true;
            opencamera_btn.textContent = 'Close Webcam';
        } catch (err) {
            console.log('An error occurred: ' + err);
        }

        // Wait for the video to start playing
        await new Promise((resolve) => video.onplaying = resolve);

        // Run prediction and draw bounding box on each frame
        const predictAndDraw = async () => {
            const predictions = await model.detect(video);
            drawBoundingBox(predictions, video, video.videoWidth, video.videoHeight);
            requestAnimationFrame(predictAndDraw);
        };
        predictAndDraw();
        
    } else { // If the camera is on
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        opencamera_btn.textContent = 'Open Webcam';
        video.srcObject = null;
        isCameraOn = false;
    }
}
export default toggleCamera;