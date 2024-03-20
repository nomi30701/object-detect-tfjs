function saveCanvasImage() {
    const obj_canvas = document.getElementById('objdetect-canvas');
    const dataUrl = obj_canvas.toDataURL("image/png");
    
    let link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'canvas_image.png';
    link.click();
}
export default saveCanvasImage;