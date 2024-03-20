function drawBoundingBox(predictions, src, width, height) {
    const obj_canvas = document.getElementById('objdetect-canvas');
    const ctx = obj_canvas.getContext('2d');

    obj_canvas.width = width;
    obj_canvas.height = height;
    ctx.drawImage(src, 0, 0);

    predictions.forEach(predict => {
        ctx.beginPath();
        ctx.rect(
            predict.bbox[0],
            predict.bbox[1],
            predict.bbox[2],
            predict.bbox[3]
        );
        ctx.lineWidth = 2;
        ctx.strokeStyle = `rgb(255, 0, 255)`;
        ctx.stroke();
        
        // Draw text and background
        ctx.fillStyle = `rgb(255, 0, 255)`;
        ctx.font = '16px Arial';
        const text = `${predict.class} | ${Math.floor((predict.score * 100))}%`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = parseInt(ctx.font, 10);
        ctx.fillRect(predict.bbox[0] - 1, predict.bbox[1] - textHeight - 4, textWidth + 4, textHeight + 4);
        ctx.fillStyle = 'white';
        ctx.fillText(text, predict.bbox[0], predict.bbox[1] - 5);
    });
}
export default drawBoundingBox;