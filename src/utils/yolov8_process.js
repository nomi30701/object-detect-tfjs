import * as tf from '@tensorflow/tfjs';
import labels from './labels.json';

async function yolo_process(model, imgWidth, imgHeight, imgel) {
    const img = tf.browser.fromPixels(imgel);
    const resized = tf.image.resizeBilinear(img, [640, 640]);
    const expanded = resized.expandDims(0);
    const normalized = expanded.div(255);
    const output = model.execute(normalized);
    // [1, 84, 8400] 84 indices, 8400 values for each index
    // first 4 values are xc, yc, w, h, the rest are probabilities of each class.

    const result = decoder(output, imgWidth/640, imgHeight/640);
    tf.dispose([img, resized, expanded, normalized, output]);
    console.log(labels[0]);
    return result;
}

async function decoder(yoloOutput, imgWidth, imgHeight) {
    let [convertedBoxes_yxyx, scores_max, classes] = tf.tidy(() => {

        // YOLOv8 output shape is [1, 84, 8400]
        const [_, values, numDetections] = yoloOutput.shape;

        // Reshape the output to [8400, 84]
        // 8400 predictions boxes, each box has 84 values
        // first 4 values are xc, yc, w, h, the rest are probabilities of each class.
        const reshaped = yoloOutput.transpose([2, 1, 0]).reshape([numDetections, values]);

        // Split the second dimension into boxes and scores
        // every row, first 4 elements (xc,yc,w,h)
        // except first 4 elements
        const [boxes, scores] = tf.split(reshaped, [4, values - 4], 1);

        const [x_center, y_center, width, height] = tf.split(boxes, 4, 1);
        
        // Convert boxes from [x_center, y_center, width, height] to [y1, x1, y2, x2]
        const width_half = width.div(2);
        const height_half = height.div(2);
        const y1 = y_center.sub(height_half);
        const x1 = x_center.sub(width_half);
        const y2 = y_center.add(height_half);
        const x2 = x_center.add(width_half);
        const convertedBoxes_yxyx = tf.concat([y1, x1, y2, x2], 1);

        // maxium score for each box
        const scores_max = scores.max(1); 
        const classes = scores.argMax(1); // maxium score of index for each box
        return [convertedBoxes_yxyx, scores_max, classes];
    });

    // Apply nonMaxSuppressionAsync to get rid of overlapping boxes
    // boxes, scores, maxOutputSize, iouThreshold, scoreThreshold
    const nmsIndices = await tf.image.nonMaxSuppressionAsync(
        convertedBoxes_yxyx, scores_max, 100, 0.4, 0.5);

    // Gather the boxes, scores, and classes based on nmsIndices
    //// .gather(tensor, index)
    //// get the values of tensor at the index
    const nmsBoxes_yxyx = tf.gather(convertedBoxes_yxyx, nmsIndices).arraySync();
    const nmsScores = tf.gather(scores_max, nmsIndices).arraySync();
    const nmsClasses = tf.gather(classes, nmsIndices).arraySync();

    // Convert the data to objects
    const result = nmsBoxes_yxyx.map((bbox, index) => {
        return {
            bbox: [
                bbox[1] * imgWidth,    // x1
                bbox[0] * imgHeight,   // y1
                (bbox[3] - bbox[1]) * imgWidth,  // w
                (bbox[2] - bbox[0]) * imgHeight,  // h
            ],
            class: labels[nmsClasses[index]],
            score: nmsScores[index]
        };
    });

    // Dispose the tensors
    tf.dispose([convertedBoxes_yxyx, scores_max, classes, nmsIndices]);
    
    return result;
}
export default yolo_process;