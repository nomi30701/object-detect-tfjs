import * as tf from '@tensorflow/tfjs';
const classnames = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
};

async function yolo_process(model, imgel) {
    const img = tf.browser.fromPixels(imgel);
    const resized = tf.image.resizeBilinear(img, [640, 640]);
    const expanded = resized.expandDims(0);
    const normalized = expanded.div(255);
    const output = model.predict(normalized);
    // [1, 84, 8400] 84 indices, 8400 values for each index
    // first 4 values are xc, yc, w, h, the rest are probabilities of each class.

    const result = decoder(output, imgel.width/640, imgel.height/640);
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
        const boxes = reshaped.slice([0, 0], [-1, 4]);  // every row, first 4 elements (xc,yc,w,h)
        const scores = reshaped.slice([0, 4], [-1, -1]); // except first 4 elements

        const x_center = boxes.slice([0, 0], [-1, 1]);
        const y_center = boxes.slice([0, 1], [-1, 1]);
        const width = boxes.slice([0, 2], [-1, 1]);
        const height = boxes.slice([0, 3], [-1, 1]);
        
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
            class: classnames[nmsClasses[index]],
            score: nmsScores[index]
        };
    });

    // Dispose the tensors
    convertedBoxes_yxyx.dispose();
    scores_max.dispose();
    classes.dispose();
    nmsIndices.dispose();

    return result;
}
export default yolo_process;