let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const maxPredictions = 5; // Adjust based on your model

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                webcamElement.srcObject = stream;
                webcamElement.addEventListener('loadeddata', () => resolve(), false);
            })
            .catch(reject);
    });
}

async function app() {
    console.log('Loading model...');
    // Load the model.
    net = await tmImage.load('https://teachablemachine.withgoogle.com/models/juR7Nsj4E/model.json', 'https://teachablemachine.withgoogle.com/models/juR7Nsj4E/metadata.json');
    // Load the model's classes.
    const model = await net.load();
    await setupWebcam();

    // Reads an image from the webcam and associates it with a specific class index.
    const addExample = classId => {
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');
        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);
    };

    // When clicking the button, add an example for that class.
    document.getElementById('start').addEventListener('click', async () => {
        while (true) {
            if (classifier.getNumClasses() > 0) {
                const activation = net.infer(webcamElement, 'conv_preds');
                const result = await classifier.predictClass(activation);
                const classes = ['Cardboard', 'Glass', 'Plastic', 'Metal']; // Update based on your classes
                document.getElementById('label-container').innerText = `
                  Prediction: ${classes[result.label]}\n
                  Probability: ${result.confidences[result.label]}
                `;
            }
            await tf.nextFrame();
        }
    });
}

app();
