camera = document.getElementById("webcam");
Webcam.set({
    width: 425,
    height: 240,
    image_format: "png",
    png_quality: "90"
});
Webcam.attach(camera);
function snapshot() {
    Webcam.snap(function (data_uri) {
        document.getElementById("result").innerHTML = "<img id='snapshot' src='" + data_uri + "'>"
    });
}
console.log("ml5.version:", ml5.version);
classifier = ml5.imageClassifier("https://teachablemachine.withgoogle.com/models/JBI9n-v9_/model.json", model_loaded);
function model_loaded() {
    console.log("Model successfull")
}
function check() {
img = document.getElementById("snapshot");
classifier.classify(img, gotResults);
}
function gotResults (errors, results) {
if (errors) {
console.error(errors);

}
else{
    console.log(results);
    document.getElementById("object_name").innerHTML=results[0].label;
    document.getElementById("accuracy_count").innerHTML=results[0].confidence.toFixed(3);
}
}
