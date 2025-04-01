import React, { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function KnifeDetector() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const imgRef = useRef(null);

  // Load the trained model from Teachable Machine
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("Loading model...");
        const loadedModel = await tf.loadLayersModel(
          "https://teachablemachine.withgoogle.com/models/1YM6sDQcC/model.json"
        );
        setModel(loadedModel);
        console.log("Model loaded successfully!");
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };
    loadModel();
  }, []);

  const classifyImage = async () => {
    if (!model || !imgRef.current) return;
    setLoading(true);

    // Convert image to tensor
    const img = tf.browser.fromPixels(imgRef.current)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();

    const predictions = await model.predict(img).data();
    setLoading(false);

    console.log("Predictions:", predictions); // Debugging output

    // Get the highest probability class
    const knifeProbability = predictions[0]; // First class is "Knife"
    setPrediction(knifeProbability > 0.5 ? "It's a Knife" : "It's not a Knife");
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        imgRef.current.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4 p-4">
      <input type="file" accept="image/*" onChange={handleImageUpload} className="mb-2" />
      <img ref={imgRef} alt="Uploaded" className="max-w-xs max-h-60 rounded-lg shadow-lg" />
      <button onClick={classifyImage} disabled={!model || loading}>
        {loading ? "Detecting..." : "Detect Knife"}
      </button>
      {prediction && <p className="text-lg font-bold">{prediction}</p>}
    </div>
  );
}
