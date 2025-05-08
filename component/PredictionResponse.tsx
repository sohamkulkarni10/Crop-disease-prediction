"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Loader2 } from "lucide-react";

interface PredictionResponse {
  prediction: string;
  image_url: string;
}

interface ApiError {
  error: string;
}

const PredictComponent: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [predictionResult, setPredictionResult] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
      setPredictionResult(null);
      setImageUrl(null);
      setError(null);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPredictionResult(null);
    setImageUrl(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data: PredictionResponse | ApiError = await response.json();

      if (!response.ok) {
        const errorData = data as ApiError;
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const successData = data as PredictionResponse;
      setPredictionResult(successData.prediction);
      setImageUrl(successData.image_url);
    } catch (err) {
      console.error("Prediction API error:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-[url('/agribg.jpg')] bg-cover bg-center bg-fixed flex items-center justify-center px-4 py-12"
    >
      <div className="w-full max-w-3xl backdrop-blur-sm bg-black/30 rounded-2xl shadow-2xl border border-green-900 p-6 space-y-6">
        <Card className="bg-transparent border-none">
          <CardHeader>
            <CardTitle className="text-4xl font-bold text-green-300 text-center drop-shadow-lg">
              🌾 Crop Disease Detector
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="fileInput" className="text-white font-semibold">
                  Upload a plant image for prediction
                </Label>
                <Input
                  type="file"
                  id="fileInput"
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={isLoading}
                  className="bg-gray-400 border border-green-600 text-white placeholder-gray-400"
                />
              </div>

              {previewUrl && (
                <div className="mt-4">
                  <p className="text-sm text-green-300 mb-2">Preview:</p>
                  <div className="rounded-lg border border-green-700 bg-gray-900 p-2">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-h-64 mx-auto rounded-lg shadow-md"
                    />
                  </div>
                </div>
              )}

              <Button
                type="submit"
                disabled={!selectedFile || isLoading}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 z-100 border-white border-2"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  "Predict Disease"
                )}
              </Button>
            </form>

            {error && (
              <Alert variant="destructive" className="bg-red-800 text-white border-red-700">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {predictionResult && imageUrl && (
              <Card className="bg-green-950/30 border border-green-700 text-white rounded-xl">
                <CardHeader>
                  <CardTitle className="text-2xl">Prediction Result</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-green-800 p-3 rounded-md">
                    <p className="text-lg font-bold text-white">{predictionResult}</p>
                  </div>
                  <div>
                    <p className="text-sm text-green-300 mb-2">Processed Image:</p>
                    <img
                      src={imageUrl}
                      alt={`Predicted ${predictionResult}`}
                      className="max-h-64 mx-auto rounded-md border border-green-700"
                    />
                  </div>
                </CardContent>
              </Card>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PredictComponent;
