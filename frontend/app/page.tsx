"use client"

import React, { useState, useCallback } from "react"
import { Upload, Leaf, AlertCircle, CheckCircle, Loader2, Camera } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

interface PredictionResult {
  prediction: string
  confidence: number
  description: string
  prevention: string
  image_url: string
}

const BACKEND_URL = "https://lds-bw2e.onrender.com"

export default function LeafDiseaseClassifier() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleFileSelect = useCallback((file: File) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setError(null)
      setResult(null)
    } else {
      setError("Please select a valid image file")
    }
  }, [])

  // Drag & Drop handlers
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)
      const files = e.dataTransfer.files
      if (files && files.length > 0) {
        handleFileSelect(files[0])
      }
    },
    [handleFileSelect],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  // File input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0])
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select an image first")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch(`${BACKEND_URL}/`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Backend error response:", errorText)
        throw new Error("Failed to classify image")
      }

      const data: PredictionResult = await response.json()
      setResult(data)
    } catch (err) {
      setError("Failed to classify the image. Please make sure the backend is running and accessible.")
      console.error("Error:", err)
      setResult(null)
    } finally {
      setIsLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "bg-green-100 text-green-800 border-green-200"
    if (confidence >= 60) return "bg-yellow-100 text-yellow-800 border-yellow-200"
    return "bg-red-100 text-red-800 border-red-200"
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-green-600 rounded-full">
              <Leaf className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900">Leaf Disease Classifier</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload an image of a plant leaf to identify potential diseases and get prevention recommendations
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card className="h-fit relative">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Upload Leaf Image
              </CardTitle>
              <CardDescription>Drag and drop an image or tap to browse</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Upload Area with fixed mobile support */}
              <label
                htmlFor="file-input"
                className={`relative block border-2 border-dashed rounded-lg p-8 text-center cursor-pointer select-none
                  ${
                    isDragOver
                      ? "border-green-500 bg-green-50"
                      : "border-gray-300 hover:border-green-400 hover:bg-green-50"
                  }
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {previewUrl ? (
                  <>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full max-h-64 mx-auto rounded-lg shadow-md object-contain"
                    />
                    <p className="text-sm text-gray-600 mt-2">{selectedFile?.name}</p>
                  </>
                ) : (
                  <>
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium text-gray-700">Drop your leaf image here</p>
                    <p className="text-sm text-gray-500">or tap to browse files</p>
                  </>
                )}
                {/* Invisible input covers entire label for tap support */}
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleFileInputChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  aria-label="Upload leaf image"
                />
              </label>

              {/* Action Buttons */}
              <div className="flex gap-3">
                <Button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!selectedFile || isLoading}
                  className="flex-1"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Leaf className="h-4 w-4 mr-2" />
                      Classify Disease
                    </>
                  )}
                </Button>
                {selectedFile && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setSelectedFile(null)
                      setPreviewUrl(null)
                      setResult(null)
                      setError(null)
                    }}
                  >
                    Clear
                  </Button>
                )}
              </div>

              {/* Error Alert */}
              {error && (
                <Alert variant="destructive" className="mt-2">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          {result && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  Classification Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Prediction */}
                <div className="space-y-2">
                  <h3 className="font-semibold text-lg">Disease Identified</h3>
                  <div className="flex items-center gap-3">
                    <Badge variant="secondary" className="text-lg px-3 py-1">
                      {result.prediction}
                    </Badge>
                    <Badge className={`${getConfidenceColor(result.confidence)} border`}>
                      {result.confidence.toFixed(1)}% confidence
                    </Badge>
                  </div>
                </div>

                {/* Description */}
                <div className="space-y-2">
                  <h3 className="font-semibold">Description</h3>
                  <p className="text-gray-700 leading-relaxed">{result.description}</p>
                </div>

                {/* Prevention */}
                <div className="space-y-2">
                  <h3 className="font-semibold">Prevention</h3>
                  <p className="text-gray-700 leading-relaxed">{result.prevention}</p>
                </div>

                {/* Uploaded Image */}
                <div className="space-y-2">
                  <h3 className="font-semibold">Uploaded Image</h3>
                  <img
                    src={`${BACKEND_URL}${result.image_url}`}
                    alt="Uploaded leaf"
                    className="max-w-full max-h-64 rounded-lg shadow-md object-contain"
                    onError={(e) => {
                      // fallback if image URL is broken
                      (e.currentTarget as HTMLImageElement).src = previewUrl || ""
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
