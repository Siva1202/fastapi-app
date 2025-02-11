import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function AnomalyDetector() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [downloadLink, setDownloadLink] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const BACKEND_URL = "http://localhost:8000"; // Change this to your deployed backend URL

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError("");
    try {
      const response = await axios.post(`${BACKEND_URL}/predict/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data.predictions) {
        setPredictions(response.data.predictions);
        setDownloadLink(`${BACKEND_URL}/download/`);
      } else {
        setError("Unexpected response from the server.");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setError("Failed to upload or process the file. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="p-4 w-full max-w-2xl mx-auto">
      <CardContent>
        <h2 className="text-xl font-bold mb-4">Anomaly Detector</h2>
        <Input type="file" onChange={handleFileChange} className="mb-4" />
        <Button onClick={handleUpload} disabled={loading}>
          {loading ? "Processing..." : "Upload & Predict"}
        </Button>

        {error && <p className="text-red-500 mt-2">{error}</p>}

        {predictions.length > 0 && (
          <div className="mt-6">
            <h3 className="font-bold">Predictions</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Transaction Amount</TableHead>
                  <TableHead>Prediction</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {predictions.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{index + 1}</TableCell>
                    <TableCell>{item.Transaction_Amount}</TableCell>
                    <TableCell>{item.Anomaly_Prediction === 1 ? "Anomalous" : "Normal"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {downloadLink && (
              <a href={downloadLink} download className="mt-4 block text-blue-500 underline">
                Download Results
              </a>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
