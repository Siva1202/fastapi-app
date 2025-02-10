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

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPredictions(response.data.predictions); // FIXED: Now using correct response structure
      setDownloadLink("http://localhost:8000/download");
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <Card className="p-4 w-full max-w-2xl mx-auto">
      <CardContent>
        <h2 className="text-xl font-bold mb-4">Anomaly Detector</h2>
        <Input type="file" onChange={handleFileChange} className="mb-4" />
        <Button onClick={handleUpload}>Upload & Predict</Button>
        {predictions.length > 0 && (
          <div className="mt-6">
            <h3 className="font-bold">Predictions</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Prediction</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {predictions.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{index + 1}</TableCell>
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
