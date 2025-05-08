"use client";

import React, { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Loader2, Calendar, Clock } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

interface HistoryItem {
  prediction_id: string;
  disease_name: string;
  image_url: string;
  timestamp: string;
}

const HistoryComponent: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/history");
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Failed to fetch history.");
        }

        setHistory(data.history);
      } catch (err) {
        console.error("Error fetching history:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

 

  return (

    <div className="min-h-screen bg-gradient-to-b from-green-900 to-emerald-800 bg-fixed relative p-8 overflow-x-hidden">
      {/* Decorative elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden opacity-20 pointer-events-none">
        <div className="absolute -top-20 -left-20 w-96 h-96 bg-yellow-300 rounded-full blur-3xl"></div>
        <div className="absolute top-1/3 -right-20 w-80 h-80 bg-green-300 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 left-1/4 w-96 h-96 bg-blue-300 rounded-full blur-3xl"></div>
      </div>

      <div className="absolute inset-0 backdrop-blur-sm z-0" />

    {/* {header} */}
    <header className="relative z-10 pt-6 px-4 sm:px-8 md:px-12">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <nav className="hidden md:block">
          <ul className="flex gap-8">
              <li><a className="text-green-100 hover:text-white font-medium transition-colors" href="/">Home</a></li>
              <li><a className="text-green-100 hover:text-white font-medium transition-colors" href="/chatbot">AI Chat</a></li>
          </ul>
        </nav>
      </div>
    </header>

      <div className="relative z-10 max-w-6xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="text-5xl font-extrabold text-green-100 mb-4 drop-shadow-lg animate-fade-in">
            🌱 Prediction History
          </h1>
          <p className="text-green-200 text-lg max-w-2xl mx-auto animate-fade-in-slow">
            Review your previous leaf disease predictions and track your plant health over time
          </p>
        </div>

        {loading ? (
          <div className="flex flex-col justify-center items-center mt-20 gap-4">
            <Loader2 className="animate-spin w-16 h-16 text-green-300" />
            <p className="text-green-200 font-medium">Loading your prediction history...</p>
          </div>
        ) : error ? (
          <Alert variant="destructive" className="bg-red-800/80 text-white border border-red-500 shadow-xl backdrop-blur-sm max-w-xl mx-auto">
            <AlertTitle className="font-bold text-lg">Error Loading History</AlertTitle>
            <AlertDescription className="text-red-100">{error}</AlertDescription>
          </Alert>
        ) : history.length === 0 ? (
          <div className="bg-green-800/40 backdrop-blur-sm rounded-xl border border-green-400/30 p-8 text-center mt-10 max-w-md mx-auto shadow-lg animate-fade-in-slow">
            <div className="mb-4 inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-700/50 text-green-200">
              <Calendar className="w-8 h-8" />
            </div>
            <p className="text-green-100 text-lg font-medium">No predictions yet.</p>
            <p className="text-green-300 mt-2">Upload a leaf image to start building your history.</p>
          </div>
        ) : (
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 animate-fade-in-slow">
            {history.map((item) => (
              <Card
                key={item.prediction_id}
                className="group relative bg-gradient-to-br from-green-800/80 to-emerald-900/80 border border-green-400/30 shadow-lg rounded-2xl overflow-hidden hover:scale-103 transition-all duration-300 hover:shadow-[0_0_25px_rgba(74,222,128,0.3)] backdrop-blur-sm"
              >
                <div className="absolute -top-8 -left-8 w-10 h-10 bg-green-400 opacity-20 rounded-full blur-2xl group-hover:bg-green-300 group-hover:opacity-30 transition-all duration-500" />
                
                <CardHeader className="p-5 z-10 relative">
                  <CardTitle className="text-xl text-green-100 font-bold flex items-center gap-2">
                    <span className="bg-green-600/30 rounded-full p-1">
                      <span className="block w-3 h-3 rounded-full bg-green-400"></span>
                    </span>
                    {item.disease_name}
                  </CardTitle>
                </CardHeader>
                
                <CardContent className="p-5 space-y-5 \ z-10">
                  <div className="rounded-xl overflow-hidden border border-green-500/30 shadow-inner">
                    <img
                      src={item.image_url}
                      alt={item.disease_name}
                      className="h-40 w-full transition duration-500 group-hover:scale-105 group-hover:brightness-110"
                    />
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm text-green-200 font-medium bg-green-800/50 p-3 rounded-lg border border-green-500/20">
                    <Clock className="w-4 h-4 text-green-300" />
                    <span className="font-semibold">Updated at:</span>{" "}
                    <span className="text-green-100">{formatDate(item.timestamp)}</span>
                  </div>
                </CardContent>
                
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-green-500 to-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </Card>
            ))}
          </div>
        )}
      </div>

      <style jsx>{`
        .animate-fade-in {
          animation: fadeIn 0.8s ease-out forwards;
        }
        .animate-fade-in-slow {
          animation: fadeIn 1.2s ease-out forwards;
        }
        @keyframes fadeIn {
          0% {
            opacity: 0;
            transform: translateY(15px);
          }
          100% {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default HistoryComponent;