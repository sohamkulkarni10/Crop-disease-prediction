"use client"
import PredictComponent from "@/component/PredictionResponse";
import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background image - maintaining the original background */}
      <div className="fixed inset-0 z-0">
        <Image
          src="/history1.jpg"
          alt="Background"
          fill
          className="object-cover"
          priority
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-green-900/40 backdrop-blur-[2px]" />
      </div>

      {/* Decorative elements (animations removed) */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-64 h-64 bg-green-200 opacity-10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-yellow-100 opacity-10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-20 w-80 h-80 bg-blue-100 opacity-10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 pt-6 px-4 sm:px-8 md:px-12">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-white">FarmAI Assistant</h1>
          </div>

          <nav className="hidden md:block">
            <ul className="flex gap-8">
              <li><a className="text-green-100 hover:text-white font-medium transition-colors" href="/">Home</a></li>
              <li><a className="text-green-100 hover:text-white font-medium transition-colors" href="/history">History</a></li>
              <li><a className="text-green-100 hover:text-white font-medium transition-colors" href="/chatbot">AI Chat</a></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 px-4 sm:px-8 md:px-12 pt-8 pb-16">
        <div className="max-w-6xl mx-auto">
          {/* Removed animate-fade-in */}
          <div className="mb-12 text-center">
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white mb-4 tracking-tight drop-shadow-md">
              <span className="inline-block bg-gradient-to-r from-green-200 to-emerald-100 text-transparent bg-clip-text">Leaf Disease</span> Detection
            </h2>
            <p className="text-lg text-green-100 max-w-2xl mx-auto">
              Upload a photo of your plant leaf and get instant disease identification with our advanced AI technology
            </p>
          </div>

          <div className="lg:flex lg:items-start lg:gap-8">
            {/* Left column with instructions (Removed animate-fade-in and style) */}
            <div className="w-full lg:w-1/3 mb-8 lg:mb-0">
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-xl">
                <h3 className="text-xl font-bold text-green-100 mb-4 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  How It Works
                </h3>
                <ol className="space-y-4 text-green-100">
                  <li className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center font-bold">1</span>
                    <span>Upload a clear photo of the affected plant leaf</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center font-bold">2</span>
                    <span>Our AI analyzes the image to identify diseases</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center font-bold">3</span>
                    <span>Get instant results with treatment recommendations</span>
                  </li>
                </ol>

                <div className="mt-8 pt-6 border-t border-white/10">
                  <h4 className="text-lg font-semibold text-green-100 mb-3">For Best Results:</h4>
                  <ul className="space-y-2 text-green-200 text-sm">
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Use good lighting conditions
                    </li>
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Focus on the affected area
                    </li>
                    <li className="flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Avoid shadows and reflections
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Right column with prediction component (Removed animate-fade-in and style) */}
            <div className="w-full lg:w-2/3">
              <div className="bg-white/5 backdrop-blur-md rounded-2xl overflow-hidden border border-white/10 shadow-2xl">
                <div className="p-1">
                  <PredictComponent />
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 pb-6 px-4 sm:px-8 mt-12">
        <div className="max-w-7xl mx-auto text-center text-green-200 text-sm">
          <p>© {new Date().getFullYear()} FarmAI Assistant | Advanced Leaf Disease Detection for Modern Farming</p>
        </div>
      </footer>

      {/* Removed Animations section */}

    </div>
  );
}
