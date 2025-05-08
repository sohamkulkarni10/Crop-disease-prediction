"use client"
import { useState } from 'react';
import { Leaf, History, MessageSquare, ChevronRight, ArrowRight } from 'lucide-react';
import { useRouter } from 'next/navigation'

export default function FarmAssistantLanding() {
  const [hoverFeature, setHoverFeature] = useState(null);
  const router=useRouter()
  const features = [
    {
      id: 'leaf-analysis',
      title: 'Leaf Disease Detection',
      description: 'Upload a photo of any plant leaf and get instant disease identification with treatment recommendations.',
      icon: <Leaf className="h-10 w-10 text-green-600" />,
      image: '/disease.jpg',
      alt: 'Leaf disease detection illustration',
      buttonText: 'Identify Disease',
      link:"/prediction"
    },
    {
      id: 'history-analysis',
      title: 'Historical Analysis',
      description: 'View past diagnoses, track disease patterns over time, and access insights based on seasonal data.',
      icon: <History className="h-10 w-10 text-green-600" />,
      image: '/historymain.jpg',
      alt: 'Historical analysis dashboard',
      buttonText: 'View History',
      link:"/history"
    },
    {
      id: 'ai-chatbot',
      title: 'AI Farming Assistant',
      description: 'Chat with our agricultural AI expert for personalized advice on crop management, pest control, and more.',
      icon: <MessageSquare className="h-10 w-10 text-green-600" />,
      image: 'chatbot.jpg',
      alt: 'AI chatbot for farmers',
      buttonText: 'Start Chatting',
      link:"/chatbot"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-6 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Leaf className="h-8 w-8 text-green-600" />
            <h1 className="text-2xl font-bold text-green-800">FarmAI Assistant</h1>
          </div>
          <nav>
            <ul className="flex gap-6">
              <li><a href="#features" className="text-green-700 hover:text-green-500">Features</a></li>
              <li><a href="#about" className="text-green-700 hover:text-green-500">About</a></li>
              <li><a href="#contact" className="text-green-700 hover:text-green-500">Contact</a></li>
            </ul>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center max-w-4xl">
          <h2 className="text-4xl md:text-5xl font-bold text-green-800 mb-6">Smart AI Solutions for Modern Farmers</h2>
          <p className="text-xl text-gray-600 mb-10">Identify plant diseases, analyze patterns, and get expert advice with our all-in-one farming assistant powered by artificial intelligence.</p>
          <a 
            href="#features" 
            className="inline-flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-6 rounded-lg transition-colors"
          >
            Explore Features <ChevronRight className="h-5 w-5" />
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16 px-4 bg-white">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center text-green-800 mb-16">Our Features</h2>
          
          <div className="grid md:grid-cols-3 gap-8 cursor-pointer">
            {features.map((feature) => (
              <div 
                key={feature.id}
                className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-100 transition-all duration-300 hover:shadow-xl"
                // onMouseEnter={() => setHoverFeature(feature.id)}
                // onMouseLeave={() => setHoverFeature(null)}
              >
                <div className="h-48 bg-gray-100 relative overflow-hidden cursor-pointer">
                  <img 
                    src={feature.image} 
                    alt={feature.alt}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end">
                    <div className="p-4">
                      <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        {feature.icon}
                        {feature.title}
                      </h3>
                    </div>
                  </div>
                </div>
                
                <div className="p-6">
                  <p className="text-gray-600 mb-6">{feature.description}</p>
                  <button 
                    className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium transition-colors cursor-pointer ${
                      hoverFeature === feature.id 
                        ? 'bg-green-600 text-white' 
                        : 'bg-green-100 text-green-700 hover:bg-green-200'
                    }`}
                    onClick={() => router.push(`${feature.link}`)}
                  >
                    {feature.buttonText}
                    <ArrowRight className={`h-5 w-5 transition-transform cursor-pointer ${
                      hoverFeature === feature.id ? 'translate-x-1' : ''
                    }`} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 bg-green-50">
        <div className="container mx-auto text-center max-w-3xl">
          <h2 className="text-3xl font-bold text-green-800 mb-6">Ready to Transform Your Farming?</h2>
          <p className="text-lg text-gray-600 mb-8">Join thousands of farmers who are using AI to improve crop yields, reduce losses, and farm more sustainably.</p>
          <button className="bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-8 rounded-lg transition-colors cursor-pointer">
            Get Started Today
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-green-800 text-white py-12 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-6 md:mb-0">
              <Leaf className="h-8 w-8" />
              <h2 className="text-xl font-bold">FarmAI Assistant</h2>
            </div>
            <div className="flex gap-8">
              <div>
                <h3 className="font-semibold mb-2">Features</h3>
                <ul className="space-y-1">
                  <li>Leaf Disease Detection</li>
                  <li>Historical Analysis</li>
                  <li>AI Chatbot</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Support</h3>
                <ul className="space-y-1">
                  <li>Help Center</li>
                  <li>Contact Us</li>
                  <li>Privacy Policy</li>
                </ul>
              </div>
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-green-700 text-center text-green-300">
            <p>&copy; {new Date().getFullYear()} FarmAI Assistant. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}