// "use client"
// import { useState, useRef, useEffect, useMemo } from 'react';
// import Head from 'next/head';
// import Image from 'next/image';
// import {
//   Select,
//   SelectContent,
//   SelectItem,
//   SelectTrigger,
//   SelectValue,
// } from '@/components/ui/select';
// import { Send, Loader2, MessageSquare, Leaf, ChevronRight } from 'lucide-react';


// interface ContextItem {
//   query: string;
//   response: string | null;
// }

// interface ChatMessage {
//   role: 'user' | 'assistant';
//   content: string;
// }

// function generateRandomText(length = 50) {
//   const chars = 'abcdefghijklmnopqrstuvwxyz';
//   let result = '';
//   for (let i = 0; i < length; i++) {
//     result += chars.charAt(Math.floor(Math.random() * chars.length));
//   }
//   return result;
// }

// export default function ChatComponent() {
//   const [messages, setMessages] = useState<ChatMessage[]>([]);
//   const [input, setInput] = useState('');
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const session_id = useMemo(() => {
//     return generateRandomText();
//   }, []);
  
//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!input.trim()) return;

//     var userMessage = input;
//     setInput('');
    
//     setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
//     setIsLoading(true);

//     try {
//       const context: ContextItem[] = [{query:"", response: ""}];
//       for (let i = 0; i < messages.length; i += 2) {
//         if (i + 1 < messages.length) {
//           context.push({
//             query: messages[i].content,
//             response: messages[i + 1].content
//           });
//         }
//       }
      
//       const response = await fetch('http://127.0.0.1:5000/final', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           query: userMessage,
//           session_id: session_id,
//           language: 'english'
//         }),
//       });

//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }

//       var data = await response.json();
      
//       setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
//     } catch (error) {
//       console.error('Error:', error);
//       setMessages(prev => [
//         ...prev, 
//         { 
//           role: 'assistant', 
//           content: 'Sorry, I encountered an error processing your request. Please try again.' 
//         }
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="flex flex-col min-h-screen bg-[url('/agribg.jpg')] bg-cover bg-fixed">
//       <Head>
//         <title>KrishiGPT - Tomato Farming Assistant</title>
//         <meta name="description" content="AI assistant for tomato farmers in India" />
//       </Head> 

//       {/* Semi-transparent overlay */}
//       <div className="absolute inset-0 bg-gradient-to-b from-green-900/70 via-green-800/60 to-green-900/70 backdrop-blur-sm z-0"></div>

//       <header className="relative z-10 bg-green-700/90 shadow-lg border-b border-green-600">
//         <div className="container mx-auto flex items-center justify-between py-4 px-6">
//           <div className="flex items-center">
//             <div className="bg-white rounded-full p-2 mr-3 shadow-md">
//               <Leaf className="h-6 w-6 text-green-600" />
//             </div>
//             <div>
//               <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center">
//                 Smart Assistent
               
//               </h1>
//               <p className="text-green-100 text-sm">For Farming </p>
//             </div>
//           </div>
          
//           <nav className="hidden md:flex space-x-4">
//             <a href="/" className="text-green-100 hover:text-white text-sm transition-colors">Home</a>
//             <a href="/history" className="text-green-100 hover:text-white text-sm transition-colors">History</a>
//             <a href="/about" className="text-green-100 hover:text-white text-sm transition-colors">About</a>
//           </nav>
//         </div>
//       </header>

//       <main className="flex-grow container mx-auto p-4 md:p-6 flex flex-col relative z-10">
//         <div className="flex-grow bg-white/10 backdrop-blur-md rounded-xl shadow-xl p-4 mb-4 overflow-y-auto max-h-[70vh] border border-white/20">
//           {messages.length === 0 ? (
//             <div className="text-center text-white p-8">
//               <div className="mx-auto w-16 h-16 bg-green-600/30 rounded-full flex items-center justify-center mb-4">
//                 <MessageSquare className="h-8 w-8 text-green-100" />
//               </div>
              
//               <h2 className="text-2xl font-bold mb-3">Welcome to Smart Assistent!</h2>
//               <p className="mb-4 text-green-100">Your AI assistant for tomato farming in India.</p>
              
//               <div className="max-w-md mx-auto bg-white/10 rounded-lg p-4 backdrop-blur-sm">
//                 <p className="text-green-100">
//                   Ask me about tomato cultivation, disease management, or government schemes for farmers.
//                 </p>
//               </div>
              
//               <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-2 max-w-2xl mx-auto">
//                 <button 
//                   onClick={() => setInput("How do I identify  leaf diseases?")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Identify tomato diseases
//                 </button>
//                 <button 
//                   onClick={() => setInput("What are the best fertilizers ?")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Tomato fertilizers
//                 </button>
//                 <button 
//                   onClick={() => setInput("Government schemes for farmers")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Government schemes
//                 </button>
//               </div>
//             </div>
//           ) : (
//             <div className="space-y-4 p-2">
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`p-4 rounded-xl flex ${
//                     message.role === 'user'
//                       ? 'bg-green-600/20 border border-green-500/30 ml-auto max-w-[80%] justify-end text-white'
//                       : 'bg-white/20 border border-white/20 mr-auto max-w-[80%] text-white'
//                   }`}
//                 >
//                   {message.content}
//                 </div>
//               ))}
//               {isLoading && (
//                 <div className="bg-white/20 border border-white/20 rounded-xl p-4 mr-auto max-w-[80%] text-white">
//                   <div className="flex space-x-2">
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </div>
//           )}
//         </div>

//         <form onSubmit={handleSubmit} className="flex gap-3">
//           <input
//             type="text"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             className="flex-grow p-3 border border-white/30 bg-white/10 backdrop-blur-md text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-green-200"
//             placeholder="Ask about tomato farming..."
//             disabled={isLoading}
//           />
//           <button
//             type="submit"
//             className="bg-green-600 hover:bg-green-700 text-white p-3 rounded-xl flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-green-700/50 disabled:cursor-not-allowed transition-colors shadow-lg"
//             disabled={isLoading || !input.trim()}
//           >
//             {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
//           </button>
//         </form>
//       </main>

//       <footer className="relative z-10 bg-green-800/80 backdrop-blur-sm p-4 text-center text-green-100 text-sm border-t border-green-700/50">
//         <p>Smart Assistent - farming knowledge assistant for Indian farmers</p>
//       </footer>
//     </div>
//   );
// }


// "use client"
// import { useState, useRef, useEffect, useMemo } from 'react';
// import Head from 'next/head';
// import Image from 'next/image';
// import {
//   Select,
//   SelectContent,
//   SelectItem,
//   SelectTrigger,
//   SelectValue,
// } from '@/components/ui/select';
// import { Send, Loader2, MessageSquare, Leaf, ChevronRight, Mic, MicOff } from 'lucide-react';

// interface ContextItem {
//   query: string;
//   response: string | null;
// }

// interface ChatMessage {
//   role: 'user' | 'assistant';
//   content: string;
// }

// function generateRandomText(length = 50) {
//   const chars = 'abcdefghijklmnopqrstuvwxyz';
//   let result = '';
//   for (let i = 0; i < length; i++) {
//     result += chars.charAt(Math.floor(Math.random() * chars.length));
//   }
//   return result;
// }

// export default function ChatComponent() {
//   const [messages, setMessages] = useState<ChatMessage[]>([]);
//   const [input, setInput] = useState('');
//   const [isLoading, setIsLoading] = useState(false);
//   const [isRecording, setIsRecording] = useState(false);
//   const [recordingStatus, setRecordingStatus] = useState('');
//   const messagesEndRef = useRef<HTMLDivElement>(null);
//   const mediaRecorderRef = useRef<MediaRecorder | null>(null);
//   const audioChunksRef = useRef<Blob[]>([]);

//   const session_id = useMemo(() => {
//     return generateRandomText();
//   }, []);
  
//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   // Start voice recording
//   const startRecording = async () => {
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//       // mediaRecorderRef.current = new MediaRecorder(stream);
//       // Use a widely supported audio format
// mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
//       audioChunksRef.current = [];

//       mediaRecorderRef.current.ondataavailable = (event) => {
//         if (event.data.size > 0) {
//           audioChunksRef.current.push(event.data);
//         }
//       };

//       mediaRecorderRef.current.onstop = async () => {
//         const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
//         await processAudio(audioBlob);
        
//         // Stop all tracks to release microphone
//         stream.getTracks().forEach(track => track.stop());
//       };

//       mediaRecorderRef.current.start();
//       setIsRecording(true);
//       setRecordingStatus('Recording...');
//     } catch (error) {
//       console.error('Error accessing microphone:', error);
//       setRecordingStatus('Error: Cannot access microphone');
//     }
//   };

//   // Stop voice recording
//   const stopRecording = () => {
//     if (mediaRecorderRef.current && isRecording) {
//       mediaRecorderRef.current.stop();
//       setIsRecording(false);
//       setRecordingStatus('Processing audio...');
//     }
//   };

//   // Process the recorded audio
//   const processAudio = async (audioBlob: Blob) => {
//     try {
//       setIsLoading(true);
      
//       // Create form data to send audio file
//       const formData = new FormData();
//       formData.append('audio', audioBlob, 'recording.wav');
//       formData.append('session_id', session_id);

//       // Send audio to backend
//       const response = await fetch('http://127.0.0.1:5000/speech_to_text', {
//         method: 'POST',
//         body: formData,
//       });

//       if (!response.ok) {
//         throw new Error('Error processing speech');
//       }

//       const data = await response.json();
      
//       if (data.text) {
//         // Add the transcribed text to the input field
//         setInput(data.text);
        
//         // Automatically submit if there's text
//         await submitMessage(data.text);
//       } else {
//         setRecordingStatus('Could not understand audio');
//         setTimeout(() => setRecordingStatus(''), 3000);
//       }
//     } catch (error) {
//       console.error('Error processing audio:', error);
//       setRecordingStatus('Error processing audio');
//       setTimeout(() => setRecordingStatus(''), 3000);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   // Submit a message (can be called from text input or after voice processing)
//   const submitMessage = async (messageText: string) => {
//     if (!messageText.trim()) return;
    
//     setMessages(prev => [...prev, { role: 'user', content: messageText }]);
//     setInput('');
//     setIsLoading(true);

//     try {
//       const context: ContextItem[] = [{query:"", response: ""}];
//       for (let i = 0; i < messages.length; i += 2) {
//         if (i + 1 < messages.length) {
//           context.push({
//             query: messages[i].content,
//             response: messages[i + 1].content
//           });
//         }
//       }
      
//       const response = await fetch('http://127.0.0.1:5000/final', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           query: messageText,
//           session_id: session_id,
//           language: 'english'
//         }),
//       });

//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }

//       var data = await response.json();
      
//       setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
//     } catch (error) {
//       console.error('Error:', error);
//       setMessages(prev => [
//         ...prev, 
//         { 
//           role: 'assistant', 
//           content: 'Sorry, I encountered an error processing your request. Please try again.' 
//         }
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     await submitMessage(input);
//   };

//   return (
//     <div className="flex flex-col min-h-screen bg-[url('/agribg.jpg')] bg-cover bg-fixed">
//       <Head>
//         <title>KrishiGPT - Tomato Farming Assistant</title>
//         <meta name="description" content="AI assistant for tomato farmers in India" />
//       </Head> 

//       {/* Semi-transparent overlay */}
//       <div className="absolute inset-0 bg-gradient-to-b from-green-900/70 via-green-800/60 to-green-900/70 backdrop-blur-sm z-0"></div>

//       <header className="relative z-10 bg-green-700/90 shadow-lg border-b border-green-600">
//         <div className="container mx-auto flex items-center justify-between py-4 px-6">
//           <div className="flex items-center">
//             <div className="bg-white rounded-full p-2 mr-3 shadow-md">
//               <Leaf className="h-6 w-6 text-green-600" />
//             </div>
//             <div>
//               <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center">
//                 Smart Assistent
               
//               </h1>
//               <p className="text-green-100 text-sm">For Farming </p>
//             </div>
//           </div>
          
//           <nav className="hidden md:flex space-x-4">
//             <a href="/" className="text-green-100 hover:text-white text-sm transition-colors">Home</a>
//             <a href="/history" className="text-green-100 hover:text-white text-sm transition-colors">History</a>
//             <a href="/about" className="text-green-100 hover:text-white text-sm transition-colors">About</a>
//           </nav>
//         </div>
//       </header>

//       <main className="flex-grow container mx-auto p-4 md:p-6 flex flex-col relative z-10">
//         <div className="flex-grow bg-white/10 backdrop-blur-md rounded-xl shadow-xl p-4 mb-4 overflow-y-auto max-h-[70vh] border border-white/20">
//           {messages.length === 0 ? (
//             <div className="text-center text-white p-8">
//               <div className="mx-auto w-16 h-16 bg-green-600/30 rounded-full flex items-center justify-center mb-4">
//                 <MessageSquare className="h-8 w-8 text-green-100" />
//               </div>
              
//               <h2 className="text-2xl font-bold mb-3">Welcome to Smart Assistent!</h2>
//               <p className="mb-4 text-green-100">Your AI assistant for tomato farming in India.</p>
              
//               <div className="max-w-md mx-auto bg-white/10 rounded-lg p-4 backdrop-blur-sm">
//                 <p className="text-green-100">
//                   Ask me about tomato cultivation, disease management, or government schemes for farmers.
//                 </p>
//               </div>
              
//               <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-2 max-w-2xl mx-auto">
//                 <button 
//                   onClick={() => setInput("How do I identify leaf diseases?")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Identify tomato diseases
//                 </button>
//                 <button 
//                   onClick={() => setInput("What are the best fertilizers?")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Tomato fertilizers
//                 </button>
//                 <button 
//                   onClick={() => setInput("Government schemes for farmers")}
//                   className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
//                 >
//                   Government schemes
//                 </button>
//               </div>
//             </div>
//           ) : (
//             <div className="space-y-4 p-2">
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`p-4 rounded-xl flex ${
//                     message.role === 'user'
//                       ? 'bg-green-600/20 border border-green-500/30 ml-auto max-w-[80%] justify-end text-white'
//                       : 'bg-white/20 border border-white/20 mr-auto max-w-[80%] text-white'
//                   }`}
//                 >
//                   {message.content}
//                 </div>
//               ))}
//               {isLoading && (
//                 <div className="bg-white/20 border border-white/20 rounded-xl p-4 mr-auto max-w-[80%] text-white">
//                   <div className="flex space-x-2">
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
//                     <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </div>
//           )}
//         </div>

//         {recordingStatus && (
//           <div className="bg-white/20 rounded-lg p-2 mb-2 text-center text-white text-sm">
//             {recordingStatus}
//           </div>
//         )}

//         <form onSubmit={handleSubmit} className="flex gap-3">
//           <input
//             type="text"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             className="flex-grow p-3 border border-white/30 bg-white/10 backdrop-blur-md text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-green-200"
//             placeholder="Ask about tomato farming..."
//             disabled={isLoading || isRecording}
//           />
          

        

//           {/* Send Button */}
//           <button
//             type="submit"
//             className="bg-green-600 hover:bg-green-700 text-white p-3 rounded-xl flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-green-700/50 disabled:cursor-not-allowed transition-colors shadow-lg"
//             disabled={isLoading || isRecording || !input.trim()}
//           >
//             {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
//           </button>
//         </form>
//       </main>

//       <footer className="relative z-10 bg-green-800/80 backdrop-blur-sm p-4 text-center text-green-100 text-sm border-t border-green-700/50">
//         <p>Smart Assistent - farming knowledge assistant for Indian farmers</p>
//       </footer>
//     </div>
//   );
// }



"use client"
import { useState, useRef, useEffect, useMemo } from 'react';
import Head from 'next/head';
import Image from 'next/image';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Send, Loader2, MessageSquare, Leaf, ChevronRight } from 'lucide-react';
import { Mic, MicOff } from 'lucide-react';


declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}




interface ContextItem {
  query: string;
  response: string | null;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

function generateRandomText(length = 50) {
  const chars = 'abcdefghijklmnopqrstuvwxyz';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export default function ChatComponent() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const session_id = useMemo(() => {
    return generateRandomText();
  }, []);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    var userMessage = input;
    setInput('');
    
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const context: ContextItem[] = [{query:"", response: ""}];
      for (let i = 0; i < messages.length; i += 2) {
        if (i + 1 < messages.length) {
          context.push({
            query: messages[i].content,
            response: messages[i + 1].content
          });
        }
      }
      
      const response = await fetch('http://127.0.0.1:5000/final', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          session_id: session_id,
          language: 'english'
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      var data = await response.json();
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: 'Sorry, I encountered an error processing your request. Please try again.' 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

useEffect(() => {
  if (typeof window !== 'undefined') {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'mr-IN'; // Marathi
      recognition.interimResults = false;

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript); // Populate input with voice text
        setIsListening(false);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error', event.error);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }
}, []);

  const handleMicClick = () => {
    if (recognitionRef.current) {
      if (isListening) {
        recognitionRef.current.stop();
        setIsListening(false);
      }else {
        recognitionRef.current.start();
        setIsListening(true);
      }
    }
  };


  return (
    <div className="flex flex-col min-h-screen bg-[url('/agribg.jpg')] bg-cover bg-fixed">
      <Head>
        <title>KrishiGPT - Diseased Crop Assistant</title>
        <meta name="description" content="AI assistant for diseased crop in India" />
      </Head> 

      {/* Semi-transparent overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-green-900/70 via-green-800/60 to-green-900/70 backdrop-blur-sm z-0"></div>

      <header className="relative z-10 bg-green-700/90 shadow-lg border-b border-green-600">
        <div className="container mx-auto flex items-center justify-between py-4 px-6">
          <div className="flex items-center">
            <div className="bg-white rounded-full p-2 mr-3 shadow-md">
              <Leaf className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center">
                Smart Assistent
               
              </h1>
              <p className="text-green-100 text-sm">For Farming </p>
            </div>
          </div>
          
          <nav className="hidden md:flex space-x-4">
            <a href="/" className="text-green-100 hover:text-white text-sm transition-colors">Home</a>
            <a href="/history" className="text-green-100 hover:text-white text-sm transition-colors">History</a>
            <a href="/about" className="text-green-100 hover:text-white text-sm transition-colors">About</a>
          </nav>
        </div>
      </header>

      <main className="flex-grow container mx-auto p-4 md:p-6 flex flex-col relative z-10">
        <div className="flex-grow bg-white/10 backdrop-blur-md rounded-xl shadow-xl p-4 mb-4 overflow-y-auto max-h-[70vh] border border-white/20">
          {messages.length === 0 ? (
            <div className="text-center text-white p-8">
              <div className="mx-auto w-16 h-16 bg-green-600/30 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="h-8 w-8 text-green-100" />
              </div>
              
              <h2 className="text-2xl font-bold mb-3">Welcome to Smart Assistent!</h2>
              <p className="mb-4 text-green-100">Your AI assistant for diseased crop in India.</p>
              
              <div className="max-w-md mx-auto bg-white/10 rounded-lg p-4 backdrop-blur-sm">
                <p className="text-green-100">
                  Ask me about cultivation, disease management, or government schemes for farmers.
                </p>
              </div>
              
              <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-2 max-w-2xl mx-auto">
                <button 
                  onClick={() => setInput("How do I identify leaf diseases?")}
                  className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
                >
                  Identify diseases
                </button>
                <button 
                  onClick={() => setInput("What are the best fertilizers ?")}
                  className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
                >
                  fertilizers
                </button>
                <button 
                  onClick={() => setInput("Government schemes for farmers")}
                  className="bg-green-700/40 hover:bg-green-700/60 text-white text-sm py-2 px-3 rounded-lg transition-colors border border-green-500/30 backdrop-blur-sm"
                >
                  Government schemes
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4 p-2">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-xl flex ${
                    message.role === 'user'
                      ? 'bg-green-600/20 border border-green-500/30 ml-auto max-w-[80%] justify-end text-white'
                      : 'bg-white/20 border border-white/20 mr-auto max-w-[80%] text-white'
                  }`}
                >
                  {message.content}
                </div>
              ))}
              {isLoading && (
                <div className="bg-white/20 border border-white/20 rounded-xl p-4 mr-auto max-w-[80%] text-white">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    <div className="w-2 h-2 bg-green-300 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="flex items-center flex-grow relative">
          <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-3 border border-white/30 bg-white/10 backdrop-blur-md text-white rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-green-200"
          placeholder="Speak or type your question..."
          disabled={isLoading}
        />
        <button
          type="button"
          onClick={handleMicClick}
          className={`absolute right-2 p-1 rounded-full ${
            isListening ? 'bg-red-600' : 'bg-green-500'
          }`}
        >
            {isListening ? <MicOff className="h-4 w-4 text-white" /> : <Mic className="h-4 w-4 text-white" />}
          </button>
        </div>

          <button
            type="submit"
            className="bg-green-600 hover:bg-green-700 text-white p-3 rounded-xl flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-green-700/50 disabled:cursor-not-allowed transition-colors shadow-lg"
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
          </button>
        </form>
      </main>

      <footer className="relative z-10 bg-green-800/80 backdrop-blur-sm p-4 text-center text-green-100 text-sm border-t border-green-700/50">
        <p>Smart Assistent - farming knowledge assistant for Indian farmers</p>
      </footer>
    </div>
  );
}
