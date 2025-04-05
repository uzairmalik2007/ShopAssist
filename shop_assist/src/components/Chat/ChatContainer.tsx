// components/Chat/ChatContainer.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Message } from '../../types';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { SendHorizontal } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: uuidv4(),
      type: 'user',
      content,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:9000/Chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          session_id: 'user123'
        })
      });

      const data = await response.json();

      const botMessage: Message = {
        id: uuidv4(),
        type: 'bot',
        content: data.response,
        timestamp: new Date(),
        recommendations: data.recommendations,
        intent: data.intent
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">


      <div className="flex-grow overflow-hidden bg-white shadow-xl rounded-lg m-4">
        <div className="flex flex-col h-full">
          <div className="flex-grow overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {loading && (
              <div className="flex items-center space-x-2">
                <div className="animate-bounce h-2 w-2 bg-gray-500 rounded-full"></div>
                <div className="animate-bounce h-2 w-2 bg-gray-500 rounded-full delay-100"></div>
                <div className="animate-bounce h-2 w-2 bg-gray-500 rounded-full delay-200"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="flex-none p-4 bg-gray-50">
            <ChatInput onSend={handleSendMessage} isLoading={loading} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatContainer;