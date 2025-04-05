// components/Chat/ChatMessage.tsx
import React from 'react';
import { Message } from '../../types';
import LaptopCard from './LaptopCard';
import { User, Bot } from 'lucide-react';

interface Props {
  message: Message;
}

const ChatMessage: React.FC<Props> = ({ message }) => {
  const isBot = message.type === 'bot';

  const formatMessage = (content: string) => {
    const sections = content.split('\n').map((line, index) => {
      // Handle section headers
      if (line.match(/^\*\*\d+\./)) {
        return (
          <h3 key={index} className="font-bold text-lg mt-4 mb-2">
            {line.replace(/\*\*/g, '')}
          </h3>
        );
      }

      // Handle bullet points
      if (line.startsWith('- ')) {
        const bulletContent = line.substring(2);
        // Process bold text within bullet points
        const parts = bulletContent.split('**').map((part, pIndex) => {
          return pIndex % 2 === 1 ? (
            <span key={pIndex} className="font-semibold">{part}</span>
          ) : part;
        });

        return (
          <div key={index} className="ml-4 my-2 flex">
            <span className="mr-2">•</span>
            <span>{parts}</span>
          </div>
        );
      }

      // Process bold text in regular paragraphs
      if (line.includes('**')) {
        const parts = line.split('**').map((part, pIndex) => {
          return pIndex % 2 === 1 ? (
            <span key={pIndex} className="font-semibold">{part}</span>
          ) : part;
        });
        return <p key={index} className="my-2">{parts}</p>;
      }

      // Regular text
      return line.trim() ? <p key={index} className="my-2">{line}</p> : null;
    });

    return <div className="space-y-1">{sections}</div>;
  };

  return (
    <div className={`flex items-start space-x-4 ${isBot ? '' : 'flex-row-reverse space-x-reverse'}`}>
      <div className={`flex-none rounded-full p-2 ${isBot ? 'bg-blue-500' : 'bg-gray-200'}`}>
        {isBot ? (
          <Bot className="h-6 w-6 text-white" />
        ) : (
          <User className="h-6 w-6 text-gray-600" />
        )}
      </div>

      <div className={`flex flex-col space-y-2 max-w-3xl ${isBot ? 'items-start' : 'items-end'}`}>
        <div className={`rounded-lg p-4 w-full ${
          isBot ? 'bg-white border border-gray-200 shadow-sm' : 'bg-blue-500 text-white'
        }`}>
          <div className="text-sm">
            {formatMessage(message.content)}
          </div>
        </div>

        {isBot && message.recommendations && message.recommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full mt-4">
            {message.recommendations.map((laptop, index) => (
              <LaptopCard key={index} laptop={laptop} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;