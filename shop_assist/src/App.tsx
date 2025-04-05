
// App.tsx
import React from 'react';
import ChatContainer from './components/Chat/ChatContainer';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <header className="py-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-gray-900">ShopAssist</h1>
            <p className="text-gray-500">AI Laptop Shopping Assistant</p>
          </div>
        </header>
        <main>
          <ChatContainer />
        </main>
      </div>
    </div>
  );
};

export default App;
